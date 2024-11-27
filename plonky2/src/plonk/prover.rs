//! plonky2 prover implementation.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::cmp::min;
use core::mem::swap;

use anyhow::{ensure, Result};
use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use super::circuit_builder::{LookupChallenges, LookupWire};
use crate::field::extension::Extendable;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::field::types::Field;
use crate::field::zero_poly_coset::ZeroPolyOnCoset;
use crate::fri::oracle::PolynomialBatch;
use crate::gates::lookup::LookupGate;
use crate::gates::lookup_table::LookupTableGate;
use crate::gates::selectors::LookupSelectors;
use crate::hash::hash_types::RichField;
use crate::iop::challenger::Challenger;
use crate::iop::generator::generate_partial_witness;
use crate::iop::target::Target;
use crate::iop::witness::{MatrixWitness, PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::NUM_COINS_LOOKUP;
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::plonk_common::PlonkOracle;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};
use crate::plonk::vanishing_poly::{eval_vanishing_poly_base_batch, get_lut_poly};
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::timed;
use crate::util::partial_products::{partial_products_and_z_gx, quotient_chunk_products};
use crate::util::timing::TimingTree;
use crate::util::{log2_ceil, transpose};

/// Set all the lookup gate wires (including multiplicities) and pad unused LU slots.
/// Warning: rows are in descending order: the first gate to appear is the last LU gate, and
/// the last gate to appear is the first LUT gate.
pub fn set_lookup_wires<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    pw: &mut PartitionWitness<F>,
) -> Result<()> {
    for (
        lut_index,
        &LookupWire {
            last_lu_gate: _,
            last_lut_gate,
            first_lut_gate,
        },
    ) in prover_data.lookup_rows.iter().enumerate()
    {
        let lut_len = common_data.luts[lut_index].len();
        let num_entries = LookupGate::num_slots(&common_data.config);
        let num_lut_entries = LookupTableGate::num_slots(&common_data.config);

        // Compute multiplicities.
        let mut multiplicities = vec![0; lut_len];

        let table_value_to_idx: HashMap<u16, usize> = common_data.luts[lut_index]
            .iter()
            .enumerate()
            .map(|(i, (inp_target, _))| (*inp_target, i))
            .collect();

        for (inp_target, _) in prover_data.lut_to_lookups[lut_index].iter() {
            let inp_value = pw.get_target(*inp_target);
            let idx = table_value_to_idx
                .get(&u16::try_from(inp_value.to_canonical_u64()).unwrap())
                .unwrap();

            multiplicities[*idx] += 1;
        }

        // Pad the last `LookupGate` with the first entry from the LUT.
        let remaining_slots = (num_entries
            - (prover_data.lut_to_lookups[lut_index].len() % num_entries))
            % num_entries;
        let (first_inp_value, first_out_value) = common_data.luts[lut_index][0];
        for slot in (num_entries - remaining_slots)..num_entries {
            let inp_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_inp(slot));
            let out_target =
                Target::wire(last_lut_gate - 1, LookupGate::wire_ith_looking_out(slot));
            pw.set_target(inp_target, F::from_canonical_u16(first_inp_value))?;
            pw.set_target(out_target, F::from_canonical_u16(first_out_value))?;

            multiplicities[0] += 1;
        }

        // We don't need to pad the last `LookupTableGate`; extra wires are set to 0 by default, which satisfies the constraints.
        for lut_entry in 0..lut_len {
            let row = first_lut_gate - lut_entry / num_lut_entries;
            let col = lut_entry % num_lut_entries;

            let mul_target = Target::wire(row, LookupTableGate::wire_ith_multiplicity(col));

            pw.set_target(
                mul_target,
                F::from_canonical_usize(multiplicities[lut_entry]),
            )?;
        }
    }

    Ok(())
}

pub fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    inputs: PartialWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let partition_witness = timed!(
        timing,
        &format!("run {} generators", prover_data.generators.len()),
        generate_partial_witness(inputs, prover_data, common_data)?
    );
    //println!("partition_witness:{:?}", partition_witness);

    prove_with_partition_witness(prover_data, common_data, partition_witness, timing)
}

pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    mut partition_witness: PartitionWitness<F>,
    timing: &mut TimingTree,
) -> Result<ProofWithPublicInputs<F, C, D>>
where
    C::Hasher: Hasher<F>,
    C::InnerHasher: Hasher<F>,
{
    let has_lookup = !common_data.luts.is_empty();
    let config = &common_data.config;
    let num_challenges = config.num_challenges;//2
    let quotient_degree = common_data.quotient_degree();//32
    let degree = common_data.degree();//4

    set_lookup_wires(prover_data, common_data, &mut partition_witness)?;
    //prover_data.public_inputs:[VirtualTarget { index: 0 }, VirtualTarget { index: 1 }, Wire(Wire { row: 0, column: 15 })]
    let public_inputs = partition_witness.get_targets(&prover_data.public_inputs);
    //public_inputs:[0, 1, 5]
    let public_inputs_hash = C::InnerHasher::hash_no_pad(&public_inputs);
    //HashOut { elements: [12460551030817792791, 6203763534542844149, 15133388778355119947, 8532039303907884673]
    //获取135*4个wire的所有的实际数据，每一列4个数据组成一个wire_values的数据
    let witness = timed!(
        timing,
        "compute full witness",
        partition_witness.full_witness()
    );
    //[0, 0, 12460551030817792791, 0], [1, 1, 6203763534542844149, 1], [1, 5, 15133388778355119947, 0]

    let wires_values: Vec<PolynomialValues<F>> = timed!(
        timing,
        "compute wire polynomials",
        witness
            .wire_values
            .par_iter()
            .map(|column| PolynomialValues::new(column.clone()))
            .collect()
    );
    //println!("wires_values:{:?}",wires_values);
    //PolynomialValues { values: [0, 0, 12460551030817792791, 0] }, PolynomialValues { values: [1, 1, 6203763534542844149, 1] }, PolynomialValues { values: [1, 5, 15133388778355119947, 0] },

    let wires_commitment = timed!(
        timing,
        "compute wires commitment",
        PolynomialBatch::<F, C, D>::from_values(
            wires_values,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::WIRES.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    let mut challenger = Challenger::<F, C::Hasher>::new();

    // Observe the instance.
    challenger.observe_hash::<C::Hasher>(prover_data.circuit_digest);
    challenger.observe_hash::<C::InnerHasher>(public_inputs_hash);

    challenger.observe_cap::<C::Hasher>(&wires_commitment.merkle_tree.cap);

    // We need 4 values per challenge: 2 for the combos, 1 for (X-combo) in the accumulators and 1 to prove that the lookup table was computed correctly.
    // We can reuse betas and gammas for two of them.
    // 每个挑战需要4个值：2个用于组合，1个用于累加器中的(X-combo)，1个用于验证查找表的正确性。
    // 我们可以重用betas和gammas中的两个。
    let num_lookup_challenges = NUM_COINS_LOOKUP * num_challenges;

    // 获取num_challenges个挑战值作为betas
    let betas = challenger.get_n_challenges(num_challenges);
    // 获取num_challenges个挑战值作为gammas
    let gammas = challenger.get_n_challenges(num_challenges);

    // 如果有查找表，则需要额外的挑战值
    let deltas = if has_lookup {
        // 创建一个容量为2*num_challenges的向量来存储deltas
        let mut delts = Vec::with_capacity(2 * num_challenges);
        // 计算需要的额外挑战值数量
        let num_additional_challenges = num_lookup_challenges - 2 * num_challenges;
        // 获取额外的挑战值
        let additional = challenger.get_n_challenges(num_additional_challenges);
        // 将betas和gammas扩展到deltas中
        delts.extend(&betas);
        delts.extend(&gammas);
        // 将额外的挑战值扩展到deltas中
        delts.extend(additional);
        delts
    } else {
        // 如果没有查找表，则deltas为空
        vec![]
    };
    assert!(
        common_data.quotient_degree_factor < common_data.config.num_routed_wires,
        "When the number of routed wires is smaller that the degree, we should change the logic to avoid computing partial products."
    );
    //置换证明中，Multiset 等价证明，计算r1=z0，r2=z0*z1=r1z1，r3=z0*z1*z2=r2*z2,……，rn=z0*z1*…*z(n-1)=r(n-1)*zn,得到r1,r2,…,rn，2个挑战因子，需要生成两组
    let mut partial_products_and_zs = timed!(
        timing,
        "compute partial products",
        all_wires_permutation_partial_products(&witness, &betas, &gammas, prover_data, common_data)
    );
    //println!("partial_products_and_z:{:?}",partial_products_and_zs);

    // Z is expected at the front of our batch; see `zs_range` and `partial_products_range`.
    let plonk_z_vecs = partial_products_and_zs
        .iter_mut()
        .map(|partial_products_and_z| partial_products_and_z.pop().unwrap())
        .collect();
    //partial_products_and_zs最后一个vec（4个元素）拿到最前面
    let zs_partial_products = [plonk_z_vecs, partial_products_and_zs.concat()].concat();
    //println!("zs_partial_products:{:?}",zs_partial_products);

    // All lookup polys: RE and partial SLDCs.
    let lookup_polys =
        compute_all_lookup_polys(&witness, &deltas, prover_data, common_data, has_lookup);

    let zs_partial_products_lookups = if has_lookup {
        [zs_partial_products, lookup_polys].concat()
    } else {
        zs_partial_products
    };

    let partial_products_zs_and_lookup_commitment = timed!(
        timing,
        "commit to partial products, Z's and, if any, lookup polynomials",
        PolynomialBatch::from_values(
            zs_partial_products_lookups,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::ZS_PARTIAL_PRODUCTS.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&partial_products_zs_and_lookup_commitment.merkle_tree.cap);

    let alphas = challenger.get_n_challenges(num_challenges);

    let quotient_polys = timed!(
        timing,
        "compute quotient polys",
        compute_quotient_polys::<F, C, D>(
            common_data,
            prover_data,
            &public_inputs_hash,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &betas,
            &gammas,
            &deltas,
            &alphas,
        )
    );

    let all_quotient_poly_chunks: Vec<PolynomialCoeffs<F>> = timed!(
        timing,
        "split up quotient polys",
        quotient_polys
            .into_par_iter()
            .flat_map(|mut quotient_poly| {
                quotient_poly.trim_to_len(quotient_degree).expect(
                    "Quotient has failed, the vanishing polynomial is not divisible by Z_H",
                );
                // Split quotient into degree-n chunks.
                quotient_poly.chunks(degree)
            })
            .collect()
    );

    let quotient_polys_commitment = timed!(
        timing,
        "commit to quotient polys",
        PolynomialBatch::<F, C, D>::from_coeffs(
            all_quotient_poly_chunks,
            config.fri_config.rate_bits,
            config.zero_knowledge && PlonkOracle::QUOTIENT.blinding,
            config.fri_config.cap_height,
            timing,
            prover_data.fft_root_table.as_ref(),
        )
    );

    challenger.observe_cap::<C::Hasher>(&quotient_polys_commitment.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();
    // To avoid leaking witness data, we want to ensure that our opening locations, `zeta` and
    // `g * zeta`, are not in our subgroup `H`. It suffices to check `zeta` only, since
    // `(g * zeta)^n = zeta^n`, where `n` is the order of `g`.
    let g = F::Extension::primitive_root_of_unity(common_data.degree_bits());
    ensure!(
        zeta.exp_power_of_2(common_data.degree_bits()) != F::Extension::ONE,
        "Opening point is in the subgroup."
    );

    let openings = timed!(
        timing,
        "construct the opening set, including lookups",
        OpeningSet::new(
            zeta,
            g,
            &prover_data.constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_commitment,
            common_data
        )
    );
    challenger.observe_openings(&openings.to_fri_openings());
    let instance = common_data.get_fri_instance(zeta);

    let opening_proof = timed!(
        timing,
        "compute opening proofs",
        PolynomialBatch::<F, C, D>::prove_openings(
            &instance,
            &[
                &prover_data.constants_sigmas_commitment,
                &wires_commitment,
                &partial_products_zs_and_lookup_commitment,
                &quotient_polys_commitment,
            ],
            &mut challenger,
            &common_data.fri_params,
            timing,
        )
    );

    let proof = Proof::<F, C, D> {
        wires_cap: wires_commitment.merkle_tree.cap,
        plonk_zs_partial_products_cap: partial_products_zs_and_lookup_commitment.merkle_tree.cap,
        quotient_polys_cap: quotient_polys_commitment.merkle_tree.cap,
        openings,
        opening_proof,
    };
    Ok(ProofWithPublicInputs::<F, C, D> {
        proof,
        public_inputs,
    })
}

///Multiset 等价证明，计算r1=z0，r2=z0*z1=r1z1，r3=z0*z1*z2=r2*z2,……，rn=z0*z1*…*z(n-1)=r(n-1)*zn,得到r1,r2,…,rn
fn all_wires_permutation_partial_products<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>, // 见证矩阵
    betas: &[F], // β 值数组
    gammas: &[F], // γ 值数组
    prover_data: &ProverOnlyCircuitData<F, C, D>, // 仅用于证明者的电路数据
    common_data: &CommonCircuitData<F, D>, // 通用电路数据
) -> Vec<Vec<PolynomialValues<F>>> {
    // common_data.config.num_challenges=2
    // 遍历所有挑战次数
    (0..common_data.config.num_challenges)
        .map(|i| {
            // 计算每个挑战的部分积和 Z 多项式
            wires_permutation_partial_products_and_zs(
                witness,
                betas[i],
                gammas[i],
                prover_data,
                common_data,
            )
        })
        .collect() // 收集结果为向量
}

/// Multiset 等价证明，计算r1=z0，r2=z0*z1=r1z1，r3=z0*z1*z2=r2*z2,……，rn=z0*z1*…*z(n-1)=r(n-1)*zn,得到r1,r2,…,rn
/// Z(g^i) = f / g。
/// Compute the partial products used in the `Z` polynomial.
/// Returns the polynomials interpolating `partial_products(f / g)`
/// where `f, g` are the products in the definition of `Z`: `Z(g^i) = f / g`.
fn wires_permutation_partial_products_and_zs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>, // 见证矩阵
    beta: F, // β 值
    gamma: F, // γ 值
    prover_data: &ProverOnlyCircuitData<F, C, D>, // 仅用于证明者的电路数据
    common_data: &CommonCircuitData<F, D>, // 通用电路数据
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.quotient_degree_factor; // 商多项式的度数因子=8
    let subgroup = &prover_data.subgroup; // 子群，degree=4
    let k_is = &common_data.k_is; // 陪集，size=80
    let num_prods = common_data.num_partial_products; // 部分积的数量=9

    // 进行置换证明时，计算两个Multiset a跟b置换等价，需要计算numerators=a+beta*i+gamma，denominators=b+beta*sigma(i)+gamma,这两个集合是根等价集合。
    //subgroup[1, 281474976710656, 18446744069414584320, 18446462594437873665]

    let all_quotient_chunk_products = subgroup
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let s_sigmas = &prover_data.sigmas[i]; // σ 值数组
            //s_sigmas:[281474976710656, 15817657382918473249, 18196947516708736925, 13907817722130613464, 17417240021601665567, 3300053643433736875, 6203009912824666656, 6744005316595460895
            // 计算分子f=wire_value + beta * s_id + gamma
            let numerators = (0..common_data.config.num_routed_wires).map(|j| {
                let wire_value = witness.get_wire(i, j); // 获取线值
                let k_i = k_is[j]; // 获取 第j个陪集的值
                let s_id = k_i * x; // x为子群的值，此处使用子群的值乘以陪集的值，没有使用wire的坐标序号，是用了优化算法。
                wire_value + beta * s_id + gamma // 计算分子
            });
            // 计算分母g=wire_value + beta * s_sigma + gamma
            let denominators = (0..common_data.config.num_routed_wires)
                .map(|j| {
                    let wire_value = witness.get_wire(i, j); // 获取线值
                    let s_sigma = s_sigmas[j]; // 获取 σ 值
                    wire_value + beta * s_sigma + gamma // 计算分母
                })
                .collect::<Vec<_>>();
            let denominator_invs = F::batch_multiplicative_inverse(&denominators); // 计算分母的逆
            // 得到f0/g0,f1/g1,f2/g2,f3/g3,...,f79/g79
            let quotient_values = numerators
                .zip(denominator_invs)
                .map(|(num, den_inv)| num * den_inv)
                .collect::<Vec<_>>();
            //将8个元素乘到一起，得到新的得到f0/g0,f1/g1,f2/g2,f3/g3,...,f9/g9，每一个f、g都是8个元素的乘积
            quotient_chunk_products(&quotient_values, degree) // 计算部分积
        })
        .collect::<Vec<_>>();

    let mut z_x = F::ONE; // 初始化 Z(x)
    let mut all_partial_products_and_zs = Vec::with_capacity(all_quotient_chunk_products.len()); // 初始化部分积和 Z 的向量
    for quotient_chunk_products in all_quotient_chunk_products {
        //计算r1=z0，r2=z0*z1=r1z1，r3=z0*z1*z2=r2*z2,……，rn=z0*z1*…*z(n-1)=r(n-1)*zn,得到r0,r2,…,r9
        let mut partial_products_and_z_gx =partial_products_and_z_gx(z_x, &quotient_chunk_products);

        //目的是把4段10个元素给串联到一个向量中，交换后，总的乘积不变，只是顺序变了。也可以先把40个向量串联到一个向量中，再调用partial_products_and_z_gx
        swap(&mut z_x, &mut partial_products_and_z_gx[num_prods]);
        all_partial_products_and_zs.push(partial_products_and_z_gx);
    }

    // 转置并收集结果为多项式值的向量
    transpose(&all_partial_products_and_zs)
        .into_par_iter()
        .map(PolynomialValues::new)
        .collect()
}

/// Computes lookup polynomials for a given challenge.
/// The polynomials hold the value of RE, Sum and Ldc of the Tip5 paper (<https://eprint.iacr.org/2023/107.pdf>). To reduce their
/// numbers, we batch multiple slots in a single polynomial. Since RE only involves degree one constraints, we can batch
/// all the slots of a row. For Sum and Ldc, batching increases the constraint degree, so we bound the number of
/// partial polynomials according to `max_quotient_degree_factor`.
/// As another optimization, Sum and LDC polynomials are shared (in so called partial SLDC polynomials), and the last value
/// of the last partial polynomial is Sum(end) - LDC(end). If the lookup argument is valid, then it must be equal to 0.
fn compute_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>, // 见证矩阵
    deltas: &[F; 4], // delta 值数组
    prover_data: &ProverOnlyCircuitData<F, C, D>, // 仅用于证明者的电路数据
    common_data: &CommonCircuitData<F, D>, // 通用电路数据
) -> Vec<PolynomialValues<F>> {
    let degree = common_data.degree(); // 获取电路的度数
    let num_lu_slots = LookupGate::num_slots(&common_data.config); // 获取查找表槽位数量
    let max_lookup_degree = common_data.config.max_quotient_degree_factor - 1; // 最大查找表度数
    let num_partial_lookups = num_lu_slots.div_ceil(max_lookup_degree); // 部分查找表的数量
    let num_lut_slots = LookupTableGate::num_slots(&common_data.config); // 查找表槽位数量
    let max_lookup_table_degree = num_lut_slots.div_ceil(num_partial_lookups); // 最大查找表度数

    // 第一个多项式是 RE，剩下的是部分 SLDCs。
    let mut final_poly_vecs = Vec::with_capacity(num_partial_lookups + 1);
    for _ in 0..num_partial_lookups + 1 {
        final_poly_vecs.push(PolynomialValues::<F>::new(vec![F::ZERO; degree])); // 初始化多项式向量
    }

    for LookupWire {
        last_lu_gate: last_lu_row, // 最后一个查找表门的行
        last_lut_gate: last_lut_row, // 最后一个查找表槽的行
        first_lut_gate: first_lut_row, // 第一个查找表槽的行
    } in prover_data.lookup_rows.clone()
    {
        // 设置部分和 RE 的值。
        for row in (last_lut_row..(first_lut_row + 1)).rev() {
            // 获取 Sum 的组合。
            let looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s)); // 获取查找表输入
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s)); // 获取查找表输出

                    looked_inp + deltas[LookupChallenges::ChallengeA as usize] * looked_out // 计算组合
                })
                .collect();
            // 获取 (alpha - 组合)。
            let minus_looked_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looked_combos[s])
                .collect();
            // 获取 1/(alpha - 组合)。
            let looked_combo_inverses = F::batch_multiplicative_inverse(&minus_looked_combos);

            // 获取查找表组合，用于检查查找表的正确性。
            let lookup_combos: Vec<F> = (0..num_lut_slots)
                .map(|s| {
                    let looked_inp = witness.get_wire(row, LookupTableGate::wire_ith_looked_inp(s)); // 获取查找表输入
                    let looked_out = witness.get_wire(row, LookupTableGate::wire_ith_looked_out(s)); // 获取查找表输出

                    looked_inp + deltas[LookupChallenges::ChallengeB as usize] * looked_out // 计算组合
                })
                .collect();

            // 计算下一行的第一个 RE 值。
            // 如果 `row == first_lut_row`，则 `final_poly_vecs[0].values[row + 1] == 0`。
            let mut new_re = final_poly_vecs[0].values[row + 1];
            for elt in &lookup_combos {
                new_re = new_re * deltas[LookupChallenges::ChallengeDelta as usize] + *elt
            }
            final_poly_vecs[0].values[row] = new_re;

            for slot in 0..num_partial_lookups {
                let prev = if slot != 0 {
                    final_poly_vecs[slot].values[row]
                } else {
                    // 如果 `row == first_lut_row`，则 `final_poly_vecs[num_partial_lookups].values[row + 1] == 0`。
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                };
                let sum = (slot * max_lookup_table_degree
                    ..min((slot + 1) * max_lookup_table_degree, num_lut_slots))
                    .fold(prev, |acc, s| {
                        acc + witness.get_wire(row, LookupTableGate::wire_ith_multiplicity(s))
                            * looked_combo_inverses[s]
                    });
                final_poly_vecs[slot + 1].values[row] = sum;
            }
        }

        // 设置部分 LDC 的值。
        for row in (last_lu_row..last_lut_row).rev() {
            // 获取查找组合。
            let looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| {
                    let looking_in = witness.get_wire(row, LookupGate::wire_ith_looking_inp(s)); // 获取查找输入
                    let looking_out = witness.get_wire(row, LookupGate::wire_ith_looking_out(s)); // 获取查找输出

                    looking_in + deltas[LookupChallenges::ChallengeA as usize] * looking_out // 计算组合
                })
                .collect();
            // 获取 (alpha - 组合)。
            let minus_looking_combos: Vec<F> = (0..num_lu_slots)
                .map(|s| deltas[LookupChallenges::ChallengeAlpha as usize] - looking_combos[s])
                .collect();
            // 获取 1 / (alpha - 组合)。
            let looking_combo_inverses = F::batch_multiplicative_inverse(&minus_looking_combos);

            for slot in 0..num_partial_lookups {
                let prev = if slot == 0 {
                    // 在任何行都有效，即使是 `first_lu_row`。
                    final_poly_vecs[num_partial_lookups].values[row + 1]
                } else {
                    final_poly_vecs[slot].values[row]
                };
                let sum = (slot * max_lookup_degree
                    ..min((slot + 1) * max_lookup_degree, num_lu_slots))
                    .fold(F::ZERO, |acc, s| acc + looking_combo_inverses[s]);
                final_poly_vecs[slot + 1].values[row] = prev - sum;
            }
        }
    }

    final_poly_vecs
}

/// Computes lookup polynomials for all challenges.
fn compute_all_lookup_polys<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    witness: &MatrixWitness<F>,
    deltas: &[F],
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    lookup: bool,
) -> Vec<PolynomialValues<F>> {
    if lookup {
        let polys: Vec<Vec<PolynomialValues<F>>> = (0..common_data.config.num_challenges)
            .map(|c| {
                compute_lookup_polys(
                    witness,
                    &deltas[c * NUM_COINS_LOOKUP..(c + 1) * NUM_COINS_LOOKUP]
                        .try_into()
                        .unwrap(),
                    prover_data,
                    common_data,
                )
            })
            .collect();
        polys.concat()
    } else {
        vec![]
    }
}

const BATCH_SIZE: usize = 32;
///计算商多项式
fn compute_quotient_polys<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    common_data: &CommonCircuitData<F, D>, // 通用电路数据
    prover_data: &'a ProverOnlyCircuitData<F, C, D>, // 仅用于证明者的电路数据
    public_inputs_hash: &<<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash, // 公共输入的哈希值
    wires_commitment: &'a PolynomialBatch<F, C, D>, // 线值承诺
    zs_partial_products_and_lookup_commitment: &'a PolynomialBatch<F, C, D>, // Z 多项式和部分积的承诺
    betas: &[F], // β 值数组
    gammas: &[F], // γ 值数组
    deltas: &[F], // δ 值数组
    alphas: &[F], // α 值数组
) -> Vec<PolynomialCoeffs<F>> {
    let num_challenges = common_data.config.num_challenges; // 挑战次数2

    let has_lookup = common_data.num_lookup_polys != 0; // 是否有查找表false

    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor); // 商多项式的度数位数3
    assert!(
        quotient_degree_bits <= common_data.config.fri_config.rate_bits,
        "Having constraints of degree higher than the rate is not supported yet. \
        If we need this in the future, we can precompute the larger LDE before computing the `PolynomialBatch`s."
    );

    // 重用在 `PolynomialBatch` 中计算的 LDE，并提取每 `step` 个点以获得匹配 `max_filtered_constraint_degree` 的 LDE。
    let step = 1 << (common_data.config.fri_config.rate_bits - quotient_degree_bits);//1
    // 在 Plonk 中打开 `Z` 多项式时，需要查看 `next_step` 步之外的点，因为我们在 `max_filtered_constraint_degree` 度数的 LDE 上工作。
    let next_step = 1 << quotient_degree_bits;//8

    let points = F::two_adic_subgroup(common_data.degree_bits() + quotient_degree_bits); // 2+3
    let lde_size = points.len(); // LDE 的大小32

    //得到一个子群的coset。每个元素都减去1，还输出每个元素的逆
    //1126 to do
    let z_h_on_coset = ZeroPolyOnCoset::new(common_data.degree_bits(), quotient_degree_bits); // 2，3；在陪集上的零多项式

    // 预计算查找表在 delta 挑战上的评估值
    // 这些值用于生成每个查找表的最终 RE 约束，并且在 `check_lookup_constraints_batched` 中每次都相同。
    // `lut_poly_evals[i][j]` 给出第 `i` 个挑战和第 `j` 个查找表的评估值
    let lut_re_poly_evals: Vec<Vec<F>> = if has_lookup {
        let num_lut_slots = LookupTableGate::num_slots(&common_data.config); // 查找表槽位数量
        (0..num_challenges)
            .map(move |i| {
                let cur_deltas = &deltas[NUM_COINS_LOOKUP * i..NUM_COINS_LOOKUP * (i + 1)];
                let cur_challenge_delta = cur_deltas[LookupChallenges::ChallengeDelta as usize];

                (LookupSelectors::StartEnd as usize..common_data.num_lookup_selectors)
                    .map(|r| {
                        let lut_row_number = common_data.luts
                            [r - LookupSelectors::StartEnd as usize]
                            .len()
                            .div_ceil(num_lut_slots);

                        get_lut_poly(
                            common_data,
                            r - LookupSelectors::StartEnd as usize,
                            cur_deltas,
                            num_lut_slots * lut_row_number,
                        )
                            .eval(cur_challenge_delta)
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };

    let lut_re_poly_evals_refs: Vec<&[F]> =
        lut_re_poly_evals.iter().map(|v| v.as_slice()).collect(); // 查找表评估值的引用

    let points_batches = points.par_chunks(BATCH_SIZE); // 将点分批处理
    let num_batches = points.len().div_ceil(BATCH_SIZE); // 批次数量

    let quotient_values: Vec<Vec<F>> = points_batches
        .enumerate()
        .flat_map(|(batch_i, xs_batch)| {
            // 每个批次必须具有相同的大小，除了最后一个批次，它可能较小。
            debug_assert!(
                xs_batch.len() == BATCH_SIZE
                    || (batch_i == num_batches - 1 && xs_batch.len() <= BATCH_SIZE)
            );

            let indices_batch: Vec<usize> =
                (BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + xs_batch.len()).collect(); // 批次索引，如[0,1,2,3,...,31]

            let mut shifted_xs_batch = Vec::with_capacity(xs_batch.len()); // 大子集K的陪集
            let mut local_zs_batch = Vec::with_capacity(xs_batch.len()); // local Z 批次
            let mut next_zs_batch = Vec::with_capacity(xs_batch.len()); // 下一个 Z 批次

            let mut local_lookup_batch = Vec::with_capacity(xs_batch.len()); // local查找表批次
            let mut next_lookup_batch = Vec::with_capacity(xs_batch.len()); // 下一个查找表批次

            let mut partial_products_batch = Vec::with_capacity(xs_batch.len()); // 部分积批次
            let mut s_sigmas_batch = Vec::with_capacity(xs_batch.len()); // σ 值批次

            let mut local_constants_batch_refs = Vec::with_capacity(xs_batch.len()); // local常量批次引用
            let mut local_wires_batch_refs = Vec::with_capacity(xs_batch.len()); // local线值批次引用

            for (&i, &x) in indices_batch.iter().zip(xs_batch) {
                let shifted_x = F::coset_shift() * x; // 计算陪集的元素
                let i_next = (i + next_step) % lde_size; // 计算下一个索引
                let local_constants_sigmas = prover_data
                    .constants_sigmas_commitment
                    .get_lde_values(i, step); // 获取本次常量和 σ 值
                let local_constants = &local_constants_sigmas[common_data.constants_range()]; // 获取local常量
                let s_sigmas = &local_constants_sigmas[common_data.sigmas_range()]; // 获取 σ 值
                let local_wires = wires_commitment.get_lde_values(i, step); // 获取local wire值
                let local_zs_partial_and_lookup =
                    zs_partial_products_and_lookup_commitment.get_lde_values(i, step); // 获取local Z 和部分积
                let next_zs_partial_and_lookup =
                    zs_partial_products_and_lookup_commitment.get_lde_values(i_next, step); // 获取下一个 Z 和部分积

                let local_zs = &local_zs_partial_and_lookup[common_data.zs_range()]; // 获取local Z

                let next_zs = &next_zs_partial_and_lookup[common_data.zs_range()]; // 获取下一个 Z

                let partial_products =
                    &local_zs_partial_and_lookup[common_data.partial_products_range()]; // 获取部分积

                if has_lookup {
                    let local_lookup_zs = &local_zs_partial_and_lookup[common_data.lookup_range()]; // 获取local查找表 Z

                    let next_lookup_zs = &next_zs_partial_and_lookup[common_data.lookup_range()]; // 获取下一个查找表 Z
                    debug_assert_eq!(local_lookup_zs.len(), common_data.num_all_lookup_polys());

                    local_lookup_batch.push(local_lookup_zs); // 添加到local查找表批次
                    next_lookup_batch.push(next_lookup_zs); // 添加到下一个查找表批次
                }

                debug_assert_eq!(local_wires.len(), common_data.config.num_wires);
                debug_assert_eq!(local_zs.len(), num_challenges);

                local_constants_batch_refs.push(local_constants); // 添加到local常量批次引用
                local_wires_batch_refs.push(local_wires); // 添加到local线值批次引用

                shifted_xs_batch.push(shifted_x); // 添加到偏移后的 x 批次
                local_zs_batch.push(local_zs); // 添加到local Z 批次
                next_zs_batch.push(next_zs); // 添加到下一个 Z 批次
                partial_products_batch.push(partial_products); // 添加到部分积批次
                s_sigmas_batch.push(s_sigmas); // 添加到 σ 值批次
            }

            // NB (JN): 我不确定下面的效率如何。需要测量。
            let mut local_constants_batch =
                vec![F::ZERO; xs_batch.len() * local_constants_batch_refs[0].len()]; // 初始化本地常量批次，32*4
            //先拿出第0个元素，拼成32个，再拿出第1个元素，拼成32个，一共4个元素，拼成32*4个
            for i in 0..local_constants_batch_refs[0].len() {
                for (j, constants) in local_constants_batch_refs.iter().enumerate() {
                    local_constants_batch[i * xs_batch.len() + j] = constants[i];
                }
            }

            let mut local_wires_batch =
                vec![F::ZERO; xs_batch.len() * local_wires_batch_refs[0].len()]; // 初始化本地线值批次
            for i in 0..local_wires_batch_refs[0].len() {
                for (j, wires) in local_wires_batch_refs.iter().enumerate() {
                    local_wires_batch[i * xs_batch.len() + j] = wires[i];
                }
            }

            let vars_batch = EvaluationVarsBaseBatch::new(
                xs_batch.len(),
                &local_constants_batch,
                &local_wires_batch,
                public_inputs_hash,
            ); // 创建评估变量批次

            let mut quotient_values_batch = eval_vanishing_poly_base_batch::<F, D>(
                common_data,
                &indices_batch,
                &shifted_xs_batch,
                vars_batch,
                &local_zs_batch,
                &next_zs_batch,
                &local_lookup_batch,
                &next_lookup_batch,
                &partial_products_batch,
                &s_sigmas_batch,
                betas,
                gammas,
                deltas,
                alphas,
                &z_h_on_coset,
                &lut_re_poly_evals_refs,
            ); // 评估消失多项式基批次

            for (&i, quotient_values) in indices_batch.iter().zip(quotient_values_batch.iter_mut())
            {
                let denominator_inv = z_h_on_coset.eval_inverse(i); // 计算分母的逆
                quotient_values
                    .iter_mut()
                    .for_each(|v| *v *= denominator_inv); // 乘以分母的逆
            }
            quotient_values_batch
        })
        .collect(); // 收集商值批次

    transpose(&quotient_values)
        .into_par_iter()
        .map(PolynomialValues::new)
        .map(|values| values.coset_ifft(F::coset_shift()))
        .collect() // 转置并收集结果为多项式值的向量
}