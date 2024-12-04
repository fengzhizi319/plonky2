#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_maybe_rayon::*;

use crate::field::extension::{flatten, unflatten, Extendable};
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::RichField;
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::plonk::plonk_common::reduce_with_powers;
use crate::timed;
use crate::util::reverse_index_bits_in_place;
use crate::util::timing::TimingTree;

/// Builds a FRI proof.
pub fn fri_proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    // Coefficients of the polynomial on which the LDT is performed. Only the first `1/rate` coefficients are non-zero.
    lde_polynomial_coeffs: PolynomialCoeffs<F::Extension>,
    // Evaluation of the polynomial on the large domain.
    lde_polynomial_values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
    timing: &mut TimingTree,
) -> FriProof<F, C::Hasher, D> {
    let n = lde_polynomial_values.len();
    assert_eq!(lde_polynomial_coeffs.len(), n);

    // Commit phase
    let (trees, final_coeffs) = timed!(
        timing,
        "fold codewords in the commitment phase",
        fri_committed_trees::<F, C, D>(
            lde_polynomial_coeffs,
            lde_polynomial_values,
            challenger,
            fri_params,
        )
    );

    // PoW phase，返回一个challenger结果前面几个为全0的输入消息
    let pow_witness = timed!(
        timing,
        "find proof-of-work witness",
        fri_proof_of_work::<F, C, D>(challenger, &fri_params.config)
    );

    // Query phase
    let query_round_proofs =
        fri_prover_query_rounds::<F, C, D>(initial_merkle_trees, &trees, challenger, n, fri_params);

    FriProof {
        commit_phase_merkle_caps: trees.iter().map(|t| t.cap.clone()).collect(),
        query_round_proofs,
        final_poly: final_coeffs,
        pow_witness,
    }
}

pub(crate) type FriCommitedTrees<F, C, const D: usize> = (
    Vec<MerkleTree<F, <C as GenericConfig<D>>::Hasher>>,
    PolynomialCoeffs<<F as Extendable<D>>::Extension>,
);

///求掉额外低度扩展的多项式0系数，根据reduction_arity_bits生成一颗新树
fn fri_committed_trees<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    mut coeffs: PolynomialCoeffs<F::Extension>,
    mut values: PolynomialValues<F::Extension>,
    challenger: &mut Challenger<F, C::Hasher>,
    fri_params: &FriParams,
) -> FriCommitedTrees<F, C, D> {
    let mut trees = Vec::with_capacity(fri_params.reduction_arity_bits.len());

    let mut shift = F::MULTIPLICATIVE_GROUP_GENERATOR;
    // println!("coeffs: {:?}", coeffs);
    // println!("values: {:?}", values);
    //reduction_arity_bits=vec![]
    for arity_bits in &fri_params.reduction_arity_bits {
        let arity = 1 << arity_bits;

        reverse_index_bits_in_place(&mut values.values);
        let chunked_values = values
            .values
            .par_chunks(arity)
            .map(|chunk: &[F::Extension]| flatten(chunk))
            .collect();
        let tree = MerkleTree::<F, C::Hasher>::new(chunked_values, fri_params.config.cap_height);

        challenger.observe_cap(&tree.cap);
        trees.push(tree);

        let beta = challenger.get_extension_challenge::<D>();
        // P(x) = sum_{i<r} x^i * P_i(x^r) becomes sum_{i<r} beta^i * P_i(x).
        coeffs = PolynomialCoeffs::new(
            coeffs
                .coeffs
                .par_chunks_exact(arity)
                .map(|chunk| reduce_with_powers(chunk, beta))
                .collect::<Vec<_>>(),
        );
        shift = shift.exp_u64(arity as u64);
        values = coeffs.coset_fft(shift.into())
    }

    //求掉额外低度扩展的多项式0系数
    // The coefficients being removed here should always be zero.
    coeffs
        .coeffs
        .truncate(coeffs.len() >> fri_params.config.rate_bits);
    //把系数添加到hash中
    challenger.observe_extension_elements(&coeffs.coeffs);
    (trees, coeffs)
}

/// Performs the proof-of-work (a.k.a. grinding) step of the FRI protocol. Returns the PoW witness.
/// 执行 FRI 协议的工作量证明（PoW）步骤。返回 PoW 见证。
pub(crate) fn fri_proof_of_work<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    challenger: &mut Challenger<F, C::Hasher>,
    config: &FriConfig,
) -> F {
    // Calculate the minimum number of leading zeros required.
    // 计算所需的最小前导零位数
    //proof_of_work_bits=16
    //F::order().bits()=64

    let min_leading_zeros = config.proof_of_work_bits + (64 - F::order().bits()) as u32;

    // The easiest implementation would be repeatedly clone our Challenger. With each clone, we'd
    // observe an incrementing PoW witness, then get the PoW response. If it contained sufficient
    // leading zeros, we'd end the search, and store this clone as our new challenger.
    //
    // However, performance is critical here. We want to avoid cloning Challenger, particularly
    // since it stores vectors, which means allocations. We'd like a more compact state to clone.
    //
    // We know that a duplex will be performed right after we send the PoW witness, so we can ignore
    // any output_buffer, which will be invalidated. We also know
    // input_buffer.len() < H::Permutation::WIDTH, an invariant of Challenger.
    //
    // We separate the duplex operation into two steps, one which can be performed now, and the
    // other which depends on the PoW witness candidate. The first step is the overwrite our sponge
    // state with any inputs (excluding the PoW witness candidate). The second step is to overwrite
    // one more element of our sponge state with the candidate, then apply the permutation,
    // obtaining our duplex's post-state which contains the PoW response.
    //
    // 最简单的实现是反复克隆我们的 Challenger。每次克隆时，我们会观察一个递增的 PoW 见证，然后获取 PoW 响应。
    // 如果它包含足够的前导零，我们就结束搜索，并将此克隆存储为我们的新 Challenger。
    //
    // 然而，性能在这里是关键。我们希望避免克隆 Challenger，特别是因为它存储了向量，这意味着分配。
    // 我们希望有一个更紧凑的状态来克隆。
    //
    // 我们知道在发送 PoW 见证后会立即执行一个双工操作，因此我们可以忽略任何 output_buffer，因为它将被无效化。
    // 我们还知道 input_buffer.len() < H::Permutation::WIDTH，这是 Challenger 的一个不变量。
    //
    // 我们将双工操作分为两步，一步可以现在执行，另一部分取决于 PoW 见证候选者。
    // 第一步是用任何输入（不包括 PoW 见证候选者）覆盖我们的海绵状态。
    // 第二步是用候选者覆盖海绵状态的另一个元素，然后应用置换，获得包含 PoW 响应的双工后状态。
    let mut duplex_intermediate_state = challenger.sponge_state;
    let witness_input_pos = challenger.input_buffer.len();
    duplex_intermediate_state.set_from_iter(challenger.input_buffer.clone(), 0);
    let t1=vec![1];

    // Find a PoW witness within the range that meets the condition.
    // 在范围内查找满足条件的 PoW 见证
    let pow_witness = (0..=F::NEG_ONE.to_canonical_u64())
        .into_par_iter()
        .find_any(|&candidate| {
            let mut duplex_state = duplex_intermediate_state;
            duplex_state.set_elt(F::from_canonical_u64(candidate), witness_input_pos);//设置hash运算的输入
            duplex_state.permute();//进行hash运算
            let pow_response = duplex_state.squeeze().iter().last().unwrap();//得到运算结果
            let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
            leading_zeros >= min_leading_zeros
        })
        .map(F::from_canonical_u64)
        .expect("Proof of work failed. This is highly unlikely!");

    // Recompute pow_response using our normal Challenger code, and make sure it matches.
    // 使用我们正常的 Challenger 代码重新计算 pow_response，并确保它匹配。
    challenger.observe_element(pow_witness);
    let pow_response = challenger.get_challenge();
    let leading_zeros = pow_response.to_canonical_u64().leading_zeros();
    assert!(leading_zeros >= min_leading_zeros);
    pow_witness
}

///生成fri_params.config.num_query_rounds个挑战值，并根据挑战值生成各种merkle树的证明
fn fri_prover_query_rounds<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>],
    trees: &[MerkleTree<F, C::Hasher>],
    challenger: &mut Challenger<F, C::Hasher>,
    n: usize,
    fri_params: &FriParams,
) -> Vec<FriQueryRound<F, C::Hasher, D>> {
    // challenger
    //     .get_n_challenges(fri_params.config.num_query_rounds)//num_query_rounds=28
    //     .into_par_iter()
    //     .map(|rand| {
    //         let x_index = rand.to_canonical_u64() as usize % n;
    //         fri_prover_query_round::<F, C, D>(initial_merkle_trees, trees, x_index, fri_params)
    //     })
    //     .collect()
    challenger
        .get_n_challenges(fri_params.config.num_query_rounds)//num_query_rounds=28
        .into_iter()
        .map(|rand| {
            let x_index = rand.to_canonical_u64() as usize % n;
            fri_prover_query_round::<F, C, D>(initial_merkle_trees, trees, x_index, fri_params)
        })
        .collect()
}

fn fri_prover_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    initial_merkle_trees: &[&MerkleTree<F, C::Hasher>], // 初始 Merkle 树的引用数组
    trees: &[MerkleTree<F, C::Hasher>], // Merkle 树的引用数组
    mut x_index: usize, // 索引值
    fri_params: &FriParams, // FRI 参数
) -> FriQueryRound<F, C::Hasher, D> {
    let mut query_steps = Vec::new(); // 存储查询步骤的向量
    // 获取初始 Merkle 树的证明
    // let initial_proof = initial_merkle_trees
    //     .iter() // 迭代初始 Merkle 树
    //     .map(|t| (t.get(x_index).to_vec(), t.prove(x_index))) // 获取索引处的值并生成证明
    //     .collect::<Vec<_>>(); // 收集结果为向量
    let mut initial_proof = Vec::new(); // 创建一个空向量来存储结果
    for t in initial_merkle_trees {
        let value = t.get(x_index).to_vec(); // 获取索引处的值并转换为向量
        let proof = t.prove(x_index); // 生成证明
        initial_proof.push((value, proof)); // 将值和证明作为元组添加到向量中
    }
    // 遍历 Merkle 树
    for (i, tree) in trees.iter().enumerate() {
        let arity_bits = fri_params.reduction_arity_bits[i]; // 获取当前树的 arity bits
        let evals = unflatten(tree.get(x_index >> arity_bits)); // 获取评估值并展开
        let merkle_proof = tree.prove(x_index >> arity_bits); // 生成 Merkle 证明

        // 将评估值和 Merkle 证明添加到查询步骤
        query_steps.push(FriQueryStep {
            evals,
            merkle_proof,
        });

        x_index >>= arity_bits; // 更新索引值
    }

    // 返回 FRI 查询轮次
    FriQueryRound {
        initial_trees_proof: FriInitialTreeProof {
            evals_proofs: initial_proof, // 初始树的评估证明
        },
        steps: query_steps, // 查询步骤
    }
}
