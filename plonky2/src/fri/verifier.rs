#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use anyhow::{ensure, Result};

use crate::field::extension::{flatten, Extendable, FieldExtension};
use crate::field::interpolation::{barycentric_weights, interpolate};
use crate::field::types::Field;
use crate::fri::proof::{FriChallenges, FriInitialTreeProof, FriProof, FriQueryRound};
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo, FriOpenings};
use crate::fri::validate_shape::validate_fri_proof_shape;
use crate::fri::{FriConfig, FriParams};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::util::reducing::ReducingFactor;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place};

/// Computes P'(x^arity) from {P(x*g^i)}_(i=0..arity), where g is a `arity`-th root of unity
/// and P' is the FRI reduced polynomial.
pub(crate) fn compute_evaluation<F: Field + Extendable<D>, const D: usize>(
    x: F,
    x_index_within_coset: usize,
    arity_bits: usize,
    evals: &[F::Extension],
    beta: F::Extension,
) -> F::Extension {
    let arity = 1 << arity_bits;
    debug_assert_eq!(evals.len(), arity);

    let g = F::primitive_root_of_unity(arity_bits);

    // The evaluation vector needs to be reordered first.
    let mut evals = evals.to_vec();
    reverse_index_bits_in_place(&mut evals);
    let rev_x_index_within_coset = reverse_bits(x_index_within_coset, arity_bits);
    let coset_start = x * g.exp_u64((arity - rev_x_index_within_coset) as u64);
    // The answer is gotten by interpolating {(x*g^i, P(x*g^i))} and evaluating at beta.
    let points = g
        .powers()
        .map(|y| (coset_start * y).into())
        .zip(evals)
        .collect::<Vec<_>>();
    let barycentric_weights = barycentric_weights(&points);
    interpolate(&points, beta, &barycentric_weights)
}

pub(crate) fn fri_verify_proof_of_work<F: RichField + Extendable<D>, const D: usize>(
    fri_pow_response: F,
    config: &FriConfig,
) -> Result<()> {
    let leading_zeros_len=fri_pow_response.to_canonical_u64().leading_zeros();
    ensure!(
        leading_zeros_len
            >= config.proof_of_work_bits + (64 - F::order().bits()) as u32,
        "Invalid proof of work witness."
    );
    // ensure!(
    //     fri_pow_response.to_canonical_u64().leading_zeros()
    //         >= config.proof_of_work_bits + (64 - F::order().bits()) as u32,
    //     "Invalid proof of work witness."
    // );

    Ok(())
}

pub fn verify_fri_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    openings: &FriOpenings<F, D>,
    challenges: &FriChallenges<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    params: &FriParams,
) -> Result<()> {
    validate_fri_proof_shape::<F, C, D>(proof, instance, params)?;

    // Size of the LDE domain.
    let n = params.lde_size();

    // Check PoW.
    fri_verify_proof_of_work(challenges.fri_pow_response, &params.config)?;

    // Check that parameters are coherent.
    ensure!(
        params.config.num_query_rounds == proof.query_round_proofs.len(),
        "Number of query rounds does not match config."
    );

    let precomputed_reduced_evals =
        PrecomputedReducedOpenings::from_os_and_alpha(openings, challenges.fri_alpha);
    for (&x_index, round_proof) in challenges
        .fri_query_indices
        .iter()
        .zip(&proof.query_round_proofs)
    {
        fri_verifier_query_round::<F, C, D>(
            instance,
            challenges,
            &precomputed_reduced_evals,
            initial_merkle_caps,
            proof,
            x_index,
            n,
            round_proof,
            params,
        )?;
    }

    Ok(())
}

fn fri_verify_initial_proof<F: RichField, H: Hasher<F>>(
    x_index: usize,
    proof: &FriInitialTreeProof<F, H>,
    initial_merkle_caps: &[MerkleCap<F, H>],
) -> Result<()> {
    for ((evals, merkle_proof), cap) in proof.evals_proofs.iter().zip(initial_merkle_caps) {
        verify_merkle_proof_to_cap::<F, H>(evals.clone(), x_index, cap, merkle_proof)?;
    }

    Ok(())
}

/// 计算FRI（快速Reed-Solomon交互式Oracle接近性证明）协议中的初始多项式评估值。
/// 具体来说，它通过将多个多项式的评估值组合在一起，生成一个新的评估值，用于后续的验证步骤。
/// 这个过程涉及到使用给定的挑战值（alpha）和子群元素（subgroup_x），并结合预先计算的评估值，来计算最终的结果。
///
/// # 参数
/// - `instance`: FRI实例信息，包含批次信息和oracle信息。
/// - `proof`: FRI初始树证明，包含多项式评估值和Merkle证明。
/// - `alpha`: 挑战值，用于多项式评估值的组合。
/// - `subgroup_x`: 子群元素，用于计算分母。
/// - `precomputed_reduced_evals`: 预先计算的简化评估值。
/// - `params`: FRI参数，包含隐藏和盲化信息。
///
/// # 返回值
/// 返回计算出的初始多项式评估值。
pub(crate) fn fri_combine_initial<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    proof: &FriInitialTreeProof<F, C::Hasher>,
    alpha: F::Extension,
    subgroup_x: F,
    precomputed_reduced_evals: &PrecomputedReducedOpenings<F, D>,
    params: &FriParams,
) -> F::Extension {
    // 确保D大于1，因为D=1的情况未实现
    assert!(D > 1, "Not implemented for D=1.");

    // 将subgroup_x从基域转换为扩展域
    let subgroup_x = F::Extension::from_basefield(subgroup_x);

    // 初始化ReducingFactor，用于简化评估值
    let mut alpha = ReducingFactor::new(alpha);

    // 初始化sum为零，用于累加最终结果
    let mut sum = F::Extension::ZERO;

    // 遍历每个批次和预先计算的简化评估值
    for (batch, reduced_openings) in instance
        .batches
        .iter()
        .zip(&precomputed_reduced_evals.reduced_openings_at_point)
    {
        let FriBatchInfo { point, polynomials } = batch;

        // 获取多项式的评估值，并根据是否隐藏和盲化进行处理
        let evals = polynomials
            .iter()
            .map(|p| {
                let poly_blinding = instance.oracles[p.oracle_index].blinding;
                let salted = params.hiding && poly_blinding;
                proof.unsalted_eval(p.oracle_index, p.polynomial_index, salted)
            })
            .map(F::Extension::from_basefield);

        // 使用alpha简化评估值
        let reduced_evals = alpha.reduce(evals);

        // 计算分子和分母
        let numerator = reduced_evals - *reduced_openings;
        let denominator = subgroup_x - *point;

        // 累加结果
        sum = alpha.shift(sum);
        sum += numerator / denominator;
    }

    // 返回最终累加的结果
    sum
}

/// 验证FRI（快速Reed-Solomon交互式Oracle接近性证明）协议中的查询轮次。
/// 具体来说，它通过验证初始树证明、计算和验证每个步骤的评估值，最终验证FRI的正确性。
///
/// # 参数
/// - `instance`: FRI实例信息，包含批次信息和oracle信息。
/// - `challenges`: FRI挑战值，包含alpha和beta等值。
/// - `precomputed_reduced_evals`: 预先计算的简化评估值。
/// - `initial_merkle_caps`: 初始Merkle树的根哈希值。
/// - `proof`: FRI证明，包含查询轮次的证明。
/// - `x_index`: 查询点的索引。
/// - `n`: LDE域的大小。
/// - `round_proof`: FRI查询轮次的证明，包含每个步骤的评估值和Merkle证明。
/// - `params`: FRI参数，包含隐藏和盲化信息。
///
/// # 返回值
/// 返回验证结果，如果验证成功则返回Ok，否则返回错误信息。
fn fri_verifier_query_round<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    instance: &FriInstanceInfo<F, D>,
    challenges: &FriChallenges<F, D>,
    precomputed_reduced_evals: &PrecomputedReducedOpenings<F, D>,
    initial_merkle_caps: &[MerkleCap<F, C::Hasher>],
    proof: &FriProof<F, C::Hasher, D>,
    mut x_index: usize,
    n: usize,
    round_proof: &FriQueryRound<F, C::Hasher, D>,
    params: &FriParams,
) -> Result<()> {
    // 验证初始树证明
    fri_verify_initial_proof::<F, C::Hasher>(
        x_index,
        &round_proof.initial_trees_proof,
        initial_merkle_caps,
    )?;
    // `subgroup_x` 是 `subgroup[x_index]`，即域中的实际字段元素。
    let log_n = log2_strict(n);
    let mut subgroup_x = F::MULTIPLICATIVE_GROUP_GENERATOR
        * F::primitive_root_of_unity(log_n).exp_u64(reverse_bits(x_index, log_n) as u64);

    // `old_eval` 是最后一个派生的评估值；它将在下一次迭代中检查与其提交的“父”值的一致性。
    let mut old_eval = fri_combine_initial::<F, C, D>(
        instance,
        &round_proof.initial_trees_proof,
        challenges.fri_alpha,
        subgroup_x,
        precomputed_reduced_evals,
        params,
    );

    // 遍历每个步骤的评估值
    for (i, &arity_bits) in params.reduction_arity_bits.iter().enumerate() {
        let arity = 1 << arity_bits; // 计算当前步骤的arity
        let evals = &round_proof.steps[i].evals; // 获取当前步骤的评估值

        // 将 x_index 拆分为陪集的索引和陪集内的索引
        let coset_index = x_index >> arity_bits;
        let x_index_within_coset = x_index & (arity - 1);

        // 检查与前一轮的旧评估值的一致性
        ensure!(evals[x_index_within_coset] == old_eval);

        // 从 {P(x)}_{x^arity=y} 推断 P(y)
        old_eval = compute_evaluation(
            subgroup_x,
            x_index_within_coset,
            arity_bits,
            evals,
            challenges.fri_betas[i],
        );

        // 验证Merkle证明
        verify_merkle_proof_to_cap::<F, C::Hasher>(
            flatten(evals),
            coset_index,
            &proof.commit_phase_merkle_caps[i],
            &round_proof.steps[i].merkle_proof,
        )?;

        // 更新点 x 到 x^arity
        subgroup_x = subgroup_x.exp_power_of_2(arity_bits);

        x_index = coset_index;
    }

    // 最终检查FRI。在所有简化之后，我们检查最终多项式是否等于证明者发送的多项式。
    ensure!(
        proof.final_poly.eval(subgroup_x.into()) == old_eval,
        "Final polynomial evaluation is invalid."
    );

    Ok(())
}

/// For each opening point, holds the reduced (by `alpha`) evaluations of each polynomial that's
/// opened at that point.
#[derive(Clone, Debug)]
pub(crate) struct PrecomputedReducedOpenings<F: RichField + Extendable<D>, const D: usize> {
    pub reduced_openings_at_point: Vec<F::Extension>,
}

impl<F: RichField + Extendable<D>, const D: usize> PrecomputedReducedOpenings<F, D> {
    /// 从给定的FRI打开值和挑战值alpha中预先计算简化的评估值。
    /// 具体来说，它通过将每个批次的多项式评估值使用alpha进行简化，生成一个新的评估值列表。
    ///
    /// # 参数
    /// - `openings`: FRI打开值，包含多个批次的多项式评估值。
    /// - `alpha`: 挑战值，用于多项式评估值的简化。
    ///
    /// # 返回值
    /// 返回包含简化评估值的结构体`PrecomputedReducedOpenings`。
    pub(crate) fn from_os_and_alpha(openings: &FriOpenings<F, D>, alpha: F::Extension) -> Self {
        // 遍历每个批次的多项式评估值，并使用alpha进行折叠
        let reduced_openings_at_point = openings
            .batches
            .iter()
            .map(|batch| ReducingFactor::new(alpha).reduce(batch.values.iter()))
            .collect();

        Self {
            reduced_openings_at_point,
        }
    }
}
