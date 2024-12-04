use anyhow::ensure;

use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::GenericConfig;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};

pub(crate) fn validate_proof_with_pis_shape<F, C, const D: usize>(
    proof_with_pis: &ProofWithPublicInputs<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let ProofWithPublicInputs {
        proof,
        public_inputs,
    } = proof_with_pis;
    validate_proof_shape(proof, common_data)?;
    ensure!(
        public_inputs.len() == common_data.num_public_inputs,
        "Number of public inputs doesn't match circuit data."
    );
    Ok(())
}
///此函数的主要作用是验证给定的证明（proof）的形状是否符合预期的电路数据（common_data）
/// 具体来说，它检查了证明中的各个部分（如电线、部分积、商多项式等）的数量和高度是否与电路数据中的配置相匹配。通过这些检查，确保证明的结构和电路数据的一致性。
fn validate_proof_shape<F, C, const D: usize>(
    proof: &Proof<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
) -> anyhow::Result<()>
where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
{
    let config = &common_data.config;
    let Proof {
        wires_cap,
        plonk_zs_partial_products_cap,
        quotient_polys_cap,
        openings,
        // The shape of the opening proof will be checked in the FRI verifier (see
        // validate_fri_proof_shape), so we ignore it here.
        opening_proof: _,
    } = proof;
    let OpeningSet {
        constants,
        plonk_sigmas,
        wires,
        plonk_zs,
        plonk_zs_next,
        partial_products,
        quotient_polys,
        lookup_zs,
        lookup_zs_next,
    } = openings;
    let cap_height = common_data.fri_params.config.cap_height;
    ensure!(wires_cap.height() == cap_height);
    ensure!(plonk_zs_partial_products_cap.height() == cap_height);
    ensure!(quotient_polys_cap.height() == cap_height);
    ensure!(constants.len() == common_data.num_constants);
    ensure!(plonk_sigmas.len() == config.num_routed_wires);
    ensure!(wires.len() == config.num_wires);
    ensure!(plonk_zs.len() == config.num_challenges);
    ensure!(plonk_zs_next.len() == config.num_challenges);
    ensure!(partial_products.len() == config.num_challenges * common_data.num_partial_products);
    ensure!(quotient_polys.len() == common_data.num_quotient_polys());
    ensure!(lookup_zs.len() == common_data.num_all_lookup_polys());
    ensure!(lookup_zs_next.len() == common_data.num_all_lookup_polys());
    Ok(())
}
