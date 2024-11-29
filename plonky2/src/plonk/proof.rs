//! plonky2 proof definition.
//!
//! Proofs can be later compressed to reduce their size, into either
//! [`CompressedProof`] or [`CompressedProofWithPublicInputs`] formats.
//! The latter can be directly passed to a verifier to assert its correctness.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use anyhow::ensure;
use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::field::extension::Extendable;
use crate::fri::oracle::PolynomialBatch;
use crate::fri::proof::{
    CompressedFriProof, FriChallenges, FriChallengesTarget, FriProof, FriProofTarget,
};
use crate::fri::structure::{
    FriOpeningBatch, FriOpeningBatchTarget, FriOpenings, FriOpeningsTarget,
};
use crate::fri::FriParams;
use crate::hash::hash_types::{MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::verifier::verify_with_challenges;
use crate::util::serialization::{Buffer, Read, Write};

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct Proof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> {
    /// Merkle cap of LDEs of wire values.
    pub wires_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of Z, in the context of Plonk's permutation argument.
    pub plonk_zs_partial_products_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of the quotient polynomial components.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: OpeningSet<F, D>,
    /// A batch FRI argument for all openings.
    pub opening_proof: FriProof<F, C::Hasher, D>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofTarget<const D: usize> {
    pub wires_cap: MerkleCapTarget,
    pub plonk_zs_partial_products_cap: MerkleCapTarget,
    pub quotient_polys_cap: MerkleCapTarget,
    pub openings: OpeningSetTarget<D>,
    pub opening_proof: FriProofTarget<D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Proof<F, C, D> {
    /// Compress the proof.
    pub fn compress(self, indices: &[usize], params: &FriParams) -> CompressedProof<F, C, D> {
        let Proof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof,
        } = self;

        CompressedProof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof: opening_proof.compress(indices, params),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct ProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof: Proof<F, C, D>,
    pub public_inputs: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
ProofWithPublicInputs<F, C, D>
{
    pub fn compress(
        self,
        circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<CompressedProofWithPublicInputs<F, C, D>> {
        let indices = self.fri_query_indices(circuit_digest, common_data)?;
        let compressed_proof = self.proof.compress(&indices, &common_data.fri_params);
        Ok(CompressedProofWithPublicInputs {
            public_inputs: self.public_inputs,
            proof: compressed_proof,
        })
    }

    pub fn get_public_inputs_hash(
        &self,
    ) -> <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash {
        C::InnerHasher::hash_no_pad(&self.public_inputs)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer
            .write_proof_with_public_inputs(self)
            .expect("Writing to a byte-vector cannot fail.");
        buffer
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<Self> {
        let mut buffer = Buffer::new(&bytes);
        let proof = buffer
            .read_proof_with_public_inputs(common_data)
            .map_err(anyhow::Error::msg)?;
        Ok(proof)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedProof<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    /// Merkle cap of LDEs of wire values.
    pub wires_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of Z, in the context of Plonk's permutation argument.
    pub plonk_zs_partial_products_cap: MerkleCap<F, C::Hasher>,
    /// Merkle cap of LDEs of the quotient polynomial components.
    pub quotient_polys_cap: MerkleCap<F, C::Hasher>,
    /// Purported values of each polynomial at the challenge point.
    pub openings: OpeningSet<F, D>,
    /// A compressed batch FRI argument for all openings.
    pub opening_proof: CompressedFriProof<F, C::Hasher, D>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
CompressedProof<F, C, D>
{
    /// Decompress the proof.
    pub(crate) fn decompress(
        self,
        challenges: &ProofChallenges<F, D>,
        fri_inferred_elements: FriInferredElements<F, D>,
        params: &FriParams,
    ) -> Proof<F, C, D> {
        let CompressedProof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof,
        } = self;

        Proof {
            wires_cap,
            plonk_zs_partial_products_cap,
            quotient_polys_cap,
            openings,
            opening_proof: opening_proof.decompress(challenges, fri_inferred_elements, params),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
#[serde(bound = "")]
pub struct CompressedProofWithPublicInputs<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
> {
    pub proof: CompressedProof<F, C, D>,
    pub public_inputs: Vec<F>,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
CompressedProofWithPublicInputs<F, C, D>
{
    pub fn decompress(
        self,
        circuit_digest: &<<C as GenericConfig<D>>::Hasher as Hasher<C::F>>::Hash,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<ProofWithPublicInputs<F, C, D>> {
        let challenges =
            self.get_challenges(self.get_public_inputs_hash(), circuit_digest, common_data)?;
        let fri_inferred_elements = self.get_inferred_elements(&challenges, common_data);
        let decompressed_proof =
            self.proof
                .decompress(&challenges, fri_inferred_elements, &common_data.fri_params);
        Ok(ProofWithPublicInputs {
            public_inputs: self.public_inputs,
            proof: decompressed_proof,
        })
    }

    pub(crate) fn verify(
        self,
        verifier_data: &VerifierOnlyCircuitData<C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<()> {
        ensure!(
            self.public_inputs.len() == common_data.num_public_inputs,
            "Number of public inputs doesn't match circuit data."
        );
        let public_inputs_hash = self.get_public_inputs_hash();
        let challenges = self.get_challenges(
            public_inputs_hash,
            &verifier_data.circuit_digest,
            common_data,
        )?;
        let fri_inferred_elements = self.get_inferred_elements(&challenges, common_data);
        let decompressed_proof =
            self.proof
                .decompress(&challenges, fri_inferred_elements, &common_data.fri_params);
        verify_with_challenges::<F, C, D>(
            decompressed_proof,
            public_inputs_hash,
            challenges,
            verifier_data,
            common_data,
        )
    }

    pub(crate) fn get_public_inputs_hash(
        &self,
    ) -> <<C as GenericConfig<D>>::InnerHasher as Hasher<F>>::Hash {
        C::InnerHasher::hash_no_pad(&self.public_inputs)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer
            .write_compressed_proof_with_public_inputs(self)
            .expect("Writing to a byte-vector cannot fail.");
        buffer
    }

    pub fn from_bytes(
        bytes: Vec<u8>,
        common_data: &CommonCircuitData<F, D>,
    ) -> anyhow::Result<Self> {
        let mut buffer = Buffer::new(&bytes);
        let proof = buffer
            .read_compressed_proof_with_public_inputs(common_data)
            .map_err(anyhow::Error::msg)?;
        Ok(proof)
    }
}

#[derive(Debug)]
pub struct ProofChallenges<F: RichField + Extendable<D>, const D: usize> {
    /// Random values used in Plonk's permutation argument.
    pub plonk_betas: Vec<F>,

    /// Random values used in Plonk's permutation argument.
    pub plonk_gammas: Vec<F>,

    /// Random values used to combine PLONK constraints.
    pub plonk_alphas: Vec<F>,

    /// Lookup challenges.
    pub plonk_deltas: Vec<F>,

    /// Point at which the PLONK polynomials are opened.
    pub plonk_zeta: F::Extension,

    pub fri_challenges: FriChallenges<F, D>,
}

pub(crate) struct ProofChallengesTarget<const D: usize> {
    pub plonk_betas: Vec<Target>,
    pub plonk_gammas: Vec<Target>,
    pub plonk_alphas: Vec<Target>,
    pub plonk_deltas: Vec<Target>,
    pub plonk_zeta: ExtensionTarget<D>,
    pub fri_challenges: FriChallengesTarget<D>,
}

/// Coset elements that can be inferred in the FRI reduction steps.
pub(crate) struct FriInferredElements<F: RichField + Extendable<D>, const D: usize>(
    pub Vec<F::Extension>,
);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProofWithPublicInputsTarget<const D: usize> {
    pub proof: ProofTarget<D>,
    pub public_inputs: Vec<Target>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
/// The purported values of each polynomial at a single point.
pub struct OpeningSet<F: RichField + Extendable<D>, const D: usize> {
    pub constants: Vec<F::Extension>,
    pub plonk_sigmas: Vec<F::Extension>,
    pub wires: Vec<F::Extension>,
    pub plonk_zs: Vec<F::Extension>,
    pub plonk_zs_next: Vec<F::Extension>,
    pub partial_products: Vec<F::Extension>,
    pub quotient_polys: Vec<F::Extension>,
    pub lookup_zs: Vec<F::Extension>,
    pub lookup_zs_next: Vec<F::Extension>,
}

impl<F: RichField + Extendable<D>, const D: usize> OpeningSet<F, D> {
    pub fn new<C: GenericConfig<D, F = F>>(
        zeta: F::Extension,
        g: F::Extension,
        constants_sigmas_commitment: &PolynomialBatch<F, C, D>,
        wires_commitment: &PolynomialBatch<F, C, D>,
        zs_partial_products_lookup_commitment: &PolynomialBatch<F, C, D>,
        quotient_polys_commitment: &PolynomialBatch<F, C, D>,
        common_data: &CommonCircuitData<F, D>,
    ) -> Self {
        // 定义一个闭包，用于评估多项式在给定点 z 处的值
        let eval_commitment = |z: F::Extension, c: &PolynomialBatch<F, C, D>| {
            c.polynomials
                .par_iter() // 并行迭代多项式
                .map(|p| p.to_extension().eval(z)) // 将多项式转换为扩展域并在 z 处评估
                .collect::<Vec<_>>() // 收集评估结果
        };

        // 评估常量和 σ 多项式在 zeta 处的值
        let constants_sigmas_eval = eval_commitment(zeta, constants_sigmas_commitment);

        // 评估置换参数和查找多项式在 zeta 处的值
        let zs_partial_products_lookup_eval =
            eval_commitment(zeta, zs_partial_products_lookup_commitment);
        // 评估置换参数和查找多项式在 g * zeta 处的值
        let zs_partial_products_lookup_next_eval =
            eval_commitment(g * zeta, zs_partial_products_lookup_commitment);
        // 评估商多项式在 zeta 处的值
        let quotient_polys = eval_commitment(zeta, quotient_polys_commitment);

        // 创建并返回 OpeningSet 实例
        Self {
            // 提取常量多项式的评估值
            constants: constants_sigmas_eval[common_data.constants_range()].to_vec(),
            // 提取 σ 多项式的评估值
            plonk_sigmas: constants_sigmas_eval[common_data.sigmas_range()].to_vec(),
            // 评估 wire 多项式在 zeta 处的值
            wires: eval_commitment(zeta, wires_commitment),
            // 提取置换参数多项式在 zeta 处的评估值
            plonk_zs: zs_partial_products_lookup_eval[common_data.zs_range()].to_vec(),
            // 提取置换参数多项式在 g * zeta 处的评估值
            plonk_zs_next: zs_partial_products_lookup_next_eval[common_data.zs_range()].to_vec(),
            // 提取部分积多项式的评估值
            partial_products: zs_partial_products_lookup_eval[common_data.partial_products_range()]
                .to_vec(),
            // 商多项式的评估值
            quotient_polys,
            // 提取查找多项式在 zeta 处的评估值
            lookup_zs: zs_partial_products_lookup_eval[common_data.lookup_range()].to_vec(),
            // 提取查找多项式在 g * zeta 处的评估值
            lookup_zs_next: zs_partial_products_lookup_next_eval[common_data.lookup_range()]
                .to_vec(),
        }
    }

    pub(crate) fn to_fri_openings(&self) -> FriOpenings<F, D> {
        // 检查是否存在查找多项式
        let has_lookup = !self.lookup_zs.is_empty();

        // 评估 zeta 批次
        let zeta_batch = if has_lookup {
            // 如果存在查找多项式，合并所有多项式的值
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(), // 常量多项式的值
                    self.plonk_sigmas.as_slice(), // σ 多项式的值
                    self.wires.as_slice(), // wire 多项式的值
                    self.plonk_zs.as_slice(), // 置换参数多项式的值
                    self.partial_products.as_slice(), // 部分积多项式的值
                    self.quotient_polys.as_slice(), // 商多项式的值
                    self.lookup_zs.as_slice(), // 查找多项式的值
                ]
                    .concat(), // 合并所有值
            }
        } else {
            // 如果不存在查找多项式，合并除查找多项式外的所有值
            FriOpeningBatch {
                values: [
                    self.constants.as_slice(), // 常量多项式的值
                    self.plonk_sigmas.as_slice(), // σ 多项式的值
                    self.wires.as_slice(), // wire 多项式的值
                    self.plonk_zs.as_slice(), // 置换参数多项式的值
                    self.partial_products.as_slice(), // 部分积多项式的值
                    self.quotient_polys.as_slice(), // 商多项式的值
                ]
                    .concat(), // 合并所有值
            }
        };

        // 评估 zeta_next 批次
        let zeta_next_batch = if has_lookup {
            // 如果存在查找多项式，合并置换参数多项式和查找多项式的值
            FriOpeningBatch {
                values: [self.plonk_zs_next.clone(), self.lookup_zs_next.clone()].concat(),
            }
        } else {
            // 如果不存在查找多项式，仅合并置换参数多项式的值
            FriOpeningBatch {
                values: self.plonk_zs_next.clone(),
            }
        };

        // 返回 FriOpenings 实例，包含 zeta 和 zeta_next 批次
        FriOpenings {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

/// The purported values of each polynomial at a single point.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpeningSetTarget<const D: usize> {
    pub constants: Vec<ExtensionTarget<D>>,
    pub plonk_sigmas: Vec<ExtensionTarget<D>>,
    pub wires: Vec<ExtensionTarget<D>>,
    pub plonk_zs: Vec<ExtensionTarget<D>>,
    pub plonk_zs_next: Vec<ExtensionTarget<D>>,
    pub lookup_zs: Vec<ExtensionTarget<D>>,
    pub next_lookup_zs: Vec<ExtensionTarget<D>>,
    pub partial_products: Vec<ExtensionTarget<D>>,
    pub quotient_polys: Vec<ExtensionTarget<D>>,
}

impl<const D: usize> OpeningSetTarget<D> {
    pub(crate) fn to_fri_openings(&self) -> FriOpeningsTarget<D> {
        let has_lookup = !self.lookup_zs.is_empty();
        let zeta_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                    self.lookup_zs.as_slice(),
                ]
                    .concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: [
                    self.constants.as_slice(),
                    self.plonk_sigmas.as_slice(),
                    self.wires.as_slice(),
                    self.plonk_zs.as_slice(),
                    self.partial_products.as_slice(),
                    self.quotient_polys.as_slice(),
                ]
                    .concat(),
            }
        };
        let zeta_next_batch = if has_lookup {
            FriOpeningBatchTarget {
                values: [self.plonk_zs_next.clone(), self.next_lookup_zs.clone()].concat(),
            }
        } else {
            FriOpeningBatchTarget {
                values: self.plonk_zs_next.clone(),
            }
        };
        FriOpeningsTarget {
            batches: vec![zeta_batch, zeta_next_batch],
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::sync::Arc;
    #[cfg(feature = "std")]
    use std::sync::Arc;

    use anyhow::Result;
    use itertools::Itertools;
    use plonky2_field::types::Sample;

    use super::*;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::gates::lookup_table::LookupTable;
    use crate::gates::noop::NoopGate;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::plonk::verifier::verify;

    #[test]
    fn test_proof_compression() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        // 配置电路参数
        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![1, 1]);
        config.fri_config.num_query_rounds = 50;

        // 创建部分见证
        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // 构建一个虚拟电路以获得有效证明
        let x = F::rand(); // 随机生成 x
        let y = F::rand(); // 随机生成 y
        let z = x * y; // 计算 z = x * y
        let xt = builder.constant(x); // 将 x 作为常量添加到电路中
        let yt = builder.constant(y); // 将 y 作为常量添加到电路中
        let zt = builder.constant(z); // 将 z 作为常量添加到电路中
        let comp_zt = builder.mul(xt, yt); // 在电路中计算 xt * yt
        builder.connect(zt, comp_zt); // 连接 zt 和 comp_zt，确保它们相等
        for _ in 0..100 {
            builder.add_gate(NoopGate, vec![]); // 添加 100 个 NoopGate 到电路中
        }
        let data = builder.build::<C>(); // 构建电路数据
        let proof = data.prove(pw)?; // 生成证明
        verify(proof.clone(), &data.verifier_only, &data.common)?; // 验证证明

        // 验证 `decompress ∘ compress = identity`
        let compressed_proof = data.compress(proof.clone())?; // 压缩证明
        let decompressed_compressed_proof = data.decompress(compressed_proof.clone())?; // 解压缩证明
        assert_eq!(proof, decompressed_compressed_proof); // 确认解压缩后的证明与原始证明相同

        verify(proof, &data.verifier_only, &data.common)?; // 再次验证原始证明
        data.verify_compressed(compressed_proof) // 验证压缩后的证明
    }

    #[test]
    fn test_proof_compression_lookup() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        use plonky2_field::types::Field;
        type F = <C as GenericConfig<D>>::F;

        let mut config = CircuitConfig::standard_recursion_config();
        config.fri_config.reduction_strategy = FriReductionStrategy::Fixed(vec![1, 1]);
        config.fri_config.num_query_rounds = 50;

        let pw = PartialWitness::new();
        let tip5_table = vec![
            0, 7, 26, 63, 124, 215, 85, 254, 214, 228, 45, 185, 140, 173, 33, 240, 29, 177, 176,
            32, 8, 110, 87, 202, 204, 99, 150, 106, 230, 14, 235, 128, 213, 239, 212, 138, 23, 130,
            208, 6, 44, 71, 93, 116, 146, 189, 251, 81, 199, 97, 38, 28, 73, 179, 95, 84, 152, 48,
            35, 119, 49, 88, 242, 3, 148, 169, 72, 120, 62, 161, 166, 83, 175, 191, 137, 19, 100,
            129, 112, 55, 221, 102, 218, 61, 151, 237, 68, 164, 17, 147, 46, 234, 203, 216, 22,
            141, 65, 57, 123, 12, 244, 54, 219, 231, 96, 77, 180, 154, 5, 253, 133, 165, 98, 195,
            205, 134, 245, 30, 9, 188, 59, 142, 186, 197, 181, 144, 92, 31, 224, 163, 111, 74, 58,
            69, 113, 196, 67, 246, 225, 10, 121, 50, 60, 157, 90, 122, 2, 250, 101, 75, 178, 159,
            24, 36, 201, 11, 243, 132, 198, 190, 114, 233, 39, 52, 21, 209, 108, 238, 91, 187, 18,
            104, 194, 37, 153, 34, 200, 143, 126, 155, 236, 118, 64, 80, 172, 89, 94, 193, 135,
            183, 86, 107, 252, 13, 167, 206, 136, 220, 207, 103, 171, 160, 76, 182, 227, 217, 158,
            56, 174, 4, 66, 109, 139, 162, 184, 211, 249, 47, 125, 232, 117, 43, 16, 42, 127, 20,
            241, 25, 149, 105, 156, 51, 53, 168, 145, 247, 223, 79, 78, 226, 15, 222, 82, 115, 70,
            210, 27, 41, 1, 170, 40, 131, 192, 229, 248, 255,
        ];
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config);
        let lut_index = builder.add_lookup_table_from_pairs(table);

        // Build dummy circuit with a lookup to get a valid proof.
        let x = F::TWO;
        let out = builder.constant(F::from_canonical_usize(26));

        let xt = builder.constant(x);
        let look_out = builder.add_lookup_from_index(xt, lut_index);
        builder.connect(look_out, out);
        for _ in 0..100 {
            builder.add_gate(NoopGate, vec![]);
        }
        let data = builder.build::<C>();

        let proof = data.prove(pw)?;
        verify(proof.clone(), &data.verifier_only, &data.common)?;

        // Verify that `decompress ∘ compress = identity`.
        let compressed_proof = data.compress(proof.clone())?;
        let decompressed_compressed_proof = data.decompress(compressed_proof.clone())?;
        assert_eq!(proof, decompressed_compressed_proof);

        verify(proof, &data.verifier_only, &data.common)?;
        data.verify_compressed(compressed_proof)
    }
}
