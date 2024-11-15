#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
PolynomialBatch<F, C, D>
{
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    /// 从多项式值表示创建一个多项式系数表示
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,//包含多项式值的向量，已经被陪集掩码过
        rate_bits: usize,
        blinding: bool,//是否启用盲化
        cap_height: usize,//FRI 配置中的 cap 高度
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        // 使用 IFFT（逆快速傅里叶变换）将多项式值转换为多项式系数，下面是原始代码：
        //     let coeffs = timed!(
        //     timing,
        //     "IFFT",
        //     values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        // );
        // 使用 IFFT（逆快速傅里叶变换）将多项式值转换为多项式系数，下面改成单线程，方便调试的代码
        let mut coeffs = Vec::new();
        for v in values {
            //println!("v:{:?}",v);
            let tmp=v.ifft();
            //println!("tmp:{:?}",tmp);
            coeffs.push(tmp);
        }
        let coeffs = timed!(timing, "IFFT", coeffs);
        //println!("coeffs:{:?}",timing);
        //结束调试
        // 从多项式系数创建多项式承诺
        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    ///
    /// # 参数
    ///
    /// * `polynomials` - 包含多项式系数的向量
    /// * `rate_bits` - FRI 配置中的速率位数
    /// * `blinding` - 是否启用盲化
    /// * `cap_height` - FRI 配置中的 cap 高度
    /// * `timing` - 用于记录时间的计时树
    /// * `fft_root_table` - FFT 根表的可选引用
    ///
    /// # 返回值
    ///
    /// 返回一个新的批量多项式承诺
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        // 获取多项式的度
        let degree = polynomials[0].len(); // 84=4+80

        // 使用 FFT 和盲化生成 LDE（低度扩展）值
        let lde_values = timed!(
        timing,
        "FFT + blinding",
        Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
    );

        // 转置 LDE 值
        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        // 反转索引位
        reverse_index_bits_in_place(&mut leaves);

        // 构建 Merkle 树
        let merkle_tree = timed!(
        timing,
        "build Merkle tree",
        MerkleTree::new(leaves, cap_height)
    );

        // 返回包含多项式批次的结构体
        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    /// 计算多项式的低度扩展（LDE）值
    ///
    /// # 参数
    ///
    /// * `polynomials` - 包含多项式系数的切片
    /// * `rate_bits` - FRI 配置中的速率位数
    /// * `blinding` - 是否启用盲化
    /// * `fft_root_table` - FFT 根表的可选引用
    ///
    /// # 返回值
    ///
    /// 返回一个包含 LDE 值的向量
    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        // 获取多项式的度
        let degree = polynomials[0].len();

        // 如果启用盲化，则在每个叶子向量中添加两个随机元素作为盐
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        polynomials
            .par_iter()
            .map(|p| {
                // 确保所有多项式的度一致
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                // 计算多项式的 LDE 值，并进行余弦 FFT 变换
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                // 如果启用盲化，则生成随机向量并添加到结果中
                (0..salt_size)
                    .into_par_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.leaves[index];
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            timing,
        );

        fri_proof
    }
}
