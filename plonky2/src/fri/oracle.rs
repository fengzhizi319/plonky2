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
    /// 从多项式值表示创建一个多项式系数表示，先从点值表示转到系数表述，先进行低度扩展，在转到陪集上，然后进行FFT变换，在建立merkle树承诺
    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,//包含多项式值的向量，已经被陪集掩码过
        rate_bits: usize,
        blinding: bool,//是否启用盲化
        cap_height: usize,//FRI 配置中的 cap 高度
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        //使用 IFFT（逆快速傅里叶变换）将多项式点值表示转话为系数表示
        //PolynomialValues { values: [0, 0, 12460551030817792791, 0] }->coeffs: [7726823775058094278, 10719920294356490043, 7726823775058094278, 10719920294356490043] }
        let coeffs = timed!(
            timing,
            "IFFT",
            values.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>()
        );

        /*
        //单线程调试模式
        let mut coeffs = Vec::new();
        for v in values {
            //PolynomialValues { values: [0, 0, 12460551030817792791, 0] }
            let tmp=v.ifft();
            //PolynomialCoeffs { coeffs: [7726823775058094278, 10719920294356490043, 7726823775058094278, 10719920294356490043] }

            coeffs.push(tmp);
        }
         */
        let coeffs = timed!(timing, "IFFT", coeffs);
        // 从多项式系数创建多项式FRI承诺
        Self::from_coeffs(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    ///先进行低度扩展，在转到陪集上，然后进行FFT变换，在建立merkle树承诺
    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {

        let degree = polynomials[0].len(); // 84=4+80

        // 先进行低度扩展，在转到陪集上，然后进行FFT变换
        let lde_values = timed!(
        timing,
        "FFT + blinding",
        Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
    );

        // 转置 LDE 值
        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        //let l1=leaves.clone();

        // 反转索引位
        reverse_index_bits_in_place(&mut leaves);

        // 构建 Merkle 树
        let merkle_tree = timed!(
        timing,
        "build Merkle tree",
        MerkleTree::new(leaves, cap_height)
    );

        // 返回包含Merkle的多项式承诺以及多项式
        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    ///先进行低度扩展，在转到陪集上，然后进行FFT变换
    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        // 获取多项式的度
        let degree = polynomials[0].len();

        // 如果启用盲化，则在每个叶子向量中添加4个随机元素作为盐
        let salt_size = if blinding { SALT_SIZE } else { 0 };
        /*
                polynomials
                    .par_iter()
                    .map(|p| {
                        // 确保所有多项式的度一致
                        assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                        //p.lde(rate_bits)元素个数扩展到以前的2^lde,用0进行填充
                        //coset_fft_with_options先转到陪集上，然后进行FFT变换
                        p.lde(rate_bits)
                            .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                            .values
                    })
                    .chain(
                        //在这个示例中，a.iter() 和 b.iter() 是两个迭代器，.chain(b.iter()) 将它们连接在一起，形成一个新的迭代器。
                        // collect() 方法将这个新的迭代器收集成一个向量。最终输出的向量包含了 a 和 b 中的所有元素
                        // 如果启用盲化，则生成随机向量并添加到结果中，数据维度保持一致
                        (0..salt_size)
                            .into_par_iter()
                            .map(|_| F::rand_vec(degree << rate_bits)),
                    )
                    .collect()


         */


        let mut lde_values = Vec::new();
        for p in polynomials {
            // Ensure all polynomials have the same degree
            assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
            // Compute the LDE values and perform the coset FFT transformation
            //p.lde(rate_bits)元素个数扩展到以前的2^lde,用0进行填充
            //PolynomialCoeffs { coeffs: [9223372035781033985, 13835058052060954625, 18446744068340842498, 13835058052060921857] }
            let p_lde = p.lde(rate_bits);
            //p_lde:PolynomialCoeffs { coeffs: [9223372035781033985, 13835058052060954625, 18446744068340842498, 13835058052060921857, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
            //先把点值多项式转化为陪集上，然后在陪集上进行FFT变换
            let coset_fft_values = p_lde.coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table);
            // println!("coset_fft_values:{:?}",coset_fft_values);
            // println!("values:{:?}",coset_fft_values.values);
            let lde = coset_fft_values.values;
            lde_values.push(lde);
        }

        if blinding {
            for _ in 0..salt_size {
                lde_values.push(F::rand_vec(degree << rate_bits));
            }
        }
        lde_values

    }

    /// 获取`index * step`th point在 LDE（低度扩展）值，返回一个切片，包含指定索引和步长的 LDE 值
    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        // 计算实际索引
        let index = index * step;
        // 反转索引的位
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        // 获取 Merkle 树中对应索引的叶子节点
        let slice = &self.merkle_tree.leaves[index];
        // 返回切片，去掉盲化部分（如果启用盲化）
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
        // 确保维度 D 大于 1
        assert!(D > 1, "Not implemented for D=1.");

        // 获取扩展域中的随机挑战 alpha
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // 初始化最终的低度多项式
        let mut final_poly = PolynomialCoeffs::empty();

        // 每个批次 `i` 包含一个开点 `z_i` 和要在该点打开的多项式 `{f_ij}_j`
        // 对于每个批次，我们计算组合多项式 `F_i = sum alpha^j f_ij`
        // 最终多项式计算为 `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // 其中 `k_i` 被选择为每个 alpha 的幂仅在最终和中出现一次
        // 通常有两个批次用于在 `zeta` 和 `g * zeta` 处的开口
        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // 收集所有多项式的系数
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });

            // 计算组合多项式
            let composition_poly = timed!(
            timing,
            &format!("reduce batch of {} polynomials", polynomials.len()),
            alpha.reduce_polys_base(polys_coeff)
        );

            // 计算商多项式
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // 填充为二的幂次
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        // 计算最终多项式的低度扩展
        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
        timing,
        &format!("perform final FFT {}", lde_final_poly.len()),
        lde_final_poly.coset_fft(F::coset_shift().into())
    );

        // 生成 FRI 证明
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
#[cfg(test)]
mod tests {
    use super::*;
    use plonky2_field::types::Field;
    use plonky2_field::goldilocks_field::GoldilocksField;
    use crate::field::polynomial::PolynomialCoeffs;
    use crate::plonk::config::PoseidonGoldilocksConfig;
    use crate::util::timing::TimingTree;

    #[test]
    fn test_lde_values() {
        //type F = GoldilocksField;
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let polynomials = vec![
            PolynomialCoeffs::new(vec![F::ONE, F::TWO, F::ONE, F::TWO]),
            PolynomialCoeffs::new(vec![F::ONE, F::TWO, F::ONE, F::TWO]),
        ];
        println!("polynomials:{:?}",polynomials);


        let rate_bits = 3;
        let blinding = true;
        let fft_root_table = None;


        let lde_values = PolynomialBatch::<F, C, 2>::lde_values(&polynomials, rate_bits, blinding, fft_root_table);

        // Add assertions to verify the correctness of the lde_values
        assert_eq!(lde_values.len(), polynomials.len());
        for lde in lde_values {
            assert_eq!(lde.len(), polynomials[0].len() << rate_bits);
        }
    }
}