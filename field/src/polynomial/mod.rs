pub(crate) mod division;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::max;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use anyhow::{ensure, Result};
use itertools::Itertools;
use plonky2_util::log2_strict;
use serde::{Deserialize, Serialize};

use crate::extension::{Extendable, FieldExtension};
use crate::fft::{fft, fft_with_options, ifft, FftRootTable};
use crate::types::Field;

/// A polynomial in point-value form.
///
/// The points are implicitly `g^i`, where `g` generates the subgroup whose size equals the number
/// of points.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolynomialValues<F: Field> {
    pub values: Vec<F>,
}

impl<F: Field> PolynomialValues<F> {
    //创建一个新的 PolynomialValues 实例，并检查向量的长度是否为 2 的幂
    pub fn new(values: Vec<F>) -> Self {
        // Check that a subgroup exists of this size, which should be a power of two.
        debug_assert!(log2_strict(values.len()) <= F::TWO_ADICITY);
        PolynomialValues { values }
    }

    //创建一个所有值都相同的多项式
    pub fn constant(value: F, len: usize) -> Self {
        Self::new(vec![value; len])
    }

    pub fn zero(len: usize) -> Self {
        Self::constant(F::ZERO, len)
    }

    pub fn is_zero(&self) -> bool {
        self.values.iter().all(|x| x.is_zero())
    }

    /// Returns the polynomial whole value is one at the given index, and zero elsewhere.
    /// 创建一个在特定索引处为一，其余为零的多项式
    pub fn selector(len: usize, index: usize) -> Self {
        let mut result = Self::zero(len);
        result.values[index] = F::ONE;
        result
    }

    /// The number of values stored.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    ///对多项式进行逆快速傅里叶变换，返回系数形式的多项式
    pub fn ifft(self) -> PolynomialCoeffs<F> {
        ifft(self)
    }


    /// 对在子集的陪集上的点值表示多项式转化为子集上的多项式系数表示形式
    /// Returns the polynomial whose evaluation on the coset `shift*H` is `self`.
    pub fn coset_ifft(self, shift: F) -> PolynomialCoeffs<F> {
        let mut shifted_coeffs = self.ifft();
        shifted_coeffs
            .coeffs
            .iter_mut()
            .zip(shift.inverse().powers())
            .for_each(|(c, r)| {
                *c *= r;
            });
        shifted_coeffs
    }

    ///对多个多项式进行低度扩展
    pub fn lde_multiple(polys: Vec<Self>, rate_bits: usize) -> Vec<Self> {
        polys.into_iter().map(|p| p.lde(rate_bits)).collect()
    }

    ///对多项式进行低度扩展
    pub fn lde(self, rate_bits: usize) -> Self {
        let coeffs = ifft(self).lde(rate_bits);
        fft_with_options(coeffs, Some(rate_bits), None)
    }

    /// Low-degree extend `Self` (seen as evaluations over the subgroup) onto a coset.
    /// 将多项式扩展到一个余子群
    pub fn lde_onto_coset(self, rate_bits: usize) -> Self {
        let coeffs = ifft(self).lde(rate_bits);
        coeffs.coset_fft_with_options(F::coset_shift(), Some(rate_bits), None)
    }

    pub fn degree(&self) -> usize {
        self.degree_plus_one().saturating_sub(1)
    }

    pub fn degree_plus_one(&self) -> usize {
        self.clone().ifft().degree_plus_one()
    }

    /// Adds `rhs * rhs_weight` to `self`. Assumes `self.len() == rhs.len()`.
    /// rhs 乘以 rhs_weight 后加到当前多项式上
    pub fn add_assign_scaled(&mut self, rhs: &Self, rhs_weight: F) {
        self.values
            .iter_mut()
            .zip_eq(&rhs.values)
            .for_each(|(self_v, rhs_v)| *self_v += *rhs_v * rhs_weight)
    }
}

impl<F: Field> From<Vec<F>> for PolynomialValues<F> {
    fn from(values: Vec<F>) -> Self {
        Self::new(values)
    }
}

/// A polynomial in coefficient form.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PolynomialCoeffs<F: Field> {
    pub coeffs: Vec<F>,
}
///PolynomialCoeffs是一个表示多项式系数形式的结构体。系数形式意味着多项式的值是通过其系数来表示的。
impl<F: Field> PolynomialCoeffs<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        PolynomialCoeffs { coeffs }
    }

    /// The empty list of coefficients, which is the smallest encoding of the zero polynomial.
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn zero(len: usize) -> Self {
        Self::new(vec![F::ZERO; len])
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|x| x.is_zero())
    }

    /// The number of coefficients. This does not filter out any zero coefficients, so it is not
    /// necessarily related to the degree.
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    //返回多项式长度的对数
    pub fn log_len(&self) -> usize {
        log2_strict(self.len())
    }

    //将多项式分块
    pub fn chunks(&self, chunk_size: usize) -> Vec<Self> {
        self.coeffs
            .chunks(chunk_size)
            .map(|chunk| PolynomialCoeffs::new(chunk.to_vec()))
            .collect()
    }

    ///在点 x 处评估多项式，返回多项式在 x 处的值
    pub fn eval(&self, x: F) -> F {
        self.coeffs
            .iter() // 迭代多项式的系数
            .rev() // 反转迭代顺序，从最高次项开始
            .fold(F::ZERO, |acc, &c| acc * x + c)
        // 使使用折叠操作来计算多项式的值。F::ZERO 是初始值，
        // |acc, &c| acc * x + c 是折叠函数。在折叠操作的每一步中，累加器的当前值 (acc) 乘以 x，
        // 然后将当前系数 (c) 加到这个乘积上。这个过程对每个系数重复，从最高次项到常数项。
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    /// 在给定点的幂处评估多项式，返回多项式在给定点的值
    pub fn eval_with_powers(&self, powers: &[F]) -> F {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + c * x)
    }


    ///在基域点 x 处评估多项式
    pub fn eval_base<const D: usize>(&self, x: F::BaseField) -> F
    where
        F: FieldExtension<D>,
    {
        self.coeffs
            .iter() // 迭代多项式的系数
            .rev() // 反转迭代顺序，从最高次项开始
            .fold(F::ZERO, |acc, &c| acc.scalar_mul(x) + c) // 使用折叠操作计算多项式的值
    }

    /// Evaluate the polynomial at a point given its powers. The first power is the point itself, not 1.
    pub fn eval_base_with_powers<const D: usize>(&self, powers: &[F::BaseField]) -> F
    where
        F: FieldExtension<D>,
    {
        debug_assert_eq!(self.coeffs.len(), powers.len() + 1);
        let acc = self.coeffs[0];
        self.coeffs[1..]
            .iter()
            .zip(powers)
            .fold(acc, |acc, (&x, &c)| acc + x.scalar_mul(c))
    }

    pub fn lde_multiple(polys: Vec<&Self>, rate_bits: usize) -> Vec<Self> {
        polys.into_iter().map(|p| p.lde(rate_bits)).collect()
    }

    pub fn lde(&self, rate_bits: usize) -> Self {
        self.padded(self.len() << rate_bits)
    }

    pub fn pad(&mut self, new_len: usize) -> Result<()> {
        ensure!(
            new_len >= self.len(),
            "Trying to pad a polynomial of length {} to a length of {}.",
            self.len(),
            new_len
        );
        self.coeffs.resize(new_len, F::ZERO);
        Ok(())
    }

    pub fn padded(&self, new_len: usize) -> Self {
        let mut poly = self.clone();
        poly.pad(new_len).unwrap();
        poly
    }

    /// Removes any leading zero coefficients.
    /// 移除多项式的前导零系数
    pub fn trim(&mut self) {
        self.coeffs.truncate(self.degree_plus_one());
    }

    /// Removes some leading zero coefficients, such that a desired length is reached. Fails if a
    /// nonzero coefficient is encountered before then.
    /// 将多项式修剪到指定长度
    pub fn trim_to_len(&mut self, len: usize) -> Result<()> {
        ensure!(self.len() >= len);
        ensure!(self.coeffs[len..].iter().all(F::is_zero));
        self.coeffs.truncate(len);
        Ok(())
    }

    /// Removes any leading zero coefficients.
    /// 返回移除前导零系数后的多项式
    pub fn trimmed(&self) -> Self {
        let coeffs = self.coeffs[..self.degree_plus_one()].to_vec();
        Self { coeffs }
    }

    /// Degree of the polynomial + 1, or 0 for a polynomial with no non-zero coefficients.
    pub fn degree_plus_one(&self) -> usize {
        (0usize..self.len())
            .rev()
            .find(|&i| self.coeffs[i].is_nonzero())
            .map_or(0, |i| i + 1)
    }

    /// Leading coefficient.返回多项式的首项系数
    pub fn lead(&self) -> F {
        self.coeffs
            .iter()
            .rev()
            .find(|x| x.is_nonzero())
            .map_or(F::ZERO, |x| *x)
    }

    /// Reverse the order of the coefficients, not taking into account the leading zero coefficients.
    /// 返回系数顺序反转的多项式
    pub(crate) fn rev(&self) -> Self {
        Self::new(self.trimmed().coeffs.into_iter().rev().collect())
    }

    pub fn fft(self) -> PolynomialValues<F> {
        fft(self)
    }

    //带选项的快速傅里叶变换
    pub fn fft_with_options(
        self,
        zero_factor: Option<usize>,
        root_table: Option<&FftRootTable<F>>,
    ) -> PolynomialValues<F> {
        fft_with_options(self, zero_factor, root_table)
    }

    /// 多项式系数表示形式在原始的子群上的点值表示，转化为在子群的陪集上的点值表示
    pub fn coset_fft(&self, shift: F) -> PolynomialValues<F> {
        self.coset_fft_with_options(shift, None, None)
    }


    /// 多项式系数表示形式在原始的子群上的点值表示，转化为在子群的陪集上的点值表示
    /// 算法过程：先将多项式的系数跟移位值(s^0,s^1,s^2,...s^(n-1))进行Hadamard乘积（逐位相乘），然后进行快速傅里叶变换
    ///  Returns the evaluation of the polynomial on the coset `shift*H`.
    pub fn coset_fft_with_options(
        &self,
        shift: F,
        zero_factor: Option<usize>,
        root_table: Option<&FftRootTable<F>>,
    ) -> PolynomialValues<F> {
        // 创建一个容量为多项式系数长度的向量，用于存储修改后的多项式
        let mut modified_poly: Vec<F> = Vec::with_capacity(self.coeffs.len());

        // 获取移位值的幂迭代器
        let mut powers_iter = shift.powers();

        // 遍历多项式的每个系数
        for &c in &self.coeffs {
            // 获取当前幂值
            let r = powers_iter.next().unwrap();
            // 将当前幂值与系数相乘，并将结果添加到修改后的多项式中
            modified_poly.push(r * c);
        }

        // 将修改后的多项式向量转换为多项式类型
        let modified_poly: Self = modified_poly.into();
        // let modified_poly: Self = shift
        //     .powers()
        //     .zip(&self.coeffs)
        //     .map(|(r, &c)| r * c)
        //     .collect::<Vec<_>>()
        //     .into();
        //println!("modified_poly: {:?}", modified_poly);
        modified_poly.fft_with_options(zero_factor, root_table)
    }

    //将多项式转换为扩展域
    pub fn to_extension<const D: usize>(&self) -> PolynomialCoeffs<F::Extension>
    where
        F: Extendable<D>,
    {
        PolynomialCoeffs::new(self.coeffs.iter().map(|&c| c.into()).collect())
    }


    //将多项式与扩展域元素相乘
    pub fn mul_extension<const D: usize>(&self, rhs: F::Extension) -> PolynomialCoeffs<F::Extension>
    where
        F: Extendable<D>,
    {
        PolynomialCoeffs::new(self.coeffs.iter().map(|&c| rhs.scalar_mul(c)).collect())
    }
}

impl<F: Field> PartialEq for PolynomialCoeffs<F> {
    fn eq(&self, other: &Self) -> bool {
        let max_terms = self.coeffs.len().max(other.coeffs.len());
        for i in 0..max_terms {
            let self_i = self.coeffs.get(i).cloned().unwrap_or(F::ZERO);
            let other_i = other.coeffs.get(i).cloned().unwrap_or(F::ZERO);
            if self_i != other_i {
                return false;
            }
        }
        true
    }
}

impl<F: Field> Eq for PolynomialCoeffs<F> {}

impl<F: Field> From<Vec<F>> for PolynomialCoeffs<F> {
    fn from(coeffs: Vec<F>) -> Self {
        Self::new(coeffs)
    }
}

impl<F: Field> Add for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    fn add(self, rhs: Self) -> Self::Output {
        let len = max(self.len(), rhs.len());
        let a = self.padded(len).coeffs;
        let b = rhs.padded(len).coeffs;
        let coeffs = a.into_iter().zip(b).map(|(x, y)| x + y).collect();
        PolynomialCoeffs::new(coeffs)
    }
}

impl<F: Field> Sum for PolynomialCoeffs<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::empty(), |acc, p| &acc + &p)
    }
}

impl<F: Field> Sub for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    fn sub(self, rhs: Self) -> Self::Output {
        let len = max(self.len(), rhs.len());
        let mut coeffs = self.padded(len).coeffs;
        for (i, &c) in rhs.coeffs.iter().enumerate() {
            coeffs[i] -= c;
        }
        PolynomialCoeffs::new(coeffs)
    }
}

impl<F: Field> AddAssign for PolynomialCoeffs<F> {
    fn add_assign(&mut self, rhs: Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, r) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            *l += r;
        }
    }
}

impl<F: Field> AddAssign<&Self> for PolynomialCoeffs<F> {
    fn add_assign(&mut self, rhs: &Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, &r) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *l += r;
        }
    }
}

impl<F: Field> SubAssign for PolynomialCoeffs<F> {
    fn sub_assign(&mut self, rhs: Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, r) in self.coeffs.iter_mut().zip(rhs.coeffs) {
            *l -= r;
        }
    }
}

impl<F: Field> SubAssign<&Self> for PolynomialCoeffs<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        let len = max(self.len(), rhs.len());
        self.coeffs.resize(len, F::ZERO);
        for (l, &r) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *l -= r;
        }
    }
}

impl<F: Field> Mul<F> for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    fn mul(self, rhs: F) -> Self::Output {
        let coeffs = self.coeffs.iter().map(|&x| rhs * x).collect();
        PolynomialCoeffs::new(coeffs)
    }
}

impl<F: Field> MulAssign<F> for PolynomialCoeffs<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.coeffs.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<F: Field> Mul for &PolynomialCoeffs<F> {
    type Output = PolynomialCoeffs<F>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self::Output {
        let new_len = (self.len() + rhs.len()).next_power_of_two();
        let a = self.padded(new_len);
        let b = rhs.padded(new_len);
        let a_evals = a.fft();
        let b_evals = b.fft();

        let mul_evals: Vec<F> = a_evals
            .values
            .into_iter()
            .zip(b_evals.values)
            .map(|(pa, pb)| pa * pb)
            .collect();
        ifft(mul_evals.into())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use rand::rngs::OsRng;
    use rand::Rng;

    use super::*;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Sample;
    #[test]
    fn test_trimmed() {
        type F = GoldilocksField;

        assert_eq!(
            PolynomialCoeffs::<F> { coeffs: vec![] }.trimmed(),
            PolynomialCoeffs::<F> { coeffs: vec![] }
        );
        assert_eq!(
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ZERO]
            }
                .trimmed(),
            PolynomialCoeffs::<F> { coeffs: vec![] }
        );
        assert_eq!(
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ONE, F::TWO, F::ZERO, F::ZERO]
            }
                .trimmed(),
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ONE, F::TWO]
            }
        );
    }

    #[test]
    fn test_coset_fft() {
        type F = GoldilocksField;

        let k = 8; // 设置多项式的阶数为 2^8
        let n = 1 << k; // 计算多项式的长度 n = 2^k
        let poly = PolynomialCoeffs::new(F::rand_vec(n)); // 生成一个随机的多项式系数
        let shift = F::rand(); // 生成一个随机的陪集移位
        let coset_evals = poly.coset_fft(shift).values; // 对多项式进行陪集快速傅里叶变换，得到点值形式的多项式

        let generator = F::primitive_root_of_unity(k); // 获取 k 阶单位根
        let naive_coset_evals = F::cyclic_subgroup_coset_known_order(generator, shift, n) // 计算陪集上的点值
            .into_iter()
            .map(|x| poly.eval(x)) // 评估多项式在这些点上的值
            .collect::<Vec<_>>();
        assert_eq!(coset_evals, naive_coset_evals); // 验证评估结果与快速傅里叶变换的结果相同

        let ifft_coeffs = PolynomialValues::new(coset_evals).coset_ifft(shift); // 对点值形式的多项式进行陪集逆快速傅里叶变换，得到系数形式的多项式
        assert_eq!(poly, ifft_coeffs); // 验证逆变换后的系数与原始多项式系数相同
    }

    #[test]
    fn test_coset_ifft() {
        type F = GoldilocksField;

        let k = 8; // 设置多项式的阶数为 2^8
        let n = 1 << k; // 计算多项式的长度 n = 2^k
        let evals = PolynomialValues::new(F::rand_vec(n)); // 生成一个随机的多项式值
        let shift = F::rand(); // 生成一个随机的移位值
        let coeffs = evals.clone().coset_ifft(shift); // 对多项式进行陪集逆快速傅里叶变换，得到系数形式的多项式

        let generator = F::primitive_root_of_unity(k); // 获取 k 阶单位根
        let naive_coset_evals = F::cyclic_subgroup_coset_known_order(generator, shift, n) // 计算陪集上的点值
            .into_iter()
            .map(|x| coeffs.eval(x)) // 评估多项式在这些点上的值
            .collect::<Vec<_>>();
        assert_eq!(evals, naive_coset_evals.into()); // 验证评估结果与原始多项式值相同

        let fft_evals = coeffs.coset_fft(shift); // 对系数形式的多项式进行陪集快速傅里叶变换，得到点值形式的多项式
        assert_eq!(evals, fft_evals); // 验证变换后的点值与原始多项式值相同
    }
    #[test]
    fn test_lde() {
        type F = GoldilocksField;
        let k = 8; // Set the degree of the polynomial to 2^8
        let n = 1 << k; // Calculate the length of the polynomial n = 2^k
        let poly = PolynomialValues::new(F::rand_vec(n)); // Generate a random polynomial
        let rate_bits = 2; // Set the rate bits for low-degree extension
        let lde_poly = poly.clone().lde(rate_bits); // Perform low-degree extension on the polynomial

        // Verify the length of the extended polynomial
        assert_eq!(lde_poly.len(), poly.len() << rate_bits);


        //assert_eq!(lde_poly.values, naive_lde_evals); // Verify the values match the expected results
    }
    #[test]
    fn test_polynomial_multiplication() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000)); // 随机生成两个多项式的度数
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg)); // 生成第一个随机多项式
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg)); // 生成第二个随机多项式
        let m1 = &a * &b; // 计算两个多项式的乘积
        let m2 = &a * &b; // 再次计算两个多项式的乘积
        for _ in 0..1000 {
            let x = F::rand(); // 生成一个随机点
            assert_eq!(m1.eval(x), a.eval(x) * b.eval(x)); // 验证在随机点处的乘积结果
            assert_eq!(m2.eval(x), a.eval(x) * b.eval(x)); // 再次验证在随机点处的乘积结果
        }
    }

    #[test]
    fn test_inv_mod_xn() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let a_deg = rng.gen_range(0..1_000); // 随机生成多项式的度数
        let n = rng.gen_range(1..1_000); // 随机生成模数
        let mut a = PolynomialCoeffs::new(F::rand_vec(a_deg + 1)); // 生成一个随机多项式
        if a.coeffs[0].is_zero() {
            a.coeffs[0] = F::ONE; // 确保多项式的首项系数非零
        }
        let b = a.inv_mod_xn(n); // 计算多项式在模 x^n 下的逆
        let mut m = &a * &b; // 计算多项式与其逆的乘积
        m.coeffs.truncate(n); // 截断多项式到长度 n
        m.trim(); // 移除多项式的前导零系数
        assert_eq!(
            m,
            PolynomialCoeffs::new(vec![F::ONE]), // 验证乘积结果是否为 1
            "a: {:#?}, b:{:#?}, n:{:#?}, m:{:#?}",
            a,
            b,
            n,
            m
        );
    }

    #[test]
    fn test_polynomial_long_division() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000)); // 随机生成两个多项式的度数
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg)); // 生成第一个随机多项式
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg)); // 生成第二个随机多项式
        let (q, r) = a.div_rem_long_division(&b); // 进行多项式长除法，得到商和余数
        for _ in 0..1000 {
            let x = F::rand(); // 生成一个随机点
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x)); // 验证在随机点处的除法结果
        }
    }

    #[test]
    fn test_polynomial_division() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000)); // 随机生成两个多项式的度数
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg)); // 生成第一个随机多项式
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg)); // 生成第二个随机多项式
        let (q, r) = a.div_rem(&b); // 进行多项式除法，得到商和余数
        for _ in 0..1000 {
            let x = F::rand(); // 生成一个随机点
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x)); // 验证在随机点处的除法结果
        }
    }

    #[test]
    fn test_polynomial_division_by_constant() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let a_deg = rng.gen_range(1..10_000); // 随机生成多项式的度数
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg)); // 生成一个随机多项式
        let b = PolynomialCoeffs::from(vec![F::rand()]); // 生成一个常数多项式
        let (q, r) = a.div_rem(&b); // 进行多项式除法，得到商和余数
        for _ in 0..1000 {
            let x = F::rand(); // 生成一个随机点
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x)); // 验证在随机点处的除法结果
        }
    }

    // 测试哪种多项式除法方法对于 (X^n - 1)/(X - a) 类型的除法更快
    #[test]
    fn test_division_linear() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let l = 14; // 设置多项式的阶数为 2^14
        let n = 1 << l; // 计算多项式的长度 n = 2^l
        let g = F::primitive_root_of_unity(l); // 获取 l 阶单位根
        let xn_minus_one = {
            let mut xn_min_one_vec = vec![F::ZERO; n + 1]; // 初始化一个长度为 n+1 的零向量
            xn_min_one_vec[n] = F::ONE; // 设置最高次项系数为 1
            xn_min_one_vec[0] = F::NEG_ONE; // 设置常数项系数为 -1
            PolynomialCoeffs::new(xn_min_one_vec) // 生成多项式 X^n - 1
        };

        let a = g.exp_u64(rng.gen_range(0..(n as u64))); // 生成一个随机点 a
        let denom = PolynomialCoeffs::new(vec![-a, F::ONE]); // 生成多项式 X - a
        let now = Instant::now();
        xn_minus_one.div_rem(&denom); // 进行多项式除法
        println!("Division time: {:?}", now.elapsed()); // 输出除法时间
        let now = Instant::now();
        xn_minus_one.div_rem_long_division(&denom); // 进行多项式长除法
        println!("Division time: {:?}", now.elapsed()); // 输出长除法时间
    }

    #[test]
    fn eq() {
        type F = GoldilocksField;
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO, F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ONE]),
            PolynomialCoeffs::new(vec![F::ONE, F::ZERO])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![F::ONE])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO, F::ONE])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ONE, F::ZERO])
        );
    }
}
