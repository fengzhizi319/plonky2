use alloc::vec::Vec;

use crate::packed::PackedField;
use crate::types::Field;

/// 预计算 `Z_H(X) = X^n - 1` 在陪集 `gK` 上的值，其中 `H <= K`。
#[derive(Debug)]
pub struct ZeroPolyOnCoset<F: Field> {
    /// `n = |H|`。
    n: F,
    /// `rate = |K|/|H|`。
    rate: usize,
    /// 存储 ` g^n * v^i - 1`，其中 `i` 在 `0..rate` 范围内，`w` 是 `K` 的生成元，`v` 是 `rate` 阶原根。
    evals: Vec<F>,
    /// 存储 `evals` 的乘法逆。
    inverses: Vec<F>,
}

impl<F: Field> ZeroPolyOnCoset<F> {
    /// 得到一个子群的陪集。每个元素都减去1，并输出每个元素的逆。
    pub fn new(n_log: usize, rate_bits: usize) -> Self {
        // 计算 g 的 n 次幂，其中 g 是 coset_shift，n 是 2 的 n_log 次幂
        let g_pow_n = F::coset_shift().exp_power_of_2(n_log); // g^2^n_log = g^n = g^4

        // 计算 coset 陪集并减去 1
        // let evals = F::two_adic_subgroup(rate_bits) // rate_bits = 3
        //     .into_iter()
        //     .map(|x| g_pow_n * x - F::ONE)
        //     .collect::<Vec<_>>();
        let subgroup = F::two_adic_subgroup(rate_bits); // rate_bits = 3
        //subgroup[1,18446744069431361537,281474976710656,1099511627520,...,18446742969902956801] ,size=8
        //println!("subgroup: {:?}", subgroup);
        let mut evals = Vec::with_capacity(subgroup.len());
        for x in subgroup {
            evals.push(g_pow_n * x - F::ONE);
        }
        // 计算 evals 的乘法逆
        let inverses = F::batch_multiplicative_inverse(&evals);

        // 返回 ZeroPolyOnCoset 结构体实例
        Self {
            n: F::from_canonical_usize(1 << n_log), // n 是 2 的 n_log 次幂
            rate: 1 << rate_bits, // rate 是 2 的 rate_bits 次幂
            evals, // 预计算的 evals
            inverses, // evals 的乘法逆
        }
    }

    /// 返回 `Z_H(g * w^i)`。
    pub fn eval(&self, i: usize) -> F {
        self.evals[i % self.rate]
    }

    /// 返回 `1 / Z_H(g * w^i)`。
    pub fn eval_inverse(&self, i: usize) -> F {
        self.inverses[i % self.rate]
    }

    /// 类似于 `eval_inverse`，但适用于从 `i_start` 开始的一系列索引。
    pub fn eval_inverse_packed<P: PackedField<Scalar = F>>(&self, i_start: usize) -> P {
        let mut packed = P::ZEROS;
        packed
            .as_slice_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(j, packed_j)| *packed_j = self.eval_inverse(i_start + j));
        packed
    }

    /// 返回 `L_0(x) = Z_H(x)/(n * (x - 1))`，其中 `x = w^i`。
    pub fn eval_l_0(&self, i: usize, x: F) -> F {
        // 也可以使用蒙哥马利算法预计算逆
        self.eval(i) * (self.n * (x - F::ONE)).inverse()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::{Field, Sample}; // Adjust the import path as necessary

    #[test]
    fn test_zero_poly_on_coset_new() {
        type F = GoldilocksField;
        // Define the parameters for the test
        let n_log = 2; // 2^2 = 4
        let rate_bits = 3; // 4>>3 = 32

        // Create a new ZeroPolyOnCoset instance
        let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);

        // Verify the values of the fields
        assert_eq!(zero_poly.n, F::from_canonical_usize(1 << n_log));
        assert_eq!(zero_poly.rate, 1 << rate_bits);

        // Verify the length of evals and inverses
        assert_eq!(zero_poly.evals.len(), zero_poly.rate);
        assert_eq!(zero_poly.inverses.len(), zero_poly.rate);

        // Verify that evals and inverses are correctly computed
        for i in 0..zero_poly.rate {
            let eval = zero_poly.eval(i);
            let inverse = zero_poly.eval_inverse(i);
            assert_eq!(eval * inverse, F::ONE);
        }
    }

    #[test]
    fn test_eval() {
        type F = GoldilocksField;
        let n_log = 2;
        let rate_bits = 3;
        let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);

        for i in 0..zero_poly.rate {
            let eval = zero_poly.eval(i);
            assert_eq!(eval, zero_poly.evals[i % zero_poly.rate]);
        }
    }

    #[test]
    fn test_eval_inverse() {
        type F = GoldilocksField;
        let n_log = 2;
        let rate_bits = 3;
        let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);

        for i in 0..zero_poly.rate {
            let inverse = zero_poly.eval_inverse(i);
            assert_eq!(inverse, zero_poly.inverses[i % zero_poly.rate]);
        }
    }

    #[test]
    fn test_eval_inverse_packed() {
        type F = GoldilocksField;
        type P = GoldilocksField; // Replace with the actual packed field type
        let n_log = 2;
        let rate_bits = 3;
        let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);

        let i_start = 0;
        let packed = zero_poly.eval_inverse_packed::<P>(i_start);
        for (j, packed_j) in packed.as_slice().iter().enumerate() {
            assert_eq!(*packed_j, zero_poly.eval_inverse(i_start + j));
        }
    }

    #[test]
    fn test_eval_l_0() {
        type F = GoldilocksField;
        let n_log = 2;
        let rate_bits = 3;
        let zero_poly = ZeroPolyOnCoset::<F>::new(n_log, rate_bits);

        for i in 0..zero_poly.rate {
            let x = F::rand(); // Generate a random point
            let l_0 = zero_poly.eval_l_0(i, x);
            let expected = zero_poly.eval(i) * (zero_poly.n * (x - F::ONE)).inverse();
            assert_eq!(l_0, expected);
        }
    }
}