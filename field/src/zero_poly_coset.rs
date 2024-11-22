use alloc::vec::Vec;

use crate::packed::PackedField;
use crate::types::Field;

/// Precomputations of the evaluation of `Z_H(X) = X^n - 1` on a coset `gK` with `H <= K`.
#[derive(Debug)]
pub struct ZeroPolyOnCoset<F: Field> {
    /// `n = |H|`.
    n: F,
    /// `rate = |K|/|H|`.
    rate: usize,
    /// Holds `g^n * (w^n)^i - 1 = g^n * v^i - 1` for `i in 0..rate`, with `w` a generator of `K` and `v` a
    /// `rate`-primitive root of unity.
    evals: Vec<F>,
    /// Holds the multiplicative inverses of `evals`.
    inverses: Vec<F>,
}

impl<F: Field> ZeroPolyOnCoset<F> {
    ///得到一个子群的coset。每个元素都减去1，还输出每个元素的逆
    pub fn new(n_log: usize, rate_bits: usize) -> Self {
    // 计算 g 的 n 次幂，其中 g 是 coset_shift，n 是 2 的 n_log 次幂
    let g_pow_n = F::coset_shift().exp_power_of_2(n_log);//g^2^n_log=g^n=g^4

    // coset陪集-1
    let evals = F::two_adic_subgroup(rate_bits)//rate_bits=3
        .into_iter()
        .map(|x| g_pow_n * x - F::ONE)
        .collect::<Vec<_>>();

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

    /// Returns `Z_H(g * w^i)`.
    pub fn eval(&self, i: usize) -> F {
        self.evals[i % self.rate]
    }

    /// Returns `1 / Z_H(g * w^i)`.
    pub fn eval_inverse(&self, i: usize) -> F {
        self.inverses[i % self.rate]
    }

    /// Like `eval_inverse`, but for a range of indices starting with `i_start`.
    pub fn eval_inverse_packed<P: PackedField<Scalar = F>>(&self, i_start: usize) -> P {
        let mut packed = P::ZEROS;
        packed
            .as_slice_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(j, packed_j)| *packed_j = self.eval_inverse(i_start + j));
        packed
    }

    /// Returns `L_0(x) = Z_H(x)/(n * (x - 1))` with `x = w^i`.
    pub fn eval_l_0(&self, i: usize, x: F) -> F {
        // Could also precompute the inverses using Montgomery.
        self.eval(i) * (self.n * (x - F::ONE)).inverse()
    }
}
