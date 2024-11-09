//! Concrete instantiation of a hash function.
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::fmt::Debug;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::{HashOut, HashOutTarget, RichField, NUM_HASH_OUT_ELTS};
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::AlgebraicHasher;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    pub fn hash_or_noop<H: AlgebraicHasher<F>>(&mut self, inputs: Vec<Target>) -> HashOutTarget {
        let zero = self.zero();
        if inputs.len() <= NUM_HASH_OUT_ELTS {
            HashOutTarget::from_partial(&inputs, zero)
        } else {
            self.hash_n_to_hash_no_pad::<H>(inputs)
        }
    }

    pub fn hash_n_to_hash_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
    ) -> HashOutTarget {
        HashOutTarget::from_vec(self.hash_n_to_m_no_pad::<H>(inputs, NUM_HASH_OUT_ELTS))
    }

    pub fn hash_n_to_m_no_pad<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
        num_outputs: usize,
    ) -> Vec<Target> {
        // 获取零值
        let zero = self.zero();
        // 初始化状态，使用零值填充，H::AlgebraicPermutation::WIDTH=8
        let mut state = H::AlgebraicPermutation::new(core::iter::repeat(zero));
         //print_state
         println!("state1:{:?}",state);
        //print H::AlgebraicPermutation::WIDTH
        println!("H::AlgebraicPermutation::WIDTH:{:?}",H::AlgebraicPermutation::WIDTH);
        println!("H::AlgebraicPermutation::RATE:{:?}",H::AlgebraicPermutation::RATE);
        // 吸收所有输入块
        //H::AlgebraicPermutation::RATE=8
        //通过 chunks 方法将 inputs 向量按 H::AlgebraicPermutation::RATE 的大小进行分块，并遍历每个分块进行处理。
        for input_chunk in inputs.chunks(H::AlgebraicPermutation::RATE) {
            // 用输入覆盖前 r 个元素。这与标准的海绵函数不同，标准海绵函数会对输入进行异或或加法操作。
            // 这种变体有时被称为“覆盖模式”。
            //println!("input_chunk:{:?}",input_chunk);
            state.set_from_slice(input_chunk, 0);
            //println!("state2:{:?}",state);
            //self.print_gates();
            self.print_copy_constraints();
            // 对状态进行置换，新增加一个PoseidonGate进行copy约束，copy约束增加到copy_constraints中
            state = self.permute::<H>(state);
            //self.print_gates();
            self.print_copy_constraints();
            //println!("state3:{:?}",state);
        }

        // 挤出直到我们得到所需数量的输出
        let mut outputs = Vec::with_capacity(num_outputs);
        loop {
            // 从状态中挤出元素
            for &s in state.squeeze() {
                println!("s:{:?}",s);
                outputs.push(s);
                // 如果输出数量达到要求，返回输出
                if outputs.len() == num_outputs {
                    return outputs;
                }
            }
            // 对状态进行再次置换
            state = self.permute::<H>(state);
        }
    }
}

/// Permutation that can be used in the sponge construction for an algebraic hash.
pub trait PlonkyPermutation<T: Copy + Default>:
    AsRef<[T]> + Copy + Debug + Default + Eq + Sync + Send
{
    const RATE: usize;
    const WIDTH: usize;

    /// Initialises internal state with values from `iter` until
    /// `iter` is exhausted or `Self::WIDTH` values have been
    /// received; remaining state (if any) initialised with
    /// `T::default()`. To initialise remaining elements with a
    /// different value, instead of your original `iter` pass
    /// `iter.chain(core::iter::repeat(F::from_canonical_u64(12345)))`
    /// or similar.
    fn new<I: IntoIterator<Item = T>>(iter: I) -> Self;

    /// Set idx-th state element to be `elt`. Panics if `idx >= WIDTH`.
    fn set_elt(&mut self, elt: T, idx: usize);

    /// Set state element `i` to be `elts[i] for i =
    /// start_idx..start_idx + n` where `n = min(elts.len(),
    /// WIDTH-start_idx)`. Panics if `start_idx > WIDTH`.
    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize);

    /// Same semantics as for `set_from_iter` but probably faster than
    /// just calling `set_from_iter(elts.iter())`.
    fn set_from_slice(&mut self, elts: &[T], start_idx: usize);

    /// Apply permutation to internal state
    fn permute(&mut self);

    /// Return a slice of `RATE` elements
    fn squeeze(&self) -> &[T];
}

/// A one-way compression function which takes two ~256 bit inputs and returns a ~256 bit output.
pub fn compress<F: Field, P: PlonkyPermutation<F>>(x: HashOut<F>, y: HashOut<F>) -> HashOut<F> {
    // TODO: With some refactoring, this function could be implemented as
    // hash_n_to_m_no_pad(chain(x.elements, y.elements), NUM_HASH_OUT_ELTS).

    debug_assert_eq!(x.elements.len(), NUM_HASH_OUT_ELTS);
    debug_assert_eq!(y.elements.len(), NUM_HASH_OUT_ELTS);
    debug_assert!(P::RATE >= NUM_HASH_OUT_ELTS);

    let mut perm = P::new(core::iter::repeat(F::ZERO));
    perm.set_from_slice(&x.elements, 0);
    perm.set_from_slice(&y.elements, NUM_HASH_OUT_ELTS);

    perm.permute();

    HashOut {
        elements: perm.squeeze()[..NUM_HASH_OUT_ELTS].try_into().unwrap(),
    }
}

/// Hash a message without any padding step. Note that this can enable length-extension attacks.
/// However, it is still collision-resistant in cases where the input has a fixed length.
pub fn hash_n_to_m_no_pad<F: RichField, P: PlonkyPermutation<F>>(
    inputs: &[F],
    num_outputs: usize,
) -> Vec<F> {
    let mut perm = P::new(core::iter::repeat(F::ZERO));

    // Absorb all input chunks.
    for input_chunk in inputs.chunks(P::RATE) {
        perm.set_from_slice(input_chunk, 0);
        perm.permute();
    }

    // Squeeze until we have the desired number of outputs.
    let mut outputs = Vec::new();
    loop {
        for &item in perm.squeeze() {
            outputs.push(item);
            if outputs.len() == num_outputs {
                return outputs;
            }
        }
        perm.permute();
    }
}

pub fn hash_n_to_hash_no_pad<F: RichField, P: PlonkyPermutation<F>>(inputs: &[F]) -> HashOut<F> {
    HashOut::from_vec(hash_n_to_m_no_pad::<F, P>(inputs, NUM_HASH_OUT_ELTS))
}
