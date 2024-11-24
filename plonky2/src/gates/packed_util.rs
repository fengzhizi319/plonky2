#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::field::extension::Extendable;
use crate::field::packable::Packable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::plonk::vars::{EvaluationVarsBaseBatch, EvaluationVarsBasePacked};

pub trait PackedEvaluableBase<F: RichField + Extendable<D>, const D: usize>: Gate<F, D> {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars_base: EvaluationVarsBasePacked<P>,
        yield_constr: StridedConstraintConsumer<P>,
    );

    ///把vars_packed中的const-wire的差值写入res中，并返回res
    /// Evaluates entire batch of points. Returns a matrix of constraints. Constraint `j` for point
    /// `i` is at `index j * batch_size + i`.
    fn eval_unfiltered_base_batch_packed(&self, vars_batch: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        let mut res = vec![F::ZERO; vars_batch.len() * self.num_constraints()];
        let (vars_packed_iter, vars_leftovers_iter) = vars_batch.pack::<<F as Packable>::Packing>();
        let leftovers_start = vars_batch.len() - vars_leftovers_iter.len();//32
        //println!("vars_packed_iter.len(): {:?}", vars_packed_iter.len());//32
        for (i, vars_packed) in vars_packed_iter.enumerate() {
            //println!("i:  {:?}vars_packed: {:?}", i,vars_packed);
            //println!("res: {:?}", res);//32
            //把vars_packed中的64个const-wire差值写入新建的StridedConstraintConsumer中，也就是res中的第i个跟i+32个。
            self.eval_unfiltered_base_packed(
                vars_packed,
                //创建一个StridedConstraintConsumer，长度为res的长度，步幅为vars_batch.len()，每次offset+1，因此可以把res填满。
                StridedConstraintConsumer::new(
                    &mut res[..],
                    vars_batch.len(),
                    <F as Packable>::Packing::WIDTH * i,
                ),
            );
        }
        //println!("vars_leftovers_iter.len(): {:?}", vars_leftovers_iter.len());//0
        for (i, vars_leftovers) in vars_leftovers_iter.enumerate() {
            self.eval_unfiltered_base_packed(
                vars_leftovers,
                StridedConstraintConsumer::new(&mut res[..], vars_batch.len(), leftovers_start + i),
            );
        }

        res
    }
}
