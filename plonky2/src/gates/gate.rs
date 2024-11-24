#[cfg(not(feature = "std"))]
use alloc::{string::String, sync::Arc, vec, vec::Vec};
use core::any::Any;
use core::fmt::{Debug, Error, Formatter};
use core::hash::{Hash, Hasher};
use core::ops::Range;
#[cfg(feature = "std")]
use std::sync::Arc;

use hashbrown::HashMap;
use serde::{Serialize, Serializer};

use crate::field::batch_util::batch_multiply_inplace;
use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::gates::selectors::UNUSED_SELECTOR;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::WitnessGeneratorRef;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
};
use crate::util::serialization::{Buffer, IoResult};

/// A custom gate.
///
/// Vanilla Plonk arithmetization only supports basic fan-in 2 / fan-out 1 arithmetic gates,
/// each of the form
///
/// $$ a.b \cdot q_M + a \cdot q_L + b \cdot q_R + c \cdot q_O + q_C = 0 $$
///
/// where:
/// - $q_M$, $q_L$, $q_R$ and $q_O$ are boolean selectors,
/// - $a$, $b$ and $c$ are values used as inputs and output respectively,
/// - $q_C$ is a constant (possibly 0).
///
/// This allows expressing simple operations like multiplication, addition, etc. For
/// instance, to define a multiplication, one can set $q_M=1$, $q_L=q_R=0$, $q_O = -1$ and $q_C = 0$.
///
/// Hence, the gate equation simplifies to $a.b - c = 0$, or equivalently to $a.b = c$.
///
/// However, such a gate is fairly limited for more complex computations. Hence, when a computation may
/// require too many of these "vanilla" gates, or when a computation arises often within the same circuit,
/// one may want to construct a tailored custom gate. These custom gates can use more selectors and are
/// not necessarily limited to 2 inputs + 1 output = 3 wires.
/// For instance, plonky2 supports natively a custom Poseidon hash gate that uses 135 wires.
///
/// Note however that extending the number of wires necessary for a custom gate comes at a price, and may
/// impact the overall performances when generating proofs for a circuit containing them.
pub trait Gate<F: RichField + Extendable<D>, const D: usize>: 'static + Send + Sync {
    /// Defines a unique identifier for this custom gate.
    ///
    /// This is used as differentiating tag in gate serializers.
    fn id(&self) -> String;

    /// Serializes this custom gate to the targeted byte buffer, with the provided [`CommonCircuitData`].
    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    /// Deserializes the bytes in the provided buffer into this custom gate, given some [`CommonCircuitData`].
    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;

    /// Defines and evaluates the constraints that enforce the statement represented by this gate.
    /// Constraints must be defined in the extension of this custom gate base field.
    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension>;

    /// Like `eval_unfiltered`, but specialized for points in the base field.
    ///
    ///
    /// `eval_unfiltered_base_batch` calls this method by default. If `eval_unfiltered_base_batch`
    /// is overridden, then `eval_unfiltered_base_one` is not necessary.
    ///
    /// By default, this just calls `eval_unfiltered`, which treats the point as an extension field
    /// element. This isn't very efficient.
    fn eval_unfiltered_base_one(
        &self,
        vars_base: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Note that this method uses `yield_constr` instead of returning its constraints.
        // `yield_constr` abstracts out the underlying memory layout.
        let local_constants = &vars_base
            .local_constants
            .iter()
            .map(|c| F::Extension::from_basefield(*c))
            .collect::<Vec<_>>();
        let local_wires = &vars_base
            .local_wires
            .iter()
            .map(|w| F::Extension::from_basefield(*w))
            .collect::<Vec<_>>();
        let public_inputs_hash = &vars_base.public_inputs_hash;
        let vars = EvaluationVars {
            local_constants,
            local_wires,
            public_inputs_hash,
        };
        let values = self.eval_unfiltered(vars);

        // Each value should be in the base field, i.e. only the degree-zero part should be nonzero.
        values.into_iter().for_each(|value| {
            debug_assert!(F::Extension::is_in_basefield(&value));
            yield_constr.one(value.to_basefield_array()[0])
        })
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        //let mut res = vec![F::ZERO; vars_base.len() * self.num_constraints()];
        // for (i, vars_base_one) in vars_base.iter().enumerate() {
        //     self.eval_unfiltered_base_one(
        //         vars_base_one,
        //         StridedConstraintConsumer::new(&mut res, vars_base.len(), i),
        //     );
        // }

        let vars_base_len=vars_base.len();//32
        let num_constraints_len=self.num_constraints();//123
        let mut res = vec![F::ZERO; vars_base_len * num_constraints_len];
        for (i, vars_base_one) in vars_base.iter().enumerate() {
            let consumer=StridedConstraintConsumer::new(&mut res, vars_base_len, i);
            self.eval_unfiltered_base_one(
                vars_base_one,
                consumer,
            );
        }
        res
    }

    /// Defines the recursive constraints that enforce the statement represented by this custom gate.
    /// This is necessary to recursively verify proofs generated from a circuit containing such gates.
    ///
    /// **Note**: The order of the recursive constraints output by this method should match exactly the order
    /// of the constraints obtained by the non-recursive [`Gate::eval_unfiltered`] method, otherwise the
    /// prover won't be able to generate proofs.
    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>>;

    fn eval_filtered(
        &self,
        mut vars: EvaluationVars<F, D>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
    ) -> Vec<F::Extension> {
        let filter = compute_filter(
            row,
            group_range,
            vars.local_constants[selector_index],
            num_selectors > 1,
        );
        vars.remove_prefix(num_selectors);
        vars.remove_prefix(num_lookup_selectors);
        self.eval_unfiltered(vars)
            .into_iter()
            .map(|c| filter * c)
            .collect()
    }
    ///（const-wire）*(1-s)(2-s)(3-s)
    /// The result is an array of length `vars_batch.len() * self.num_constraints()`. Constraint `j`
    /// for point `i` is at index `j * batch_size + i`.
    fn eval_filtered_base_batch(
        &self,
        mut vars_batch: EvaluationVarsBaseBatch<F>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
    ) -> Vec<F> {
        //vars:EvaluationVarsBase {
        // local_constants: PackedStridedView { start_ptr: 0x15d80d200, length: 4, stride: 32, _phantom: PhantomData<&[plonky2_field::goldilocks_field::GoldilocksField]> },
        // local_wires: PackedStridedView { start_ptr: 0x160008000, length: 135, stride: 32, _phantom: PhantomData<&[plonky2_field::goldilocks_field::GoldilocksField]> },
        // public_inputs_hash: HashOut { elements: [12460551030817792791, 6203763534542844149, 15133388778355119947, 8532039303907884673] } }

        //println!("row:{},selector_index:{},group_range:{:?},num_selectors:{},num_lookup_selectors:{}",row,selector_index,group_range,num_selectors,num_lookup_selectors);
        // let filters: Vec<_> = vars_batch
        //     .iter()
        //     .map(|vars| {
        //         compute_filter(
        //             row,
        //             group_range.clone(),
        //             vars.local_constants[selector_index],
        //             num_selectors > 1,
        //         )
        //     })
        //     .collect();
        //begin debug
        //vars_batch.len():32
        let mut filters = Vec::with_capacity(vars_batch.len());
        //println!("vars_batch.len():{:?}",vars_batch.len());//32
        //println!("vars_batch.local_constants: {:?}",vars_batch.local_constants);//32
        for vars in vars_batch.iter() {
            // println!("vars.local_constants:{:?},{:?},{:?},{:?}",vars.local_constants[0],vars.local_constants[1],vars.local_constants[2],vars.local_constants[3]);
            //println!("row:{:?},selector_index={:?},vars.local_constants[selector_index]:{:?}",row,selector_index,vars.local_constants[selector_index]);//550480699699462186
            // vars.local_constants:550480699699462186,13964441375994449852,5633733284883438778,13404483256729242856
            // vars.local_constants[selector_index]:550480699699462186
            //if row=0，then (1-s)(2-s)(3-s)，if row=1，then (0-s)(2-s)(3-s)，if row=2，then (0-s)(1-s)(3-s)，if row=3，then (0-s)(1-s)(2-s)
            //if row=1，then (0-s)(2-s)(3-s)
            //println!("row:{},vars.local_constants[selector_index]:{:?}",row,vars.local_constants[selector_index]);
            let filter = compute_filter(
                //row:0,selector_index:0,group_range:0..3, :2,num_lookup_selectors:0
                row,
                group_range.clone(),
                vars.local_constants[selector_index],
                num_selectors > 1,
            );
            //println!("filter:{:?}",filter);//filter:10708010571758048252
            filters.push(filter);
        }
        //end
        //移除 vars_batch中local_constants中前 num_selectors + num_lookup_selectors）batch_size 个元素。如移除2*32=64个元素
        //println!("num_selectors:{:?}",num_selectors);
        //println!("begin vars_batch.local_constants: {:?}",vars_batch.local_constants);//32
        //取出vars_batch中的local_constants的第（num_selectors + num_lookup_selectors）batch_size 个元素开始的切片
        vars_batch.remove_prefix(num_selectors + num_lookup_selectors);
        //println!("end vars_batch.local_constants: {:?}",vars_batch.local_constants);//32
        //计算vars_batch中的const-wire的差值，并返回给res_batch
        let mut res_batch = self.eval_unfiltered_base_batch(vars_batch);
        //println!("res_batch:{:?}",res_batch);
        //将res_batch按filters.len()的大小分割成多个可变的切片（chunk）。chunks_exact_mut方法确保每个切
        // 片的大小都正好是filters.len()，如果res_batch的长度不是filters.len()的整数倍，则会忽略最后不足的部分
        //filters.len()=32
        for res_chunk in res_batch.chunks_exact_mut(filters.len()) {
            //res_chunk与filters逐个元素相乘，结果存储在res_chunk中res_chunk[i]=res_chunk[i]*filters[i]
            batch_multiply_inplace(res_chunk, &filters);
        }
        res_batch
    }

    /// Adds this gate's filtered constraints into the `combined_gate_constraints` buffer.
    fn eval_filtered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        mut vars: EvaluationTargets<D>,
        row: usize,
        selector_index: usize,
        group_range: Range<usize>,
        num_selectors: usize,
        num_lookup_selectors: usize,
        combined_gate_constraints: &mut [ExtensionTarget<D>],
    ) {
        let filter = compute_filter_circuit(
            builder,
            row,
            group_range,
            vars.local_constants[selector_index],
            num_selectors > 1,
        );
        vars.remove_prefix(num_selectors);
        vars.remove_prefix(num_lookup_selectors);
        let my_constraints = self.eval_unfiltered_circuit(builder, vars);
        for (acc, c) in combined_gate_constraints.iter_mut().zip(my_constraints) {
            *acc = builder.mul_add_extension(filter, c, *acc);
        }
    }

    /// The generators used to populate the witness.
    ///
    /// **Note**: This should return exactly 1 generator per operation in the gate.
    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>>;

    /// The number of wires used by this gate.
    ///
    /// While vanilla Plonk can only evaluate one addition/multiplication at a time, a wider
    /// configuration may be able to accommodate several identical gates at once. This is
    /// particularly helpful for tiny custom gates that are being used extensively in circuits.
    ///
    /// For instance, the [crate::gates::multiplication_extension::MulExtensionGate] takes `3*D`
    /// wires per multiplication (where `D`` is the degree of the extension), hence for a usual
    /// configuration of 80 routed wires with D=2, one can evaluate 13 multiplications within a
    /// single gate.
    fn num_wires(&self) -> usize;

    /// The number of constants used by this gate.
    fn num_constants(&self) -> usize;

    /// The maximum degree among this gate's constraint polynomials.
    fn degree(&self) -> usize;

    /// The number of constraints defined by this sole custom gate.
    fn num_constraints(&self) -> usize;

    /// Number of operations performed by the gate.
    fn num_ops(&self) -> usize {
        self.generators(0, &vec![F::ZERO; self.num_constants()])
            .len()
    }

    /// Enables gates to store some "routed constants", if they have both unused constants and
    /// unused routed wires.
    ///
    /// Each entry in the returned `Vec` has the form `(constant_index, wire_index)`. `wire_index`
    /// must correspond to a *routed* wire.
    fn extra_constant_wires(&self) -> Vec<(usize, usize)> {
        vec![]
    }
}

/// A wrapper trait over a `Gate`, to allow for gate serialization.
//定义了一个名为 AnyGate 的公共特性。它有一个泛型参数 F，要求 F 实现 RichField 和 Extendable<D> 特性，并且有一个常量参数 D
//Gate<F, D>：表示 AnyGate 特性继承了 Gate<F, D> 特性。也就是说，任何实现 AnyGate 的类型也必须实现 Gate 特性
pub trait AnyGate<F: RichField + Extendable<D>, const D: usize>: Gate<F, D> {
    //定义了一个名为 as_any 的方法。这个方法返回一个对实现了 Any 特性的动态对象的引用。Any 特性允许在运行时进行类型检查和转换。
    //Any 特性是 Rust 标准库中的一个特性，用于在运行时进行类型检查和转换。它允许将类型信息存储在运行时，并在需要时进行类型转换。Any 特性通常与动态分发和类型擦除一起使用
    fn as_any(&self) -> &dyn Any;
}

//这行代码表示为任何实现了 Gate<F, D> 特性的类型 T 实现 AnyGate<F, D> 特性。F 是一个泛型参数，要求实现 RichField 和 Extendable<D> 特性，并且有一个常量参数 D。
impl<T: Gate<F, D>, F: RichField + Extendable<D>, const D: usize> AnyGate<F, D> for T {
    //这是 AnyGate 特性中的方法实现。这个方法返回一个对实现了 Any 特性的动态对象的引用。
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A wrapper around an `Arc<AnyGate>` which implements `PartialEq`, `Eq` and `Hash` based on gate IDs.
#[derive(Clone)]
//定义了一个名为 GateRef 的元组结构体。它包含一个字段，这个字段的类型是 Arc<dyn AnyGate<F, D>>。这个字段是公共的（pub），可以在模块外部访问。
/*在 Rust 中，元组结构体是一种特殊的结构体，它的字段没有名称，通过位置来访问。以下是定义元组结构体的语法：
//定义一个包含两个 i32 字段的元组结构体
    pub struct MyTupleStruct(i32, i32);
    //你可以通过位置来访问元组结构体的字段：
    let instance = MyTupleStruct(10, 20);
    let first_field = instance.0; // 访问第一个字段
    let second_field = instance.1; // 访问第二个字段
 */
pub struct GateRef<F: RichField + Extendable<D>, const D: usize>(pub Arc<dyn AnyGate<F, D>>);


impl<F: RichField + Extendable<D>, const D: usize> GateRef<F, D> {
    pub fn new<G: Gate<F, D>>(gate: G) -> GateRef<F, D> {
        GateRef(Arc::new(gate))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PartialEq for GateRef<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Hash for GateRef<F, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state)
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Eq for GateRef<F, D> {}

impl<F: RichField + Extendable<D>, const D: usize> Debug for GateRef<F, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{}", self.0.id())
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Serialize for GateRef<F, D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.id())
    }
}

/// Map between gate parameters and available slots.
/// An available slot is of the form `(row, op)`, meaning the current available slot
/// is at gate index `row` in the `op`-th operation.
#[derive(Clone, Debug, Default)]
pub struct CurrentSlot<F: RichField + Extendable<D>, const D: usize> {
    pub current_slot: HashMap<Vec<F>, (usize, usize)>,
}

/// A gate along with any constants used to configure it.
#[derive(Clone, Debug)]
pub struct GateInstance<F: RichField + Extendable<D>, const D: usize> {
    pub gate_ref: GateRef<F, D>,
    pub constants: Vec<F>,
}

/// Map each gate to a boolean prefix used to construct the gate's selector polynomial.
#[derive(Debug, Clone)]
pub struct PrefixedGate<F: RichField + Extendable<D>, const D: usize> {
    pub gate: GateRef<F, D>,
    pub prefix: Vec<bool>,
}

/// (0-s)*(1-s)*(2-s)*...*(x-s)*...*(t-1-s)其中x不等于row；因此x等于row时不等于0，其他group_range值为0
fn compute_filter<K: Field>(row: usize, group_range: Range<usize>, s: K, many_selector: bool) -> K {
    // 断言 group_range 包含 row
    debug_assert!(group_range.contains(&row));
    //println!("row:{},group_range:{:?}",row,group_range);
    //row:0,group_range:0..3，many_selector=true，s=550480699699462186
    //chain 是 Rust 中迭代器的一种方法，用于将两个迭代器连接在一起，形成一个新的迭代器。
    // 这个新迭代器会先遍历第一个迭代器的所有元素，然后继续遍历第二个迭代器的所有元素。

    // 过滤掉 group_range 中等于 row 的元素，并根据 many_selector 决定是否添加 UNUSED_SELECTOR
    group_range
        .filter(|&i| i != row)
        .chain(many_selector.then_some(UNUSED_SELECTOR))
        // 将每个元素转换为 K 类型并减去 s
        .map(|i| K::from_canonical_usize(i) - s)
        // 计算所有元素的乘积
        .product()

}

fn compute_filter_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    row: usize,
    group_range: Range<usize>,
    s: ExtensionTarget<D>,
    many_selectors: bool,
) -> ExtensionTarget<D> {
    debug_assert!(group_range.contains(&row));
    let v = group_range
        .filter(|&i| i != row)
        .chain(many_selectors.then_some(UNUSED_SELECTOR))
        .map(|i| {
            let c = builder.constant_extension(F::Extension::from_canonical_usize(i));
            builder.sub_extension(c, s)
        })
        .collect::<Vec<_>>();
    builder.mul_many_extension(v)
}
