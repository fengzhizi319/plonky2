#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate which can perform a weighted multiply-add, i.e. `result = c0.x.y + c1.z`. If the config
/// has enough routed wires, it can support several such operations in one gate.
/// ArithmeticGate 是一个在电路中执行加权乘加操作的门。它可以执行如下操作：result = c0.x.y + c1.z，
/// 其中 c0 和 c1 是常数，x、y 和 z 是输入。这个门可以在一个门中支持多个这样的操作，具体取决于配置
#[derive(Debug, Clone)]
pub struct ArithmeticGate {
    /// Number of arithmetic operations performed by an arithmetic gate.
    /// num_ops: 表示一个 ArithmeticGate 可以执行的算术操作的数量。
    pub num_ops: usize,
}

impl ArithmeticGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    ///  确定在给定配置下一个门中可以容纳的最大操作数
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 4;
        config.num_routed_wires / wires_per_op
    }
    ///返回第 i 个操作的乘数、加数和输出的线索引。每个操作占用4个字段，所以要乘以4

    pub(crate) const fn wire_ith_multiplicand_0(i: usize) -> usize {
        4 * i
    }
    pub(crate) const fn wire_ith_multiplicand_1(i: usize) -> usize {
        4 * i + 1
    }
    pub(crate) const fn wire_ith_addend(i: usize) -> usize {
        4 * i + 2
    }
    pub(crate) const fn wire_ith_output(i: usize) -> usize {
        4 * i + 3
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for ArithmeticGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    //eval_unfiltered 的方法，用于评估 ArithmeticGate 的约束条件
    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        //从 vars 中提取局部常量 const_0 和 const_1
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[Self::wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[Self::wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[Self::wire_ith_addend(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = multiplicand_0 * multiplicand_1 * const_0 + addend * const_1;

            constraints.push(output - computed_output);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[Self::wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[Self::wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[Self::wire_ith_addend(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = {
                let scaled_mul =
                    builder.mul_many_extension([const_0, multiplicand_0, multiplicand_1]);
                builder.mul_add_extension(const_1, addend, scaled_mul)
            };

            let diff = builder.sub_extension(output, computed_output);
            constraints.push(diff);
        }

        constraints
    }

    /// ArithmeticGate 的每个算术操作生成一个见证生成器，并将这些生成器收集到一个向量中返回。

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    ArithmeticBaseGenerator {
                        row,
                        const_0: local_constants[0],
                        const_1: local_constants[1],
                        i,
                    }
                        .adapter(),
                )
            })
            .collect()
    }


    /*
    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let mut generators = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {

            let arithmetic_Generator = ArithmeticBaseGenerator {
                row,
                const_0: local_constants[0],
                const_1: local_constants[1],
                i,
            };
            let generator = arithmetic_Generator.adapter();
            //println!("generator: {:?}", generator);
            let generator1 = WitnessGeneratorRef::new(generator);
            //println!("generator: {:?}", generator1);

            //generators.push(WitnessGeneratorRef::new(generator));
            generators.push(generator1);
        }
        //println!("generator: {:?}", generators);
        generators
    }
    */

    fn num_wires(&self) -> usize {
        self.num_ops * 4
    }

    fn num_constants(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        3
    }

    fn num_constraints(&self) -> usize {
        self.num_ops
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for ArithmeticGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        for i in 0..self.num_ops {
            let multiplicand_0 = vars.local_wires[Self::wire_ith_multiplicand_0(i)];
            let multiplicand_1 = vars.local_wires[Self::wire_ith_multiplicand_1(i)];
            let addend = vars.local_wires[Self::wire_ith_addend(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = multiplicand_0 * multiplicand_1 * const_0 + addend * const_1;

            yield_constr.one(output - computed_output);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ArithmeticBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    const_0: F,
    const_1: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
for ArithmeticBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "ArithmeticBaseGenerator".to_string()
    }

    /// 获取当前生成器的依赖项
    ///
    /// # 返回值
    ///
    /// 返回一个包含依赖项目标的向量
    fn dependencies(&self) -> Vec<Target> {
        // 创建一个包含乘数、加数的线索引数组
        [
            ArithmeticGate::wire_ith_multiplicand_0(self.i), // 第 i 个操作的第一个乘数的线索引，4*i
            ArithmeticGate::wire_ith_multiplicand_1(self.i), // 第 i 个操作的第二个乘数的线索引，4*i+1
            ArithmeticGate::wire_ith_addend(self.i),         // 第 i 个操作的addend的线索引，4*i+2
        ]
            .iter() // 创建一个迭代器
            .map(|&i| Target::wire(self.row, i)) // 将线索引映射为目标
            .collect() // 收集结果到向量中
    }

    /// `run_once` 方法计算算术操作并将结果设置到 `out_buffer` 中。
    ///
    /// # 参数
    ///
    /// * `&self` - 方法的所有者。
    /// * `witness` - 一个 `PartitionWitness<F>` 类型的引用，用于获取目标值。
    /// * `out_buffer` - 一个可变引用，类型为 `GeneratedValues<F>`，用于存储生成的值。
    ///
    /// # 返回值
    ///
    /// 返回一个 `Result<()>`，表示操作是否成功。
    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        // 定义一个闭包函数，用于获取指定线的值。
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };


        //获取第 i 个乘数 0 的值。
        let multiplicand_0 = get_wire(ArithmeticGate::wire_ith_multiplicand_0(self.i));
        //获取第 i 个乘数 1 的值。
        let multiplicand_1 = get_wire(ArithmeticGate::wire_ith_multiplicand_1(self.i));
        //获取第 i 个加数的值。
        let addend = get_wire(ArithmeticGate::wire_ith_addend(self.i));

        // 获取输出目标所在的线。
        let output_target = Target::wire(self.row, ArithmeticGate::wire_ith_output(self.i));
        //println!("output_target:{:?}", output_target);
        // 计算输出值：乘数 0 * 乘数 1 * 常数 0 + 加数 * 常数 1。
        let computed_output =
            multiplicand_0 * multiplicand_1 * self.const_0 + addend * self.const_1;
        //println!("computed_output:{:?}", computed_output);
        // 将计算出的值设置到 `out_buffer` 中对应的目标位置。
        out_buffer.set_target(output_target, computed_output)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_field(self.const_1)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let const_1 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            const_0,
            const_1,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::arithmetic_base::ArithmeticGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = ArithmeticGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }
}
