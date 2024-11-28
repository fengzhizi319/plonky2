#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Debug;
use core::marker::PhantomData;

use anyhow::{anyhow, Result};

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, IoResult, Read, Write};


/// 根据inputs把所有wire填充为对应的实际值的`PartitionWitness`。
pub fn generate_partial_witness<
    'a,
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D>,
    common_data: &'a CommonCircuitData<F, D>,
) -> Result<PartitionWitness<'a, F>> {
    // 获取电路配置
    let config = &common_data.config;
    // 获取生成器集合
    let generators = &prover_data.generators;
    //println!("generators:{:?}", generators);
    // 获取按监视目标索引的生成器索引
    let generator_indices_by_watches = &prover_data.generator_indices_by_watches;
    //println!("generator_indices_by_watches:{:?}", generator_indices_by_watches);
    /*
    //解释
    540: [133, 137],
    540表示在forest中的第540个parent。
    133表示generator的编号。
    因此540: [133, 137]表示，133，137个generator中，有parent为540的wire。
    generator_indices_by_watches：
    3: [134, 135],
    7: [135, 136],
    11: [136], 15: [137],
    405: [137],
    406: [133, 134, 135, 136],
    540: [133, 137],
    541: [133, 134, 137]
     */

    // 创建一个新的见证
    let mut witness = PartitionWitness::new(
        config.num_wires,
        common_data.degree(),
        &prover_data.representative_map,
    );
    //println!("representative_map:{:?}", prover_data.representative_map);

    // 设置输入目标的值，这样所有相同约束的目标都会被设置
    //println!("target_values:{:?}", inputs.target_values);
    for (t, v) in inputs.target_values.into_iter() {
        witness.set_target(t, v)?;
    }
    //println!("witness_values:{:?}", witness.values);

    // 构建一个“待处理”生成器列表，初始时所有生成器都在队列中
    let mut pending_generator_indices: Vec<_> = (0..generators.len()).collect();

    // 跟踪已经返回 false 的“过期”生成器列表
    let mut generator_is_expired = vec![false; generators.len()];
    let mut remaining_generators = generators.len();

    // 创建一个空的生成值缓冲区
    let mut buffer = GeneratedValues::empty();

    // 持续运行生成器，直到无法取得进展
    while !pending_generator_indices.is_empty() {
        let mut next_pending_generator_indices = Vec::new();
        //println!("next_pending_generator_indices:{:?}", pending_generator_indices);

        for &generator_idx in &pending_generator_indices {
            if generator_is_expired[generator_idx] {
                continue;
            }
            let generator = &generators[generator_idx].0;
            let finished = generator.run(&witness, &mut buffer);

            if finished {
                generator_is_expired[generator_idx] = true;
                remaining_generators -= 1;
            }

            // 将生成的值合并到见证中，并获取新填充目标的代表列表
            let mut new_target_reps = Vec::with_capacity(buffer.target_values.len());
            for (t, v) in buffer.target_values.drain(..) {
                let reps = witness.set_target_returning_rep(t, v)?;
                new_target_reps.extend(reps);
            }

            // 将监视新填充目标的未完成生成器加入队列
            //println!("new_target_reps:{:?}", new_target_reps);
            for watch in new_target_reps {
                // 3: [134, 135],
                // 7: [135, 136],
                // 11: [136],
                // 15: [137],
                // 405: [137],
                // 406: [133, 134, 135, 136],
                // 540: [133, 137],
                // 541: [133, 134, 137]
                let opt_watchers = generator_indices_by_watches.get(&watch);
                if let Some(watchers) = opt_watchers {
                    for &watching_generator_idx in watchers {
                        if !generator_is_expired[watching_generator_idx] {
                            next_pending_generator_indices.push(watching_generator_idx);
                        }
                    }
                }
            }
        }

        pending_generator_indices = next_pending_generator_indices;
    }
    //println!("next_pending_generator_indices:{:?}", pending_generator_indices);

    // 检查是否所有生成器都已运行
    if remaining_generators != 0 {
        return Err(anyhow!("{} generators weren't run", remaining_generators));
    }

    Ok(witness)
}

/// A generator participates in the generation of the witness.
pub trait WitnessGenerator<F: RichField + Extendable<D>, const D: usize>:
'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    /// Targets to be "watched" by this generator. Whenever a target in the watch list is populated,
    /// the generator will be queued to run.
    fn watch_list(&self) -> Vec<Target>;

    /// Run this generator, returning a flag indicating whether the generator is finished. If the
    /// flag is true, the generator will never be run again, otherwise it will be queued for another
    /// run next time a target in its watch list is populated.
    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool;

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

/// A wrapper around an `Box<WitnessGenerator>` which implements `PartialEq`
/// and `Eq` based on generator IDs.
pub struct WitnessGeneratorRef<F: RichField + Extendable<D>, const D: usize>(
    pub Box<dyn WitnessGenerator<F, D>>,
);

impl<F: RichField + Extendable<D>, const D: usize> WitnessGeneratorRef<F, D> {
    pub fn new<G: WitnessGenerator<F, D>>(generator: G) -> WitnessGeneratorRef<F, D> {
        WitnessGeneratorRef(Box::new(generator))
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PartialEq for WitnessGeneratorRef<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Eq for WitnessGeneratorRef<F, D> {}

impl<F: RichField + Extendable<D>, const D: usize> Debug for WitnessGeneratorRef<F, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "id: {}, as_ref: {:?}", self.0.id(), self.0.as_ref())

    }
}

/// Values generated by a generator invocation.
#[derive(Debug)]
pub struct GeneratedValues<F: Field> {
    pub target_values: Vec<(Target, F)>,
}

impl<F: Field> From<Vec<(Target, F)>> for GeneratedValues<F> {
    fn from(target_values: Vec<(Target, F)>) -> Self {
        Self { target_values }
    }
}

impl<F: Field> WitnessWrite<F> for GeneratedValues<F> {
    fn set_target(&mut self, target: Target, value: F) -> Result<()> {
        self.target_values.push((target, value));
        Ok(())
    }
}

impl<F: Field> GeneratedValues<F> {
    pub fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    pub fn empty() -> Self {
        Vec::new().into()
    }

    pub fn singleton_wire(wire: Wire, value: F) -> Self {
        Self::singleton_target(Target::Wire(wire), value)
    }

    pub fn singleton_target(target: Target, value: F) -> Self {
        vec![(target, value)].into()
    }

    pub fn singleton_extension_target<const D: usize>(
        et: ExtensionTarget<D>,
        value: F::Extension,
    ) -> Result<Self>
    where
        F: RichField + Extendable<D>,
    {
        let mut witness = Self::with_capacity(D);
        witness.set_extension_target(et, value)?;

        Ok(witness)
    }
}

/// A generator which runs once after a list of dependencies is present in the witness.
pub trait SimpleGenerator<F: RichField + Extendable<D>, const D: usize>:
'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    fn dependencies(&self) -> Vec<Target>;

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()>;

    fn adapter(self) -> SimpleGeneratorAdapter<F, Self, D>
    where
        Self: Sized,
    {
        SimpleGeneratorAdapter {
            inner: self,
            _phantom: PhantomData,
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()>;

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self>
    where
        Self: Sized;
}

#[derive(Debug)]
pub struct SimpleGeneratorAdapter<
    F: RichField + Extendable<D>,
    SG: SimpleGenerator<F, D> + ?Sized,
    const D: usize,
> {
    _phantom: PhantomData<F>,//是一个标记类型，?Sized类型，用于告诉编译器 SimpleGeneratorAdapter 结构体中实际上包含了类型 F，即使它没有直接存储 F 类型的值。
    inner: SG,
}

impl<F: RichField + Extendable<D>, SG: SimpleGenerator<F, D>, const D: usize> WitnessGenerator<F, D>
for SimpleGeneratorAdapter<F, SG, D>
{
    fn id(&self) -> String {
        self.inner.id()
    }

    fn watch_list(&self) -> Vec<Target> {
        self.inner.dependencies()
    }

    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
        // 检查见证是否包含所有依赖项
        //println!("inner:{:?}", self.inner);
        let depen=self.inner.dependencies();
        //if witness.contains_all(&self.inner.dependencies()) {
        if witness.contains_all(&depen) {
            // 运行生成器一次，并检查是否成功
            self.inner.run_once(witness, out_buffer).is_ok()
        } else {
            // 如果见证不包含所有依赖项，返回 false
            false
        }
    }

    fn serialize(&self, dst: &mut Vec<u8>, common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        self.inner.serialize(dst, common_data)
    }

    fn deserialize(src: &mut Buffer, common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            inner: SG::deserialize(src, common_data)?,
            _phantom: PhantomData,
        })
    }
}

/// A generator which copies one wire to another.
#[derive(Debug, Default)]
pub struct CopyGenerator {
    pub(crate) src: Target,
    pub(crate) dst: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for CopyGenerator {
    fn id(&self) -> String {
        "CopyGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.src]
    }

    /// `run_once` 方法从 `witness` 中获取一个值，并将其设置到 `out_buffer` 中。
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
        // 从 `witness` 中获取与 `self.src` 相关联的目标值。
        let value = witness.get_target(self.src);
        // 将获取到的值设置到 `out_buffer` 中与 `self.dst` 相关联的目标位置。
        out_buffer.set_target(self.dst, value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.src)?;
        dst.write_target(self.dst)
    }

    fn deserialize(source: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let src = source.read_target()?;
        let dst = source.read_target()?;
        Ok(Self { src, dst })
    }
}

/// A generator for including a random value
#[derive(Debug, Default)]
pub struct RandomValueGenerator {
    pub(crate) target: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for RandomValueGenerator {
    fn id(&self) -> String {
        "RandomValueGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        Vec::new()
    }

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        // 生成一个随机值
        let random_value = F::rand();
        // 将生成的随机值设置到目标位置
        out_buffer.set_target(self.target, random_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.target)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let target = src.read_target()?;
        Ok(Self { target })
    }
}

/// A generator for testing if a value equals zero
#[derive(Debug, Default)]
pub struct NonzeroTestGenerator {
    pub(crate) to_test: Target,
    pub(crate) dummy: Target,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for NonzeroTestGenerator {
    fn id(&self) -> String {
        "NonzeroTestGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.to_test]
    }

    /// `run_once` 方法从 `witness` 中获取一个值，并根据该值计算一个新的值，将其设置到 `out_buffer` 中。
    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        // 从 `witness` 中获取与 `self.to_test` 相关联的目标值。
        let to_test_value = witness.get_target(self.to_test);

        // 根据获取到的值计算一个新的值。
        let dummy_value = if to_test_value == F::ZERO {
            // 如果值为零，则设置为一。
            F::ONE
        } else {
            // 否则，计算该值的逆。
            to_test_value.inverse()
        };

        // 将计算出的值设置到 `out_buffer` 中与 `self.dummy` 相关联的目标位置。
        out_buffer.set_target(self.dummy, dummy_value)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_target(self.to_test)?;
        dst.write_target(self.dummy)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let to_test = src.read_target()?;
        let dummy = src.read_target()?;
        Ok(Self { to_test, dummy })
    }
}

/// A generator used to fill an extra constant.
/// 用于填充额外常量的生成器。
#[derive(Debug, Clone, Default)]
pub struct ConstantGenerator<F: Field> {
    /// The row index in the circuit where this constant is used.
    /// 此常量在电路中使用的行索引。
    pub row: usize,

    /// The index of the constant within the gate.
    /// 门内常量的索引。
    pub constant_index: usize,

    /// The wire index in the circuit where this constant is connected.
    /// 此常量连接到电路中的线索引。
    pub wire_index: usize,

    /// The value of the constant.
    /// 常量的值。
    pub constant: F,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn set_constant(&mut self, c: F) {
        self.constant = c;
    }
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for ConstantGenerator<F> {
    fn id(&self) -> String {
        "ConstantGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }
    /// `run_once` 方法将一个常量值设置到 `out_buffer` 中。

    fn run_once(
        &self,
        _witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        // 将常量值设置到 `out_buffer` 中与指定行和线索索引相关联的目标位置。
        out_buffer.set_target(Target::wire(self.row, self.wire_index), self.constant)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.constant_index)?;
        dst.write_usize(self.wire_index)?;
        dst.write_field(self.constant)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let constant_index = src.read_usize()?;
        let wire_index = src.read_usize()?;
        let constant = src.read_field()?;
        Ok(Self {
            row,
            constant_index,
            wire_index,
            constant,
        })
    }
}
