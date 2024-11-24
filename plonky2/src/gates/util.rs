use core::marker::PhantomData;

use crate::field::packed::PackedField;

/// 将门生成的约束写入缓冲区，具有给定的步幅。
/// 允许我们抽象底层的内存布局。特别是，我们可以制作一个约束矩阵，其中每一列是一个评估点，每一行是一个约束索引，矩阵以行连续形式存储。

/// Writes constraints yielded by a gate to a buffer, with a given stride.
/// Permits us to abstract the underlying memory layout. In particular, we can make a matrix of
/// constraints where every column is an evaluation point and every row is a constraint index, with
/// the matrix stored in row-contiguous form.

#[derive(Debug)]
pub struct StridedConstraintConsumer<'a, P: PackedField> {
    // 这是一种特别巧妙的方法，比使用切片更好。我们在每一步增加 start 的值，步幅为 stride，并在等于 end 时终止。
    start: *mut P::Scalar,
    end: *mut P::Scalar,
    stride: usize,
    _phantom: PhantomData<&'a mut [P::Scalar]>,
}

impl<'a, P: PackedField> StridedConstraintConsumer<'a, P> {
    /// 创建一个新的 StridedConstraintConsumer 实例。
    ///
    /// # 参数
    /// - `buffer`：一个可变的标量切片，表示约束的缓冲区。
    /// - `stride`：步幅，表示每次增加的步长。
    /// - `offset`：偏移量，表示起始位置的偏移。
    pub fn new(buffer: &'a mut [P::Scalar], stride: usize, offset: usize) -> Self {
        // 确保步幅大于等于 P::WIDTH
        assert!(stride >= P::WIDTH);
        // 确保偏移量小于步幅
        assert!(offset < stride);
        // 确保缓冲区的长度是步幅的整数倍
        assert_eq!(buffer.len() % stride, 0);
        //as_mut_ptr_range返回一个指向切片的可变指针范围
        let ptr_range = buffer.as_mut_ptr_range();
        // wrapping_add 是 Rust 中用于执行加法运算的方法，当发生溢出时不会引发 panic，而是会执行环绕（wrap around）操作。
        // 也就是说，如果结果超出了类型的最大值，它会从最小值重新开始计算。
        let start = ptr_range.start.wrapping_add(offset);
        let end = ptr_range.end.wrapping_add(offset);
        Self {
            start,
            end,
            stride,
            _phantom: PhantomData,
        }
    }

    /// 将一个约束赋值给self.start，同时更新self.start+stride。
    pub fn one(&mut self, constraint: P) {
        if self.start != self.end {
            // # 安全性
            // `new` 方法中的检查保证了这个指针指向有效的空间。
            unsafe {
                *self.start.cast() = constraint;
            }
            // 参见 `new` 方法中的注释。如果我们刚刚耗尽了缓冲区（因此我们将 `self.start` 设置为指向缓冲区末尾之后），则需要 `wrapping_add` 以避免未定义行为。
            self.start = self.start.wrapping_add(self.stride);
        } else {
            panic!("gate produced too many constraints");
        }
        //println!("self: {:?}", self._phantom.);
    }

    /// 将多个约束constraints赋值给StridedConstraintConsumer的start，self.start+stride。
    pub fn many<I: IntoIterator<Item = P> + std::fmt::Debug>(&mut self, constraints: I) {
        constraints
            .into_iter()
            .for_each(|constraint| self.one(constraint));
        //println!("constraints: {:?}", constraints);
        // for constraint in constraints {
        //     println!("constraint: {:?}", constraint);
        //     self.one(constraint);
        // }
    }

}

