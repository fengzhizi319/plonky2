#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hashbrown::HashMap;
use plonky2_maybe_rayon::*;

use crate::field::polynomial::PolynomialValues;
use crate::field::types::Field;
use crate::iop::target::Target;
use crate::iop::wire::Wire;

/// Disjoint Set Forest data-structure following <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>.
#[derive(Debug)]
pub struct Forest {
    /// A map of parent pointers, stored as indices.
    pub(crate) parents: Vec<usize>,

    num_wires: usize,
    num_routed_wires: usize,
    degree: usize,
}

impl Forest {
    pub fn new(
        num_wires: usize,
        num_routed_wires: usize,
        degree: usize,
        num_virtual_targets: usize,
    ) -> Self {
        let capacity = num_wires * degree + num_virtual_targets;
        Self {
            parents: Vec::with_capacity(capacity),
            num_wires,
            num_routed_wires,
            degree,
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    /// Add a new partition with a single member.
    pub fn add(&mut self, t: Target) {
        let index = self.parents.len();
        //println!(self.target_index(t))
        //println!("self.target_index(t):{:?}",self.target_index(t));
        debug_assert_eq!(self.target_index(t), index);
        self.parents.push(index);
    }

    /// Path compression method, see <https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Finding_set_representatives>.
    pub fn find(&mut self, mut x_index: usize) -> usize {
        // Note: We avoid recursion here since the chains can be long, causing stack overflows.

        // First, find the representative of the set containing `x_index`.
        let mut representative = x_index;

        while self.parents[representative] != representative {
            representative = self.parents[representative];
        }

        // Then, update each node in this chain to point directly to the representative.
        while self.parents[x_index] != x_index {
            let old_parent = self.parents[x_index];
            self.parents[x_index] = representative;
            // if old_parent != representative {
            //     println!("x_index:{:?}",x_index);
            //     println!("old_parent:{:?}",old_parent);
            //     println!("new:{:?}",representative);
            // }

            x_index = old_parent;
        }

        let parents_representative=self.parents[representative];
        representative
    }

    /// Merge two sets.
    pub fn merge(&mut self, tx: Target, ty: Target) {
        // println!("tx:{:?},ty:{:?}",tx,ty);
        // println!("self.target_index(tx):{:?}",self.target_index(tx));
        // println!("self.target_index(ty):{:?}",self.target_index(ty));
        let x_index = self.find(self.target_index(tx));
        let y_index = self.find(self.target_index(ty));

        if x_index == y_index {
            return;
        }

        self.parents[y_index] = x_index;
    }

    /// Compress all paths. After calling this, every `parent` value will point to the node's
    /// representative.
    pub(crate) fn compress_paths(&mut self) {
        for i in 0..self.parents.len() {
            self.find(i);
        }
    }

    /// Assumes `compress_paths` has already been called.
    /// 相同拷贝约束的wire都指向同一个parent，即被放入同一个vec中
    pub fn wire_partition(&mut self) -> WirePartition {
        let mut partition = HashMap::<_, Vec<_>>::new();

        // Here we keep just the Wire targets, filtering out everything else.
        for row in 0..self.degree {
            for column in 0..self.num_routed_wires {
                let w = Wire { row, column };
                let t = Target::Wire(w);
                let x_parent = self.parents[self.target_index(t)];
                partition.entry(x_parent).or_default().push(w);
                //println!("partition[{:?}]{:?}",x_parent,partition[&x_parent]);

            }
        }

        let partition = partition.into_values().collect();
        WirePartition { partition }
    }
}

#[derive(Debug)]
pub struct WirePartition {
    partition: Vec<Vec<Wire>>,
}

impl WirePartition {
    pub(crate) fn get_sigma_polys<F: Field>(
        &self,
        degree_log: usize,
        k_is: &[F],
        subgroup: &[F],
    ) -> Vec<PolynomialValues<F>> {
        let degree = 1 << degree_log;
        let sigma = self.get_sigma_map(degree, k_is.len());

        sigma
            .chunks(degree)
            .map(|chunk| {
                let values = chunk
                    .par_iter()
                    .map(|&x| k_is[x / degree] * subgroup[x % degree])
                    .collect::<Vec<_>>();
                PolynomialValues::new(values)
            })
            .collect()
    }

    /// Generates sigma in the context of Plonk, which is a map from `[kn]` to `[kn]`, where `k` is
    /// the number of routed wires and `n` is the number of gates.

    fn get_sigma_map(&self, degree: usize, num_routed_wires: usize) -> Vec<usize> {
        // Find a wire's "neighbor" in the context of Plonk's "extended copy constraints" check. In
        // other words, find the next wire in the given wire's partition. If the given wire is last in
        // its partition, this will loop around. If the given wire has a partition all to itself, it
        // is considered its own neighbor.
        // 创建一个哈希映射，用于存储每个 wire 的“邻居”。
        // 在 Plonk 的“扩展复制约束”检查中，找到给定 wire 的下一个 wire。
        // 换句话说，找到给定 wire 在其分区中的下一个 wire。如果给定的 wire 是其分区中的最后一个 wire，则循环回到第一个 wire。
        // 如果给定的 wire 有一个独立的分区，它被认为是它自己的邻居。
        let mut neighbors = HashMap::with_capacity(self.partition.len());

        // 遍历每个分区子集
        for subset in &self.partition {
            // 遍历子集中的每个 wire
            //println!("subset:{:?}",subset);
            for n in 0..subset.len() {
                // 将当前 wire 和下一个 wire（或循环回到第一个 wire）插入哈希映射
                neighbors.insert(subset[n], subset[(n + 1) % subset.len()]);
            }
        }
        //print the neighbors
        println!("neighbors:{:?}",neighbors);

        // 创建一个向量，用于存储 sigma 映射
        let mut sigma = Vec::with_capacity(num_routed_wires * degree);

        // 遍历所有列
        for column in 0..num_routed_wires {
            // 遍历所有行
            for row in 0..degree {
                // 创建一个 wire 实例
                let wire = Wire { row, column };
                // 获取 wire 的邻居
                let neighbor = neighbors[&wire];
                // 将邻居的列和行转换为索引，并添加到 sigma 向量中
                sigma.push(neighbor.column * degree + neighbor.row);
            }
        }

        // 返回 sigma 映射
        sigma
    }
}
