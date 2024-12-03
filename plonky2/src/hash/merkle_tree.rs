#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice;

use plonky2_maybe_rayon::*;
use serde::{Deserialize, Serialize};

use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::plonk::config::{GenericHashOut, Hasher};
use crate::util::log2_strict;

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(bound = "")]
// TODO: Change H to GenericHashOut<F>, since this only cares about the hash, not the hasher.
pub struct MerkleCap<F: RichField, H: Hasher<F>>(pub Vec<H::Hash>);

impl<F: RichField, H: Hasher<F>> Default for MerkleCap<F, H> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<F: RichField, H: Hasher<F>> MerkleCap<F, H> {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        log2_strict(self.len())
    }

    pub fn flatten(&self) -> Vec<F> {
        self.0.iter().flat_map(|&h| h.to_vec()).collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// MerkleTree 结构体，用于表示 Merkle 树。
pub struct MerkleTree<F: RichField, H: Hasher<F>> {
    /// Merkle 树叶子节点的数据。
    pub leaves: Vec<Vec<F>>,

    /// 树中的哈希值。由 `cap.len()` 个子树组成，每个子树对应 `cap` 中的一个元素。
    /// 每个子树是连续存储的，位于 `digests[digests.len() / cap.len() * i..digests.len() / cap.len() * (i + 1)]`。
    /// 在每个子树中，兄弟节点是相邻存储的。布局为：
    /// 左子树 || 左子节点哈希值 || 右子节点哈希值 || 右子树，
    /// 其中左子节点哈希值和右子节点哈希值是 H::Hash 类型，左子树和右子树是递归的。
    /// 注意，节点的哈希值是由其父节点存储的。因此，根节点的哈希值不存储在这里（可以在 `cap` 中找到）。
    pub digests: Vec<H::Hash>,

    /// Merkle 树的顶层节点（Merkle cap）。
    pub cap: MerkleCap<F, H>,
}

impl<F: RichField, H: Hasher<F>> Default for MerkleTree<F, H> {
    fn default() -> Self {
        Self {
            leaves: Vec::new(),
            digests: Vec::new(),
            cap: MerkleCap::default(),
        }
    }
}

pub(crate) fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

pub(crate) fn fill_subtree<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
) -> H::Hash {
    assert_eq!(leaves.len(), digests_buf.len() / 2 + 1);
    if digests_buf.is_empty() {
        H::hash_or_noop(&leaves[0])
    } else {
        // Layout is: left recursive output || left child digest
        //             || right child digest || right recursive output.
        // Split `digests_buf` into the two recursive outputs (slices) and two child digests
        // (references).
        let (left_digests_buf, right_digests_buf) = digests_buf.split_at_mut(digests_buf.len() / 2);
        let (left_digest_mem, left_digests_buf) = left_digests_buf.split_last_mut().unwrap();
        let (right_digest_mem, right_digests_buf) = right_digests_buf.split_first_mut().unwrap();
        // Split `leaves` between both children.
        let (left_leaves, right_leaves) = leaves.split_at(leaves.len() / 2);

        let (left_digest, right_digest) = plonky2_maybe_rayon::join(
            || fill_subtree::<F, H>(left_digests_buf, left_leaves),
            || fill_subtree::<F, H>(right_digests_buf, right_leaves),
        );

        left_digest_mem.write(left_digest);
        right_digest_mem.write(right_digest);
        H::two_to_one(left_digest, right_digest)
    }
}

pub(crate) fn fill_digests_buf<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &[Vec<F>],
    cap_height: usize,
) {
    // Special case of a tree that's all cap. The usual case will panic because we'll try to split
    // an empty slice into chunks of `0`. (We would not need this if there was a way to split into
    // `blah` chunks as opposed to chunks _of_ `blah`.)
    if digests_buf.is_empty() {
        debug_assert_eq!(cap_buf.len(), leaves.len());
        cap_buf
            .par_iter_mut()
            .zip(leaves)
            .for_each(|(cap_buf, leaf)| {
                cap_buf.write(H::hash_or_noop(leaf));
            });
        return;
    }

    let subtree_digests_len = digests_buf.len() >> cap_height;
    let subtree_leaves_len = leaves.len() >> cap_height;
    let digests_chunks = digests_buf.par_chunks_exact_mut(subtree_digests_len);
    let leaves_chunks = leaves.par_chunks_exact(subtree_leaves_len);
    assert_eq!(digests_chunks.len(), cap_buf.len());
    assert_eq!(digests_chunks.len(), leaves_chunks.len());
    digests_chunks.zip(cap_buf).zip(leaves_chunks).for_each(
        |((subtree_digests, subtree_cap), subtree_leaves)| {
            // We have `1 << cap_height` sub-trees, one for each entry in `cap`. They are totally
            // independent, so we schedule one task for each. `digests_buf` and `leaves` are split
            // into `1 << cap_height` slices, one for each sub-tree.
            subtree_cap.write(fill_subtree::<F, H>(subtree_digests, subtree_leaves));
        },
    );
}

pub(crate) fn merkle_tree_prove1<F: RichField, H: Hasher<F>>(
    leaf_index: usize,
    leaves_len: usize,
    cap_height: usize,
    digests: &[H::Hash],
) -> Vec<H::Hash> {
    let num_layers = log2_strict(leaves_len) - cap_height;
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    let digest_len = 2 * (leaves_len - (1 << cap_height));
    assert_eq!(digest_len, digests.len());

    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers;
        let tree_len = digest_len >> cap_height;
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)]
    };

    // Mask out high bits to get the index within the sub-tree.
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1;
            pair_index >>= 1;

            // The layers' data is interleaved as follows:
            // [layer 0, layer 1, layer 0, layer 2, layer 0, layer 1, layer 0, layer 3, ...].
            // Each of the above is a pair of siblings.
            // `pair_index` is the index of the pair within layer `i`.
            // The index of that the pair within `digests` is
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`.
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // We have an index for the _pair_, but we want the index of the _sibling_.
            // Double the pair index to get the index of the left sibling. Conditionally add `1`
            // if we are to retrieve the right sibling.
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index]
        })
        .collect()
}
pub(crate) fn merkle_tree_prove<F: RichField, H: Hasher<F>>(
    leaf_index: usize, // 叶子节点的索引
    leaves_len: usize, // 叶子节点的数量
    cap_height: usize, // Merkle 树顶层节点的高度
    digests: &[H::Hash], // Merkle 树中的哈希值
) -> Vec<H::Hash> {
    // 计算从叶子节点到顶层节点的层数
    let num_layers = log2_strict(leaves_len) - cap_height;
    // 确保叶子节点索引在有效范围内
    debug_assert_eq!(leaf_index >> (cap_height + num_layers), 0);

    // 计算哈希值的总长度
    let digest_len = 2 * (leaves_len - (1 << cap_height));
    // 确保哈希值的长度与计算的长度一致
    assert_eq!(digest_len, digests.len());

    // 获取子树的哈希值
    let digest_tree: &[H::Hash] = {
        let tree_index = leaf_index >> num_layers; // 计算子树的索引
        let tree_len = digest_len >> cap_height; // 计算子树的长度
        &digests[tree_len * tree_index..tree_len * (tree_index + 1)] // 获取子树的哈希值
    };

    // 屏蔽高位以获取子树内的索引
    let mut pair_index = leaf_index & ((1 << num_layers) - 1);
    (0..num_layers)
        .map(|i| {
            let parity = pair_index & 1; // 计算奇偶性
            pair_index >>= 1; // 更新索引

            // 各层的数据交错排列如下：
            // [第0层, 第1层, 第0层, 第2层, 第0层, 第1层, 第0层, 第3层, ...]。
            // 上述每一项都是一对兄弟节点。
            // `pair_index` 是层 `i` 中对的索引。
            // 该对在 `digests` 中的索引为
            // `pair_index * 2 ** (i + 1) + (2 ** i - 1)`。
            let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
            // 我们有一个对的索引，但我们需要兄弟节点的索引。
            // 将对的索引加倍以获得左兄弟节点的索引。如果要检索右兄弟节点，则有条件地加 `1`。
            let sibling_index = 2 * siblings_index + (1 - parity);
            digest_tree[sibling_index] // 返回兄弟节点的哈希值
        })
        .collect() // 收集结果为向量
}
impl<F: RichField, H: Hasher<F>> MerkleTree<F, H> {
    pub fn new(leaves: Vec<Vec<F>>, cap_height: usize) -> Self {
        // 计算叶子节点数量的对数（以 2 为底）
        let log2_leaves_len = log2_strict(leaves.len());
        // 确保 cap_height 不大于叶子节点数量的对数
        assert!(
            cap_height <= log2_leaves_len,
            "cap_height={} should be at most log2(leaves.len())={}",
            cap_height,
            log2_leaves_len
        );

        // 计算哈希值的数量
        let num_digests = 2 * (leaves.len() - (1 << cap_height));
        // 为哈希值分配空间
        let mut digests = Vec::with_capacity(num_digests);

        // 计算 Merkle cap 的长度
        let len_cap = 1 << cap_height;
        // 为 Merkle cap 分配空间
        let mut cap = Vec::with_capacity(len_cap);

        // 获取哈希值缓冲区
        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        // 获取 Merkle cap 缓冲区
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        // 填充哈希值缓冲区和 Merkle cap 缓冲区
        fill_digests_buf::<F, H>(digests_buf, cap_buf, &leaves[..], cap_height);

        unsafe {
            // 安全：`fill_digests_buf` 和 `cap` 初始化了备用容量，分别为 `num_digests` 和 `len_cap`
            digests.set_len(num_digests);
            cap.set_len(len_cap);
        }

        // 返回 MerkleTree 实例
        Self {
            leaves, // 叶子节点
            digests, // 哈希值
            cap: MerkleCap(cap), // Merkle cap
        }
    }

    pub fn get(&self, i: usize) -> &[F] {
        &self.leaves[i]
    }

    /// Create a Merkle proof from a leaf index.
    pub fn prove(&self, leaf_index: usize) -> MerkleProof<F, H> {
        let cap_height = log2_strict(self.cap.len());
        let siblings =
            merkle_tree_prove::<F, H>(leaf_index, self.leaves.len(), cap_height, &self.digests);

        MerkleProof { siblings }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::extension::Extendable;
    use crate::hash::merkle_proofs::verify_merkle_proof_to_cap;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    pub(crate) fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
        (0..n).map(|_| F::rand_vec(k)).collect()
    }

    fn verify_all_leaves<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        leaves: Vec<Vec<F>>,
        cap_height: usize,
    ) -> Result<()> {
        let tree = MerkleTree::<F, C::Hasher>::new(leaves.clone(), cap_height);
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i);
            verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof)?;
        }
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_cap_height_too_big() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let cap_height = log_n + 1; // Should panic if `cap_height > len_n`.

        let leaves = random_data::<F>(1 << log_n, 7);
        let _ = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves, cap_height);
    }

    #[test]
    fn test_cap_height_eq_log2_len() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, log_n)?;

        Ok(())
    }

    #[test]
    fn test_merkle_trees() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let log_n = 8;
        let n = 1 << log_n;
        let leaves = random_data::<F>(n, 7);

        verify_all_leaves::<F, C, D>(leaves, 1)?;

        Ok(())
    }
    #[test]
    fn test_merkle_tree_prove() {
        const D: usize = 2; // Define the constant D
        type C = PoseidonGoldilocksConfig; // Define the configuration type
        type F = <C as GenericConfig<D>>::F; // Define the field type

        let log_n = 8; // Define the logarithm base 2 of the number of leaves
        let n = 1 << log_n; // Calculate the number of leaves (2^log_n)
        let leaves0 = random_data::<F>(n, 7); // Generate random data for the leaves
        let leaves = leaves0.clone(); // Clone the leaves for later use
        let tree = MerkleTree::<F, <C as GenericConfig<D>>::Hasher>::new(leaves0, 1); // Create a new Merkle tree with the leaves and a cap height of 1

        // Iterate over the leaves and their indices
        for (i, leaf) in leaves.into_iter().enumerate() {
            let proof = tree.prove(i); // Generate a Merkle proof for the current leaf
            let re = verify_merkle_proof_to_cap(leaf, i, &tree.cap, &proof); // Verify the Merkle proof against the tree's cap
            assert!(re.is_ok()); // Assert that the verification is successful
        }
    }
}
