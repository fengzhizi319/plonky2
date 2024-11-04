use alloc::vec::Vec;

use num::bigint::BigUint;

use crate::types::Field;

/// Finds a set of shifts that result in unique cosets for the multiplicative subgroup of size
/// `2^subgroup_bits`.
///  查找一组移位值，这些移位值会生成大小为 `2^subgroup_bits` 的乘法子群的唯一陪集。
pub fn get_unique_coset_shifts<F: Field>(subgroup_size: usize, num_shifts: usize) -> Vec<F> {
    // From Lagrange's theorem.根据拉格朗日定理计算陪集的数量。
    //拉格朗日定理：群的阶=子群的阶*陪集的数量
    let num_cosets = (F::order() - 1u32) / (subgroup_size as u32);
    assert!(
        BigUint::from(num_shifts) <= num_cosets,
        "The subgroup does not have enough distinct cosets"
    );

    // Let g be a generator of the entire multiplicative group. Let n be the order of the subgroup.
    // The subgroup can be written as <g^(|F*| / n)>. We can use g^0, ..., g^(num_shifts - 1) as our
    // shifts, since g^i <g^(|F*| / n)> are distinct cosets provided i < |F*| / n, which we checked.

    // 设 g 是整个乘法群的生成元。设 n 是子群的阶。
    // 子群可以表示为 <g^(|F*| / n)>。我们可以使用 g^0, ..., g^(num_shifts - 1) 作为我们的移位值，
    // 因为 g^i <g^(|F*| / n)> 是不同的陪集，前提是 i < |F*| / n，这一点我们已经检查过了。

    F::MULTIPLICATIVE_GROUP_GENERATOR
        .powers()//获取生成元的幂
        .take(num_shifts)// 取前 num_shifts 个幂
        .collect()// 收集成一个向量
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::cosets::get_unique_coset_shifts;
    use crate::goldilocks_field::GoldilocksField;
    use crate::types::Field;

    #[test]
    fn distinct_cosets() {
        type F = GoldilocksField;//p=2**32 * (2**32 - 1) + 1
        const SUBGROUP_BITS: usize = 5;
        const NUM_SHIFTS: usize = 50;

        let generator = F::primitive_root_of_unity(SUBGROUP_BITS);
        let subgroup_size = 1 << SUBGROUP_BITS;

        let shifts = get_unique_coset_shifts::<F>(subgroup_size, NUM_SHIFTS);
        let mut union = HashSet::new();
        for shift in shifts {
            let coset = F::cyclic_subgroup_coset_known_order(generator, shift, subgroup_size);
            assert!(
                coset.into_iter().all(|x| union.insert(x)),
                "Duplicate element!"
            );
        }
    }
}
