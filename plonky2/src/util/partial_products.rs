#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::iter;

use itertools::Itertools;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::plonk::circuit_builder::CircuitBuilder;

///将 quotient_values 切片按指定大小分块，并计算每个块中所有元素的乘积
pub(crate) fn quotient_chunk_products<F: Field>(
    quotient_values: &[F],
    max_degree: usize,
) -> Vec<F> {
    debug_assert!(max_degree > 1);
    assert!(!quotient_values.is_empty());
    let chunk_size = max_degree;//8
    quotient_values
        // 将 `quotient_values` 切片按 `chunk_size` 大小分块
        .chunks(chunk_size)
        // 对每个块应用闭包，计算块中所有元素的乘积
        .map(|chunk| chunk.iter().copied().product())
        // 将结果收集到一个向量中
        .collect()
}
///计算r1=q0，r2=q0*q1=r1q1，r3=q0*q1*q2=r2*q2,……，rn=q0*q1*…*q(n-1)=r(n-1)*qn,得到r1,r2,…,rn，如[2,3,5]得到[2,2*3,2*3*5]
/// Compute partial products of the original vector `v` such that all products consist of `max_degree`
/// or less elements. This is done until we've computed the product `P` of all elements in the vector.
pub(crate) fn partial_products_and_z_gx<F: Field>(z_x: F, quotient_chunk_products: &[F]) -> Vec<F> {
    assert!(!quotient_chunk_products.is_empty());
    let mut res = Vec::with_capacity(quotient_chunk_products.len());
    let mut acc = z_x;
    for &quotient_chunk_product in quotient_chunk_products {
        acc *= quotient_chunk_product;
        res.push(acc);
    }
    res
}
///上取整-1，如5/2-1=3-1=2，4/2-1=2-1=1
/// Returns the length of the output of `partial_products()` on a vector of length `n`.
pub(crate) fn num_partial_products(n: usize, max_degree: usize) -> usize {
    debug_assert!(max_degree > 1);
    let chunk_size = max_degree;
    // We'll split the product into `n.div_ceil( chunk_size)` chunks, but the last chunk will
    // be associated with Z(gx) itself. Thus we subtract one to get the chunks associated with
    // partial products.
    //div_ceil整除上取整，整除下取整为div_floor
    n.div_ceil(chunk_size) - 1
}

/// Checks the relationship between each pair of partial product accumulators. In particular, this
/// sequence of accumulators starts with `Z(x)`, then contains each partial product polynomials
/// `p_i(x)`, and finally `Z(g x)`. See the partial products section of the Plonky2 paper.
pub(crate) fn check_partial_products<F: Field>(
    numerators: &[F],
    denominators: &[F],
    partials: &[F],
    z_x: F,
    z_gx: F,
    max_degree: usize,
) -> Vec<F> {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    let chunk_size = max_degree;
    numerators
        .chunks(chunk_size)
        .zip_eq(denominators.chunks(chunk_size))
        .zip_eq(product_accs.tuple_windows())
        .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
            let num_chunk_product = nume_chunk.iter().copied().product();
            let den_chunk_product = deno_chunk.iter().copied().product();
            prev_acc * num_chunk_product - next_acc * den_chunk_product
        })
        .collect()
}
pub(crate) fn check_partial_products1<F: Field>(
    numerators: &[F],//分子，[1, 2, 3, 4, 5, 6]
    denominators: &[F],//分母，[1, 1, 1, 1, 1, 1]
    partials: &[F],
    z_x: F,
    z_gx: F,//最终的乘积
    max_degree: usize,
) -> Vec<F> {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    println!("product_accs： {:?}", product_accs);//[1, 2, 24, 720]
    let chunk_size = max_degree;
    println!("chunk_size： {:?}", chunk_size);//2
    println!("numerators： {:?}", numerators);//[1, 2, 3, 4, 5, 6]
    println!("denominators： {:?}", denominators);//[1, 1, 1, 1, 1, 1]
    println!("partials： {:?}", partials);//[2, 24]
    println!("z_x： {:?}", z_x);//1
    println!("z_gx： {:?}", z_gx);//720
    let chunked_numerators = numerators.chunks(chunk_size);//Chunks { v: [1, 2, 3, 4, 5, 6], chunk_size: 2 }
    println!("chunked_numerators： {:?}", chunked_numerators);//
    let chunked_denominators = denominators.chunks(chunk_size);//Chunks { v: [1, 1, 1, 1, 1, 1], chunk_size: 2 }
    println!("chunked_denominators： {:?}", chunked_denominators);//[[1, 1], [1, 1],[1, 1]]
    /*
    zip_eq: 将两个迭代器逐一配对，要求它们长度相等。如果长度不相等，则会引发 panic
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];
    let zipped: Vec<_> = a.iter().zip_eq(b.iter()).collect();
    println!("{:?}", zipped); // 输出: [(1, 4), (2, 5), (3, 6)]
     */
    let zipped_chunks = chunked_numerators.zip_eq(chunked_denominators);
    println!("zipped_chunks： {:?}", zipped_chunks);//[([1,2], [1,1]), ([3,4], [1,1]),([5,6], [1,1])]
    let product_acc_windows = product_accs.tuple_windows();
    println!("product_acc_windows： {:?}", product_acc_windows);//[(1, 2), (2, 24), (24, 720)]

    let result = zipped_chunks
        .zip_eq(product_acc_windows)
        .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
            let num_chunk_product = nume_chunk.iter().copied().product();
            println!("num_chunk_product: {:?}", num_chunk_product); // 2, 12, 30

            let den_chunk_product = deno_chunk.iter().copied().product();
            println!("den_chunk_product: {:?}", den_chunk_product); // 1, 1, 1
            println!("prev_acc: {:?}, next_acc: {:?}", prev_acc, next_acc); //(1, 2), (2, 24), (24, 720)
            let tem0 = prev_acc * num_chunk_product;
            let tem1 = next_acc * den_chunk_product;
            println!("tem0: {:?}, tem1: {:?}", tem0, tem1); // 2, 120

            tem0 - tem1
        })
        .collect();
    result
}
/// Checks the relationship between each pair of partial product accumulators. In particular, this
/// sequence of accumulators starts with `Z(x)`, then contains each partial product polynomials
/// `p_i(x)`, and finally `Z(g x)`. See the partial products section of the Plonky2 paper.
pub(crate) fn check_partial_products_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    numerators: &[ExtensionTarget<D>],
    denominators: &[ExtensionTarget<D>],
    partials: &[ExtensionTarget<D>],
    z_x: ExtensionTarget<D>,
    z_gx: ExtensionTarget<D>,
    max_degree: usize,
) -> Vec<ExtensionTarget<D>> {
    debug_assert!(max_degree > 1);
    let product_accs = iter::once(&z_x)
        .chain(partials.iter())
        .chain(iter::once(&z_gx));
    let chunk_size = max_degree;
    numerators
        .chunks(chunk_size)
        .zip_eq(denominators.chunks(chunk_size))
        .zip_eq(product_accs.tuple_windows())
        .map(|((nume_chunk, deno_chunk), (&prev_acc, &next_acc))| {
            let nume_product = builder.mul_many_extension(nume_chunk);
            let deno_product = builder.mul_many_extension(deno_chunk);
            let next_acc_deno = builder.mul_extension(next_acc, deno_product);
            // Assert that next_acc * deno_product = prev_acc * nume_product.
            builder.mul_sub_extension(prev_acc, nume_product, next_acc_deno)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;

    #[test]
    fn test_partial_products() {
        type F = GoldilocksField;
        let denominators = vec![F::ONE; 6];
        let z_x = F::ONE;
        let v = field_vec(&[1, 2, 3, 4, 5, 6]);
        let z_gx = F::from_canonical_u64(720);
        //将 v 切片按每块长度为2进行分块，并计算每个块中所有元素的乘积
        let quotient_chunks_prods = quotient_chunk_products(&v, 2);
        assert_eq!(quotient_chunks_prods, field_vec(&[2, 12, 30]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        println!("pps_and_z_gx： {:?}", pps_and_z_gx);// [2, 24, 720]
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        println!("pps： {:?}", pps);//[2, 24]
        assert_eq!(pps_and_z_gx, field_vec(&[2, 24, 720]));

        let nums = num_partial_products(v.len(), 2);
        assert_eq!(pps.len(), nums);
        let nums1 = num_partial_products(4, 2);
        println!("nums1： {:?}", nums1);//2
        assert!(check_partial_products1(&v, &denominators, pps, z_x, z_gx, 2)
            .iter()
            .all(|x| x.is_zero()));
        /*
                let quotient_chunks_prods = quotient_chunk_products(&v, 3);
                println!("quotient_chunks_prods：{:?}", quotient_chunks_prods);//[6, 120]
                assert_eq!(quotient_chunks_prods, field_vec(&[6, 120]));
                let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
                println!("pps_and_z_gx： {:?}", pps_and_z_gx);//[6, 720]
                let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
                assert_eq!(pps_and_z_gx, field_vec(&[6, 720]));
                let nums = num_partial_products(v.len(), 3);
                assert_eq!(pps.len(), nums);
                assert!(check_partial_products1(&v, &denominators, pps, z_x, z_gx, 3)
                    .iter()
                    .all(|x| x.is_zero()));

         */
    }
    #[test]
    fn test_partial_products1() {
        type F = GoldilocksField;
        let denominators = vec![F::ONE; 6];
        let z_x = F::ONE;
        let v = field_vec(&[1, 2, 3, 4, 5, 6]);
        let z_gx = F::from_canonical_u64(720);
        //将 v 切片按每块长度为2进行分块，并计算每个块中所有元素的乘积
        let quotient_chunks_prods = quotient_chunk_products(&v, 2);
        assert_eq!(quotient_chunks_prods, field_vec(&[2, 12, 30]));
        let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
        println!("pps_and_z_gx： {:?}", pps_and_z_gx);// [2, 24, 720]
        let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
        println!("pps： {:?}", pps);//[2, 24]
        assert_eq!(pps_and_z_gx, field_vec(&[2, 24, 720]));

        let nums = num_partial_products(v.len(), 2);
        assert_eq!(pps.len(), nums);
        let nums1 = num_partial_products(4, 2);
        println!("nums1： {:?}", nums1);//2
        assert!(check_partial_products1(&v, &denominators, pps, z_x, z_gx, 2)
            .iter()
            .all(|x| x.is_zero()));
        /*
                let quotient_chunks_prods = quotient_chunk_products(&v, 3);
                println!("quotient_chunks_prods：{:?}", quotient_chunks_prods);//[6, 120]
                assert_eq!(quotient_chunks_prods, field_vec(&[6, 120]));
                let pps_and_z_gx = partial_products_and_z_gx(z_x, &quotient_chunks_prods);
                println!("pps_and_z_gx： {:?}", pps_and_z_gx);//[6, 720]
                let pps = &pps_and_z_gx[..pps_and_z_gx.len() - 1];
                assert_eq!(pps_and_z_gx, field_vec(&[6, 720]));
                let nums = num_partial_products(v.len(), 3);
                assert_eq!(pps.len(), nums);
                assert!(check_partial_products1(&v, &denominators, pps, z_x, z_gx, 3)
                    .iter()
                    .all(|x| x.is_zero()));

         */
    }

    fn field_vec<F: Field>(xs: &[usize]) -> Vec<F> {
        xs.iter().map(|&x| F::from_canonical_usize(x)).collect()
    }
    #[test]
    fn test_check_partial_products() {
        type F = GoldilocksField;
        let numerators = field_vec(&[1, 2, 3, 4, 5, 6]);
        let denominators = vec![F::ONE; 6];
        let z_x = F::ONE;
        let z_gx = F::from_canonical_u64(720);
        let partials = field_vec(&[2, 24]);

        let result = check_partial_products(&numerators, &denominators, &partials, z_x, z_gx, 2);
        assert!(result.iter().all(|x| x.is_zero()));
    }

}
