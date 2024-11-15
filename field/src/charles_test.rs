
#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    //use alloc::vec::Vec;
    use crate::fft::{fft, fft_with_options, ifft};
    use crate::goldilocks_field::GoldilocksField;
    use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
    use crate::types::Field;
    use plonky2_util::{log2_ceil, log2_strict};
    use std::time::Instant;

    use rand::rngs::OsRng;
    use rand::Rng;

    //use super::*;
    use crate::types::Sample;
    #[test]
    fn fft_and_ifft() {
        type F = GoldilocksField;
        let degree = 200usize;
        let degree_padded = degree.next_power_of_two();

        // Create a vector of coeffs; the first degree of them are
        // "random", the last degree_padded-degree of them are zero.
        let coeffs = (0..degree)
            .map(|i| F::from_canonical_usize(i * 1337 % 100))
            .chain(core::iter::repeat(F::ZERO).take(degree_padded - degree))
            .collect::<Vec<_>>();
        assert_eq!(coeffs.len(), degree_padded);
        let coefficients = PolynomialCoeffs { coeffs };
        //println!("coefficients:{:?}", coefficients);

        let points = fft(coefficients.clone());
        assert_eq!(points, evaluate_naive(&coefficients));

        let interpolated_coefficients = ifft(points);
        for i in 0..degree {
            assert_eq!(interpolated_coefficients.coeffs[i], coefficients.coeffs[i]);
        }
        for i in degree..degree_padded {
            assert_eq!(interpolated_coefficients.coeffs[i], F::ZERO);
        }

        for r in 0..4 {
            // expand coefficients by factor 2^r by filling with zeros
            let zero_tail = coefficients.lde(r);
            assert_eq!(
                fft(zero_tail.clone()),
                fft_with_options(zero_tail, Some(r), None)
            );
        }
    }

    fn evaluate_naive<F: Field>(coefficients: &PolynomialCoeffs<F>) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_padded = 1 << log2_ceil(degree);

        let coefficients_padded = coefficients.padded(degree_padded);
        evaluate_naive_power_of_2(&coefficients_padded)
    }

    fn evaluate_naive_power_of_2<F: Field>(
        coefficients: &PolynomialCoeffs<F>,
    ) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_log = log2_strict(degree);

        let subgroup = F::two_adic_subgroup(degree_log);

        let values = subgroup
            .into_iter()
            .map(|x| evaluate_at_naive(coefficients, x))
            .collect();
        PolynomialValues::new(values)
    }

    fn evaluate_at_naive<F: Field>(coefficients: &PolynomialCoeffs<F>, point: F) -> F {
        let mut sum = F::ZERO;
        let mut point_power = F::ONE;
        for &c in &coefficients.coeffs {
            sum += c * point_power;
            point_power *= point;
        }
        sum
    }
    #[test]
    fn test_trimmed() {
        type F = GoldilocksField;

        assert_eq!(
            PolynomialCoeffs::<F> { coeffs: vec![] }.trimmed(),
            PolynomialCoeffs::<F> { coeffs: vec![] }
        );
        assert_eq!(
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ZERO]
            }
                .trimmed(),
            PolynomialCoeffs::<F> { coeffs: vec![] }
        );
        assert_eq!(
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ONE, F::TWO, F::ZERO, F::ZERO]
            }
                .trimmed(),
            PolynomialCoeffs::<F> {
                coeffs: vec![F::ONE, F::TWO]
            }
        );
    }

    #[test]
    fn test_coset_fft() {
        type F = GoldilocksField;

        let k = 8;
        let n = 1 << k;
        let poly = PolynomialCoeffs::new(F::rand_vec(n));
        let shift = F::rand();
        let coset_evals = poly.coset_fft(shift).values;

        let generator = F::primitive_root_of_unity(k);
        let naive_coset_evals = F::cyclic_subgroup_coset_known_order(generator, shift, n)
            .into_iter()
            .map(|x| poly.eval(x))
            .collect::<Vec<_>>();
        assert_eq!(coset_evals, naive_coset_evals);

        let ifft_coeffs = PolynomialValues::new(coset_evals).coset_ifft(shift);
        assert_eq!(poly, ifft_coeffs);
    }

    #[test]
    fn test_coset_ifft() {
        type F = GoldilocksField;

        let k = 8;
        let n = 1 << k;
        let evals = PolynomialValues::new(F::rand_vec(n));
        let shift = F::rand();
        let coeffs = evals.clone().coset_ifft(shift);

        let generator = F::primitive_root_of_unity(k);
        let naive_coset_evals = F::cyclic_subgroup_coset_known_order(generator, shift, n)
            .into_iter()
            .map(|x| coeffs.eval(x))
            .collect::<Vec<_>>();
        assert_eq!(evals, naive_coset_evals.into());

        let fft_evals = coeffs.coset_fft(shift);
        assert_eq!(evals, fft_evals);
    }

    #[test]
    fn test_polynomial_multiplication() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000));
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg));
        let m1 = &a * &b;
        let m2 = &a * &b;
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(m1.eval(x), a.eval(x) * b.eval(x));
            assert_eq!(m2.eval(x), a.eval(x) * b.eval(x));
        }
    }

    #[test]
    fn test_inv_mod_xn() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let a_deg = rng.gen_range(0..1_000);
        let n = rng.gen_range(1..1_000);
        let mut a = PolynomialCoeffs::new(F::rand_vec(a_deg + 1));
        if a.coeffs[0].is_zero() {
            a.coeffs[0] = F::ONE; // First coefficient needs to be nonzero.
        }
        let b = a.inv_mod_xn(n);
        let mut m = &a * &b;
        m.coeffs.truncate(n);
        m.trim();
        assert_eq!(
            m,
            PolynomialCoeffs::new(vec![F::ONE]),
            "a: {:#?}, b:{:#?}, n:{:#?}, m:{:#?}",
            a,
            b,
            n,
            m
        );
    }

    #[test]
    fn test_polynomial_long_division() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000));
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg));
        let (q, r) = a.div_rem_long_division(&b);
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x));
        }
    }

    #[test]
    fn test_polynomial_division() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let (a_deg, b_deg) = (rng.gen_range(1..10_000), rng.gen_range(1..10_000));
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::new(F::rand_vec(b_deg));
        let (q, r) = a.div_rem(&b);
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x));
        }
    }

    #[test]
    fn test_polynomial_division_by_constant() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let a_deg = rng.gen_range(1..10_000);
        let a = PolynomialCoeffs::new(F::rand_vec(a_deg));
        let b = PolynomialCoeffs::from(vec![F::rand()]);
        let (q, r) = a.div_rem(&b);
        for _ in 0..1000 {
            let x = F::rand();
            assert_eq!(a.eval(x), b.eval(x) * q.eval(x) + r.eval(x));
        }
    }

    // Test to see which polynomial division method is faster for divisions of the type
    // `(X^n - 1)/(X - a)
    #[test]
    fn test_division_linear() {
        type F = GoldilocksField;
        let mut rng = OsRng;
        let l = 14;
        let n = 1 << l;
        let g = F::primitive_root_of_unity(l);
        let xn_minus_one = {
            let mut xn_min_one_vec = vec![F::ZERO; n + 1];
            xn_min_one_vec[n] = F::ONE;
            xn_min_one_vec[0] = F::NEG_ONE;
            PolynomialCoeffs::new(xn_min_one_vec)
        };

        let a = g.exp_u64(rng.gen_range(0..(n as u64)));
        let denom = PolynomialCoeffs::new(vec![-a, F::ONE]);
        let now = Instant::now();
        xn_minus_one.div_rem(&denom);
        println!("Division time: {:?}", now.elapsed());
        let now = Instant::now();
        xn_minus_one.div_rem_long_division(&denom);
        println!("Division time: {:?}", now.elapsed());
    }

    #[test]
    fn eq() {
        type F = GoldilocksField;
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO, F::ZERO])
        );
        assert_eq!(
            PolynomialCoeffs::<F>::new(vec![F::ONE]),
            PolynomialCoeffs::new(vec![F::ONE, F::ZERO])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![]),
            PolynomialCoeffs::new(vec![F::ONE])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ZERO, F::ONE])
        );
        assert_ne!(
            PolynomialCoeffs::<F>::new(vec![F::ZERO]),
            PolynomialCoeffs::new(vec![F::ONE, F::ZERO])
        );
    }
}