 use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

/// An example of using Plonky2 to prove a statement of the form
/// "I know the 100th element of the Fibonacci sequence, starting with constants a and b."
/// When a == 0 and b == 1, this is proving knowledge of the 100th (standard) Fibonacci number.
fn main() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // The arithmetic circuit.
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for i in 0..4 {
        // println!("{}", i,);
        // println!("begin prev_target:{:?}", prev_target);
        // println!("begin cur_target:{:?}", cur_target);


        let temp = builder.add(prev_target, cur_target);
        // if i==21 {
        //     println!("-----------------------");
        // }
        // println!("after prev_target:{:?}", prev_target);
        // println!("after cur_target:{:?}", cur_target);
        // println!("after output:{:?}", temp);
        // println!("gate_instances{:?}", builder.gate_instances);
        //builder.print_copy_constraints();
        //println!("current_slots{:?}", builder.current_slots);
        //println!("constants_to_targets{:?}", builder.constants_to_targets);
       //builder.print_constants_to_targets();

        //println!("current_slots{:?}", builder.current_slots);
        // println!("gates{:?}", builder.gates);
        //println!("base_arithmetic_results{:?}", builder.base_arithmetic_results);

        prev_target = cur_target;
        cur_target = temp;
    }
    // println!("gate_instances{:?}", builder.gate_instances);
    //println!("copy_constraints{:?}", builder.copy_constraints);

    // Public inputs are the two initial values (provided below) and the result (which is generated).
    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);
    builder.print_public_inputs();

    // Provide initial values.
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO)?;
    pw.set_target(initial_b, F::ONE)?;
    //println!("pw:{:?}", pw);

    let data = builder.build::<C>();
    let proof = data.prove(pw)?;

    println!(
        "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
        proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
    );

    data.verify(proof)
}
