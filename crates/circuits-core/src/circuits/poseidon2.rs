//! Poseidon2 permutation circuit over M31 (p = 2^31 - 1).
use std::array::from_fn;

use crate::{Circuit, CircuitBuilder, Feed, Node, ops::{add_m31, mul_m31}};

pub(crate) const N_STATE: usize = 16;
pub(crate) const N_HALF_FULL_ROUNDS: usize = 4;
pub(crate) const N_PARTIAL_ROUNDS: usize = 14;

/// Number of M31 elements in the rate portion of the sponge state.
pub const RATE: usize = 8;

const MAT_INTERNAL_DIAG_M_1: [u32; N_STATE] = [                                                                                  
    0x07b80ac4, 0x6bd9cb33, 0x48ee3f9f, 0x4f63dd19,                                                                              
    0x18c546b3, 0x5af89e8b, 0x4ff23de8, 0x4f78aaf6,                                                                              
    0x53bdc6d4, 0x5c59823e, 0x2a471c72, 0x4c975e79,                                                                              
    0x58dc64d4, 0x06e9315d, 0x2cf32286, 0x2fb6755d,                                                                              
];  

const EXTERNAL_ROUND_CONSTS: [[u32; N_STATE]; 2 * N_HALF_FULL_ROUNDS] = [
    [0x768bab52, 0x70e0ab7d, 0x3d266c8a, 0x6da42045, 0x600fef22, 0x41dace6b, 0x64f9bdd4, 0x5d42d4fe, 0x76b1516d, 0x6fc9a717, 0x70ac4fb6, 0x00194ef6, 0x22b644e2, 0x1f7916d5, 0x47581be2, 0x2710a123],
    [0x6284e867, 0x018d3afe, 0x5df99ef3, 0x4c1e467b, 0x566f6abc, 0x2994e427, 0x538a6d42, 0x5d7bf2cf, 0x7fda2dab, 0x0fd854c4, 0x46922fca, 0x3d7763a1, 0x19fd05ca, 0x0a4bbb43, 0x15075851, 0x3d903d76],
    [0x2d290ff7, 0x40809fa0, 0x59dac6ec, 0x127927a2, 0x6bbf0ea0, 0x0294140f, 0x24742976, 0x6e84c081, 0x22484f4a, 0x354cae59, 0x0453ffe1, 0x3f47a3cc, 0x0088204e, 0x6066e109, 0x3b7c4b80, 0x6b55665d],
    [0x3bc4b897, 0x735bf378, 0x508daf42, 0x1884fc2b, 0x7214f24c, 0x7498be0a, 0x1a60e640, 0x3303f928, 0x29b46376, 0x5c96bb68, 0x65d097a5, 0x1d358e9f, 0x4a9a9017, 0x4724cf76, 0x347af70f, 0x1e77e59a],
    [0x57090613, 0x1fa42108, 0x17bbef50, 0x1ff7e11c, 0x047b24ca, 0x4e140275, 0x4fa086f5, 0x079b309c, 0x1159bd47, 0x6d37e4e5, 0x075d8dce, 0x12121ca0, 0x7f6a7c40, 0x68e182ba, 0x5493201b, 0x0444a80e],
    [0x0064f4c6, 0x6467abe6, 0x66975762, 0x2af68f9b, 0x345b33be, 0x1b70d47f, 0x053db717, 0x381189cb, 0x43b915f8, 0x20df3694, 0x0f459d26, 0x77a0e97b, 0x2f73e739, 0x1876c2f9, 0x65a0e29a, 0x4cabefbe],
    [0x5abd1268, 0x4d34a760, 0x12771799, 0x69a0c9ac, 0x39091e55, 0x7f611cd0, 0x3af055da, 0x7ac0bbdf, 0x6e0f3a24, 0x41e3b6f7, 0x49b3756d, 0x568bc538, 0x20c079d8, 0x1701c72c, 0x7670dc6c, 0x5a439035],
    [0x7c93e00e, 0x561fbb4d, 0x1178907b, 0x02737406, 0x32fb24f1, 0x6323b60a, 0x6ab12418, 0x42c99cea, 0x155a0b97, 0x53d1c6aa, 0x2bd20347, 0x279b3d73, 0x4f5f3c70, 0x0245af6c, 0x238359d3, 0x49966a59],
];

const INTERNAL_ROUND_CONSTS: [u32; N_PARTIAL_ROUNDS] = [
    0x7f7ec4bf, 0x0421926f, 0x5198e669, 0x34db3148, 0x4368bafd, 0x66685c7f, 0x78d3249a, 0x60187881, 0x76dad67a, 0x0690b437, 0x1ea95311, 0x40e5369a, 0x38f103fc, 0x1d226a21,
];

type Word = [Node<Feed>; 31];                                                                                                    
type State = [Word; N_STATE];

// S-box: x^5 mod p
fn pow5(builder: &mut CircuitBuilder, x: Word) -> Word {
    let x2 = mul_m31(builder, x, x);
    let x4 = mul_m31(builder, x2, x2);
    mul_m31(builder, x4, x)
}

fn add_constant(builder: &mut CircuitBuilder, x: Word, c: u32) -> Word {
    let c_bits: Word = from_fn(|i| {
        if (c >> i) & 1 == 1 {
            builder.get_const_one()
        } else {
            builder.get_const_zero()
        }
    });
    add_m31(builder, x, c_bits)
}

fn apply_m4(builder: &mut CircuitBuilder, x: [Word; 4]) -> [Word; 4] {
    let t0 = add_m31(builder, x[0], x[1]);
    let t1 = add_m31(builder, x[2], x[3]);

    let t02 = add_m31(builder, t0, t0);       // 2*t0
    let t12 = add_m31(builder, t1, t1);       // 2*t1

    let x1_doubled = add_m31(builder, x[1], x[1]);
    let t2 = add_m31(builder, x1_doubled, t1); // 2*x1 + t1

    let x3_doubled = add_m31(builder, x[3], x[3]);
    let t3 = add_m31(builder, x3_doubled, t0); // 2*x3 + t0

    let t12_doubled = add_m31(builder, t12, t12); // 4*t1
    let t4 = add_m31(builder, t12_doubled, t3);   // 4*t1 + t3

    let t02_doubled = add_m31(builder, t02, t02); // 4*t0
    let t5 = add_m31(builder, t02_doubled, t2);   // 4*t0 + t2

    let t6 = add_m31(builder, t3, t5);
    let t7 = add_m31(builder, t2, t4);

    [t6, t5, t7, t4]
}

fn apply_external_round_matrix(builder: &mut CircuitBuilder, mut state: State) -> State {
    for i in 0..4 {
        let chunk = apply_m4(builder, [state[4*i], state[4*i+1], state[4*i+2], state[4*i+3]]);
        state[4*i]   = chunk[0];
        state[4*i+1] = chunk[1];
        state[4*i+2] = chunk[2];
        state[4*i+3] = chunk[3];
    }

    for j in 0..4 {
        let s0 = add_m31(builder, state[j], state[j+4]);
        let s1 = add_m31(builder, state[j+8], state[j+12]);
        let s  = add_m31(builder, s0, s1);

        state[j]    = add_m31(builder, state[j],    s);
        state[j+4]  = add_m31(builder, state[j+4],  s);
        state[j+8]  = add_m31(builder, state[j+8],  s);
        state[j+12] = add_m31(builder, state[j+12], s);
    }

    state
}

fn apply_internal_round_matrix(builder: &mut CircuitBuilder, mut state: State) -> State {                                        
    let mut sum = state[0];                                                                                                      
    for i in 1..N_STATE {
        sum = add_m31(builder, sum, state[i]);                                                                                   
    }           
                                                                                                                                
    // state[i] = state[i] * diag[i] + sum
    for i in 0..N_STATE {                                                                                                        
        let diag_bits: Word = from_fn(|b| {
            if (MAT_INTERNAL_DIAG_M_1[i] >> b) & 1 == 1 {                                                                        
                builder.get_const_one()                                                                                          
            } else {                                                                                                             
                builder.get_const_zero()                                                                                         
            }                                                                                                                    
        });     
        let product = mul_m31(builder, state[i], diag_bits);
        state[i] = add_m31(builder, product, sum);                                                                               
    }
                                                                                                                                
    state       
}

/// Returns a Poseidon2 permutation circuit:
///
/// `fn(state: [u32; 16]) -> [u32; 16]`
///
/// Inputs and outputs are M31 field elements (31-bit values in [0, p)).
pub fn permute() -> Circuit {
    let mut builder = CircuitBuilder::new();

    let state: State = from_fn(|_| from_fn(|_| builder.add_input()));

    let output = permute_internal(&mut builder, state);

    for word in output {
        for node in word {
            builder.add_output(node);
        }
    }

    builder.build().unwrap()
}

/// Returns a Poseidon2 permutation circuit using 32-bit word representation:
///
/// `fn(state: [u32; 16]) -> [u32; 16]`
///
/// Each word is a 32-bit value where bit 31 is ignored on input and set to 0 on
/// output. The low 31 bits hold the M31 field element.
pub fn permute_u32() -> Circuit {
    println!("Hello from Poseidon Permute u32");
    let mut builder = CircuitBuilder::new();

    let state_u32: [[_; 32]; N_STATE] = from_fn(|_| from_fn(|_| builder.add_input()));

    // Use only the low 31 bits of each word.
    let state_m31: State = from_fn(|i| from_fn(|b| state_u32[i][b]));

    let output_m31 = permute_internal(&mut builder, state_m31);

    // Each output word gets its own zero gate (same node cannot appear in
    // multiple output slots — the builder's id_map would alias them).
    for word in output_m31 {
        for node in word {
            builder.add_output(node);
        }
        let zero = builder.add_xor_gate(word[0], word[0]);
        builder.add_output(zero);
    }

    builder.build().unwrap()
}

/// Returns a circuit that absorbs `RATE` M31 elements into the rate portion of
/// the Poseidon2 state using field addition:
///
/// `fn(rate: [u32; 8], input: [u32; 8]) -> [u32; 8]`
///
/// Each word is a 32-bit value where bit 31 is ignored on input and set to 0 on
/// output.
pub fn absorb_m31() -> Circuit {
    let mut builder = CircuitBuilder::new();

    let rate: [[_; 32]; RATE] = from_fn(|_| from_fn(|_| builder.add_input()));
    let input: [[_; 32]; RATE] = from_fn(|_| from_fn(|_| builder.add_input()));

    for i in 0..RATE {
        let rate_m31: Word = from_fn(|b| rate[i][b]);
        let input_m31: Word = from_fn(|b| input[i][b]);
        let sum = add_m31(&mut builder, rate_m31, input_m31);
        for node in sum {
            builder.add_output(node);
        }
        let zero = builder.add_xor_gate(sum[0], sum[0]);
        builder.add_output(zero);
    }

    builder.build().unwrap()
}

fn permute_internal(builder: &mut CircuitBuilder, mut state: State) -> State {
    state = apply_external_round_matrix(builder, state);

    for round in 0..N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] = add_constant(builder, state[i], EXTERNAL_ROUND_CONSTS[round][i]);
        }
        for i in 0..N_STATE {
            state[i] = pow5(builder, state[i]);
        }
        state = apply_external_round_matrix(builder, state);
    }

    for round in 0..N_PARTIAL_ROUNDS {
        state[0] = add_constant(builder, state[0], INTERNAL_ROUND_CONSTS[round]);
        state[0] = pow5(builder, state[0]);
        state = apply_internal_round_matrix(builder, state);
    }

    for round in N_HALF_FULL_ROUNDS..2 * N_HALF_FULL_ROUNDS {
        for i in 0..N_STATE {
            state[i] = add_constant(builder, state[i], EXTERNAL_ROUND_CONSTS[round][i]);
        }
        for i in 0..N_STATE {
            state[i] = pow5(builder, state[i]);
        }
        state = apply_external_round_matrix(builder, state);
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon2_permute() {
        let circ = permute();

        let input: [u32; 16] = std::array::from_fn(|i| i as u32);

        let expected: [u32; 16] = [
            0x505d9689, 0x3b64c904, 0x79e2fd81, 0x4ba8015f,
            0x24b6d2f5, 0x23845add, 0x521f4314, 0x69dfb019,
            0x2aaae419, 0x6cb4502c, 0x6f7fa65a, 0x75feff24,
            0x128d6587, 0x515877e4, 0x037f4dd7, 0x134b427f,
        ];

        let input_bits: Vec<bool> = input.iter()
            .flat_map(|&x| (0..31).map(move |b| (x >> b) & 1 == 1))
            .collect();

        let output_bits: Vec<bool> = circ.evaluate(input_bits.into_iter()).unwrap().into_iter().collect();
 
        let output: [u32; 16] = std::array::from_fn(|i| {
            (0..31).fold(0u32, |acc, b| {
                if output_bits[i * 31 + b] { acc | (1 << b) } else { acc }
            })
        });

        assert_eq!(output, expected);
    }

}