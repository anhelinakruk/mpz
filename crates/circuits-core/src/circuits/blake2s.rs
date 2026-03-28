//! Blake2s compression circuit.

use std::array::from_fn;

use crate::{
    Circuit, CircuitBuilder, Feed, Node,
    ops::{wrapping_add, xor},
};

// SIGMA permutation schedule used in Blake2s hashing.
// Each of the 10 rounds uses a different permutation of message word indices.
// Ref: https://www.ietf.org/rfc/rfc7693#section-2.7
pub(crate) const SIGMA: [[usize; 16]; 10] = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
    [11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4],
    [ 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8],
    [ 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13],
    [ 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9],
    [12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11],
    [13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10],
    [ 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5],
    [10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0],
];

type State = [[Node<Feed>; 32]; 16];
type Msg = [[Node<Feed>; 32]; 16];

/// Returns a Blake2s compression circuit with the following signature:
///
/// `fn(h: [u32; 8], m: [u32; 16], v_upper: [u32; 8]) -> [u32; 8]`
///
/// - `h`: chaining value (previous hash state)
/// - `m`: message block
/// - `v_upper`: lower half of the initial working state, pre-computed by the caller as:
///   `[IV[0], IV[1], IV[2], IV[3], IV[4]^t[0], IV[5]^t[1], IV[6]^f, IV[7]]`
///   where `t` is the byte counter and `f` is `0xFFFFFFFF` for the last block, else `0`.
pub fn compress() -> Circuit {
    let mut builder = CircuitBuilder::new();

    let h: [[_; 32]; 8] = from_fn(|_| from_fn(|_| builder.add_input()));
    let m: Msg = from_fn(|_| from_fn(|_| builder.add_input()));
    let v_upper: [[_; 32]; 8] = from_fn(|_| from_fn(|_| builder.add_input()));

    let output = compress_internal(&mut builder, h, m, v_upper);

    for word in output {
        for node in word {
            builder.add_output(node);
        }
    }

    builder.build().unwrap()
}

// Mix function used in Blake2s hashing.
// Ref: https://www.ietf.org/rfc/rfc7693#section-3.1
fn mix(
    builder: &mut CircuitBuilder,
    state: &mut State,
    msg: &Msg,
    state_indices: (usize, usize, usize, usize),
    msg_indices: (usize, usize),
) {
    // Note that bits are stored in LSB0 order, so the word-wise rightward rotation
    // translates to a leftward rotation of the bitslice.

    let (a, b, c, d) = state_indices;
    let (mx, my) = msg_indices;
    state[a] = wrapping_add(builder, &state[a], &state[b])
        .as_slice()
        .try_into()
        .unwrap();
    state[a] = wrapping_add(builder, &state[a], &msg[mx])
        .as_slice()
        .try_into()
        .unwrap();

    state[d] = xor(builder, state[d], state[a]);
    state[d].rotate_left(16);

    state[c] = wrapping_add(builder, &state[c], &state[d])
        .as_slice()
        .try_into()
        .unwrap();

    state[b] = xor(builder, state[b], state[c]);
    state[b].rotate_left(12);

    state[a] = wrapping_add(builder, &state[a], &state[b])
        .as_slice()
        .try_into()
        .unwrap();
    state[a] = wrapping_add(builder, &state[a], &msg[my])
        .as_slice()
        .try_into()
        .unwrap();

    state[d] = xor(builder, state[d], state[a]);
    state[d].rotate_left(8);

    state[c] = wrapping_add(builder, &state[c], &state[d])
        .as_slice()
        .try_into()
        .unwrap();

    state[b] = xor(builder, state[b], state[c]);
    state[b].rotate_left(7);
}

// Round function used in Blake2s hashing.
// Ref: https://www.rfc-editor.org/rfc/rfc7693#section-3.2
fn round(builder: &mut CircuitBuilder, state: &mut State, msg: &Msg, sigma: &[usize; 16]) {
    // Mix columns.
    mix(builder, state, msg, (0, 4,  8, 12), (sigma[0],  sigma[1]));
    mix(builder, state, msg, (1, 5,  9, 13), (sigma[2],  sigma[3]));
    mix(builder, state, msg, (2, 6, 10, 14), (sigma[4],  sigma[5]));
    mix(builder, state, msg, (3, 7, 11, 15), (sigma[6],  sigma[7]));

    // Mix diagonals.
    mix(builder, state, msg, (0, 5, 10, 15), (sigma[8],  sigma[9]));
    mix(builder, state, msg, (1, 6, 11, 12), (sigma[10], sigma[11]));
    mix(builder, state, msg, (2, 7,  8, 13), (sigma[12], sigma[13]));
    mix(builder, state, msg, (3, 4,  9, 14), (sigma[14], sigma[15]));
}

// Compression function used in Blake2s hashing.
// Ref: https://www.rfc-editor.org/rfc/rfc7693#section-3.2
fn compress_internal(
    builder: &mut CircuitBuilder,
    h: [[Node<Feed>; 32]; 8],
    m: Msg,
    v_upper: [[Node<Feed>; 32]; 8],
) -> [[Node<Feed>; 32]; 8] {
    // Build the full working state v[0..16]:
    //   v[0..7]  = h[0..7]
    //   v[8..15] = v_upper[0..7]  (caller pre-computed IV ^ t/f)
    let mut v: State = from_fn(|i| {
        if i < 8 { h[i] } else { v_upper[i - 8] }
    });

    // 10 rounds, each using a different SIGMA permutation.
    for sigma in &SIGMA {
        round(builder, &mut v, &m, sigma);
    }

    // Finalize: h[i] ^= v[i] ^ v[i+8]  for i in 0..8
    from_fn(|i| {
        let tmp = xor(builder, v[i], v[i + 8]);
        xor(builder, h[i], tmp)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate;

    // Builds a circuit that applies mix to positions (0,1,2,3) with msg indices (0,1)
    // and outputs the first 4 state words.
    fn build_mix_circuit() -> crate::Circuit {
        let mut builder = CircuitBuilder::new();
        let mut state: [[_; 32]; 16] = from_fn(|_| from_fn(|_| builder.add_input()));
        let msg: [[_; 32]; 16] = from_fn(|_| from_fn(|_| builder.add_input()));
        mix(&mut builder, &mut state, &msg, (0, 1, 2, 3), (0, 1));
        for (i, word) in state.into_iter().enumerate() {
            if i < 4 {
                for node in word {
                    builder.add_output(node);
                }
            }
        }
        builder.build().unwrap()
    }

    #[test]
    fn test_blake2s_mix() {
        let circ = build_mix_circuit();

        let cases: &[([u32; 16], [u32; 16])] = &[
            ([0u32; 16], [0u32; 16]),
            ([1u32; 16], [1u32; 16]),
            ([u32::MAX; 16], [u32::MAX; 16]),
            (
                [0xdeadbeef, 0xcafebabe, 0xfeedface, 0xbadc0ded, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0x11111111, 0x22222222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            (
                [0xAAAAAAAA, 0x55555555, 0xFFFF0000, 0x0000FFFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0xF0F0F0F0, 0x0F0F0F0F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];

        for (n, (state_vals, msg_vals)) in cases.iter().enumerate() {
            let mut expected = *state_vals;
            reference::mix(&mut expected, 0, 1, 2, 3, msg_vals[0], msg_vals[1]);

            let mut inputs = Vec::new();
            inputs.extend_from_slice(state_vals);
            inputs.extend_from_slice(msg_vals);

            let output: Vec<u32> = evaluate!(&circ, inputs).unwrap();
            for i in 0..4 {
                assert_eq!(output[i], expected[i], "mix case {n}: word {i} mismatch");
            }
        }
    }

    #[test]
    fn test_blake2s_round() {
        let mut builder = CircuitBuilder::new();
        let mut state: [[_; 32]; 16] = from_fn(|_| from_fn(|_| builder.add_input()));
        let msg: [[_; 32]; 16] = from_fn(|_| from_fn(|_| builder.add_input()));

        round(&mut builder, &mut state, &msg, &SIGMA[0]);

        for word in state {
            for node in word {
                builder.add_output(node);
            }
        }
        let circ = builder.build().unwrap();

        let state_vals: [u32; 16] = std::array::from_fn(|i| i as u32);
        let msg_vals = [0u32; 16];

        let mut expected = state_vals;
        reference::round(&mut expected, &msg_vals, &SIGMA[0]);

        let mut inputs = Vec::new();
        inputs.extend_from_slice(&state_vals);
        inputs.extend_from_slice(&msg_vals);

        let output: Vec<u32> = evaluate!(&circ, inputs).unwrap();
        for i in 0..16 {
            assert_eq!(output[i], expected[i], "round: word {i} mismatch");
        }
    }

    #[test]
    fn test_blake2s_compress() {
        let circ = compress();

        // inputs to circuit: h[8] + m[16] + v_upper[8]
        // inputs to reference: same
        let cases: &[([u32; 8], [u32; 16], [u32; 8])] = &[
            ([0u32; 8], [0u32; 16], [0u32; 8]),
            ([1u32; 8], [1u32; 16], [1u32; 8]),
            ([u32::MAX; 8], [u32::MAX; 16], [u32::MAX; 8]),
            (
                std::array::from_fn(|i| i as u32),
                std::array::from_fn(|i| (i + 8) as u32),
                std::array::from_fn(|i| (i + 24) as u32),
            ),
        ];

        for (n, (h, m, v_upper)) in cases.iter().enumerate() {
            let mut expected_h = *h;
            reference::compress(&mut expected_h, m, v_upper);

            let mut inputs = Vec::new();
            inputs.extend_from_slice(h);
            inputs.extend_from_slice(m);
            inputs.extend_from_slice(v_upper);

            let output: Vec<u32> = evaluate!(&circ, inputs).unwrap();
            for i in 0..8 {
                assert_eq!(output[i], expected_h[i], "compress case {n}: word {i} mismatch");
            }
        }
    }

    mod reference {
        use super::SIGMA;

        // G mixing function — identical to Blake3, same rotation constants (16,12,8,7).
        // Ref: https://www.rfc-editor.org/rfc/rfc7693#section-3.1
        pub(crate) fn mix(
            v: &mut [u32; 16],
            a: usize, b: usize, c: usize, d: usize,
            x: u32, y: u32,
        ) {
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
            v[d] = (v[d] ^ v[a]).rotate_right(16);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(12);
            v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
            v[d] = (v[d] ^ v[a]).rotate_right(8);
            v[c] = v[c].wrapping_add(v[d]);
            v[b] = (v[b] ^ v[c]).rotate_right(7);
        }

        // One round using the given SIGMA permutation.
        // Ref: https://www.rfc-editor.org/rfc/rfc7693#section-3.2
        pub(crate) fn round(v: &mut [u32; 16], m: &[u32; 16], sigma: &[usize; 16]) {
            mix(v, 0, 4,  8, 12, m[sigma[0]],  m[sigma[1]]);
            mix(v, 1, 5,  9, 13, m[sigma[2]],  m[sigma[3]]);
            mix(v, 2, 6, 10, 14, m[sigma[4]],  m[sigma[5]]);
            mix(v, 3, 7, 11, 15, m[sigma[6]],  m[sigma[7]]);
            mix(v, 0, 5, 10, 15, m[sigma[8]],  m[sigma[9]]);
            mix(v, 1, 6, 11, 12, m[sigma[10]], m[sigma[11]]);
            mix(v, 2, 7,  8, 13, m[sigma[12]], m[sigma[13]]);
            mix(v, 3, 4,  9, 14, m[sigma[14]], m[sigma[15]]);
        }

        // Full Blake2s compression F.
        // Ref: https://www.rfc-editor.org/rfc/rfc7693#section-3.2
        pub(crate) fn compress(h: &mut [u32; 8], m: &[u32; 16], v_upper: &[u32; 8]) {
            let mut v = [0u32; 16];
            v[..8].copy_from_slice(h);
            v[8..].copy_from_slice(v_upper);

            for sigma in &SIGMA {
                round(&mut v, m, sigma);
            }

            for i in 0..8 {
                h[i] ^= v[i] ^ v[i + 8];
            }
        }
    }
}
