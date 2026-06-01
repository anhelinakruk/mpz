//! Compact binary serialization for circuits.
//!
//! Uses 1-byte gate type + u32 IDs instead of bincode's 4-byte variant + usize IDs.
//! XOR/AND gate: 13 bytes vs 28 bytes in bincode.

use crate::{
    Circuit,
    components::{Feed, Gate, Node, Sink},
};

const MAGIC: &[u8; 4] = b"CIRC";
const VERSION: u8 = 1;
// MAGIC(4) + VERSION(1) + n_inputs(4) + output_start(4) + output_end(4) + feed_count(4) + n_gates(4)
const HEADER_LEN: usize = 25;

const TYPE_XOR: u8 = 0;
const TYPE_AND: u8 = 1;
const TYPE_INV: u8 = 2;
const TYPE_ID: u8 = 3;

/// Serializes a circuit into a compact binary format.
pub fn serialize(circuit: &Circuit) -> Vec<u8> {
    let n_gates = circuit.gates.len();
    let mut out = Vec::with_capacity(HEADER_LEN + n_gates * 13);

    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&(circuit.inputs.end as u32).to_le_bytes());
    out.extend_from_slice(&(circuit.outputs.start as u32).to_le_bytes());
    out.extend_from_slice(&(circuit.outputs.end as u32).to_le_bytes());
    out.extend_from_slice(&(circuit.feed_count as u32).to_le_bytes());
    out.extend_from_slice(&(n_gates as u32).to_le_bytes());

    for gate in &circuit.gates {
        match gate {
            Gate::Xor { x, y, z } => {
                out.push(TYPE_XOR);
                out.extend_from_slice(&(x.id as u32).to_le_bytes());
                out.extend_from_slice(&(y.id as u32).to_le_bytes());
                out.extend_from_slice(&(z.id as u32).to_le_bytes());
            }
            Gate::And { x, y, z } => {
                out.push(TYPE_AND);
                out.extend_from_slice(&(x.id as u32).to_le_bytes());
                out.extend_from_slice(&(y.id as u32).to_le_bytes());
                out.extend_from_slice(&(z.id as u32).to_le_bytes());
            }
            Gate::Inv { x, z } => {
                out.push(TYPE_INV);
                out.extend_from_slice(&(x.id as u32).to_le_bytes());
                out.extend_from_slice(&(z.id as u32).to_le_bytes());
            }
            Gate::Id { x, z } => {
                out.push(TYPE_ID);
                out.extend_from_slice(&(x.id as u32).to_le_bytes());
                out.extend_from_slice(&(z.id as u32).to_le_bytes());
            }
        }
    }

    out
}

/// Deserializes a circuit from the compact binary format.
pub fn deserialize(bytes: &[u8]) -> Result<Circuit, String> {
    if bytes.len() < HEADER_LEN {
        return Err(format!("too short: {} bytes", bytes.len()));
    }
    if &bytes[0..4] != MAGIC {
        return Err("invalid magic".to_string());
    }
    if bytes[4] != VERSION {
        return Err(format!("unsupported version: {}", bytes[4]));
    }

    let read_u32 = |bytes: &[u8], pos: usize| -> usize {
        u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize
    };

    let n_inputs = read_u32(bytes, 5);
    let output_start = read_u32(bytes, 9);
    let output_end = read_u32(bytes, 13);
    let feed_count = read_u32(bytes, 17);
    let n_gates = read_u32(bytes, 21);

    let mut gates = Vec::with_capacity(n_gates);
    let mut and_count = 0usize;
    let mut xor_count = 0usize;
    let mut pos = HEADER_LEN;

    for _ in 0..n_gates {
        if pos >= bytes.len() {
            return Err("unexpected end of data".to_string());
        }
        let gate_type = bytes[pos];
        pos += 1;
        let x_id = read_u32(bytes, pos);
        pos += 4;

        match gate_type {
            TYPE_XOR => {
                let y_id = read_u32(bytes, pos); pos += 4;
                let z_id = read_u32(bytes, pos); pos += 4;
                xor_count += 1;
                gates.push(Gate::Xor {
                    x: Node::<Sink>::new(x_id),
                    y: Node::<Sink>::new(y_id),
                    z: Node::<Feed>::new(z_id),
                });
            }
            TYPE_AND => {
                let y_id = read_u32(bytes, pos); pos += 4;
                let z_id = read_u32(bytes, pos); pos += 4;
                and_count += 1;
                gates.push(Gate::And {
                    x: Node::<Sink>::new(x_id),
                    y: Node::<Sink>::new(y_id),
                    z: Node::<Feed>::new(z_id),
                });
            }
            TYPE_INV => {
                let z_id = read_u32(bytes, pos); pos += 4;
                gates.push(Gate::Inv {
                    x: Node::<Sink>::new(x_id),
                    z: Node::<Feed>::new(z_id),
                });
            }
            TYPE_ID => {
                let z_id = read_u32(bytes, pos); pos += 4;
                gates.push(Gate::Id {
                    x: Node::<Sink>::new(x_id),
                    z: Node::<Feed>::new(z_id),
                });
            }
            other => return Err(format!("unknown gate type: {other}")),
        }
    }

    Ok(Circuit {
        inputs: 0..n_inputs,
        outputs: output_start..output_end,
        gates,
        feed_count,
        and_count,
        xor_count,
    })
}
