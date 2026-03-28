//! Blake2s hash function.

use std::{
    array::from_fn,
    sync::{Arc, LazyLock},
};

use mpz_circuits::{Circuit, CircuitBuilder, BLAKE2S_COMPRESS};
use mpz_core::bitvec::BitVec;
use mpz_vm_core::{
    Call, CallableExt, Vm, VmError,
    memory::{
        Array, MemoryExt, Slice, ToRaw, Vector, ViewExt,
        binary::{Binary, U8, U32},
    },
};

// Serializes the state as bytes in little-endian order.
// Blake2s uses little-endian byte order, so each 32-bit word is output
// byte by byte from LSB to MSB. In LSB0 bit representation used by the
// circuit, chunks of 8 bits in natural order are already little-endian.
static SERIALIZE_STATE: LazyLock<Arc<Circuit>> = LazyLock::new(|| {
    let mut builder = CircuitBuilder::new();

    for _ in 0..8 {
        let word: [_; 32] = from_fn(|_| builder.add_input());
        for byte in word.chunks_exact(8) {
            for &bit in byte {
                let out = builder.add_id_gate(bit);
                builder.add_output(out);
            }
        }
    }

    Arc::new(builder.build().unwrap())
});

// Blake2s initialization vector (same as SHA-256 IV).
// Ref: https://www.rfc-editor.org/rfc/rfc7693#section-2.6
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

// Initial chaining value for Blake2s-256 with no key and 32-byte output.
//
// h[0] = IV[0] ^ parameter_block_word_0
// parameter_block_word_0 = digest_length(1B) | key_length(1B) | fanout(1B) | depth(1B)
//                        = 0x20           | 0x00            | 0x01        | 0x01
//                        = 0x01010020  (little-endian u32)
const H_INIT: [u32; 8] = [
    IV[0] ^ 0x01010020,
    IV[1], IV[2], IV[3], IV[4], IV[5], IV[6], IV[7],
];

/// Block size in bits.
const BLOCK_SIZE: usize = 512;

#[derive(Debug, Clone)]
struct Block {
    data: Vec<Slice>,
    len: usize, // in bits
}

impl Default for Block {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
        }
    }
}

// Wraps the current chaining value (hash state h[0..8]) in the VM.
#[derive(Debug, Copy, Clone)]
struct ChainingValue(Array<U32, 8>);

impl ChainingValue {
    fn new(vm: &mut dyn Vm<Binary>, value: [u32; 8]) -> Result<Self, Blake2sError> {
        let state = vm.alloc()?;
        vm.mark_public(state)?;
        vm.assign(state, value)?;
        vm.commit(state)?;
        Ok(Self(state))
    }

    fn set(&mut self, state: Array<U32, 8>) {
        self.0 = state;
    }
}

/// Blake2s-256 hasher.
///
/// Implements sequential (non-keyed) Blake2s with 32-byte output,
/// operating through a VM for use in MPC protocols.
#[derive(Debug, Clone)]
pub struct Blake2s {
    chaining_value: ChainingValue,
    blocks: Vec<Block>,
    processed_bytes: usize,
}

impl Blake2s {
    /// Creates a new Blake2s hasher, initializing the chaining value in the VM.
    pub fn new(vm: &mut dyn Vm<Binary>) -> Result<Self, Blake2sError> {
        let chaining_value = ChainingValue::new(vm, H_INIT)?;
        Ok(Self {
            chaining_value,
            blocks: Vec::new(),
            processed_bytes: 0,
        })
    }

    /// Adds data to the hash state.
    ///
    /// Eagerly compresses full blocks while keeping at least one block buffered,
    /// since we cannot know if the current last block is the final one.
    pub fn update(
        &mut self,
        vm: &mut dyn Vm<Binary>,
        data: &Vector<U8>,
    ) -> Result<(), Blake2sError> {
        self.buffer_slice(data.to_raw());
        self.compress_full_blocks(vm)?;
        Ok(())
    }

    /// Finalizes the hash and returns the 32-byte output.
    pub fn finalize(&mut self, vm: &mut dyn Vm<Binary>) -> Result<Array<U8, 32>, Blake2sError> {
        // Ensure there is at least one block for the final compression.
        if self.blocks.is_empty() {
            self.blocks.push(Block::default());
        }

        // Compress all remaining blocks except the last.
        while self.blocks.len() > 1 {
            let block = self.blocks.remove(0);
            self.processed_bytes += block.len / 8;
            let t = self.processed_bytes as u64;
            self.compress_block(vm, block, t, false)?;
        }

        // Total input bytes (for the byte counter t, excluding zero padding).
        let total_bytes = self.processed_bytes + self.blocks[0].len / 8;

        // Compress the last block with the finalization flag set.
        let last = self.blocks.remove(0);
        self.compress_block(vm, last, total_bytes as u64, true)?;

        // Serialize state to bytes (little-endian).
        let call = Call::builder(SERIALIZE_STATE.clone())
            .arg(self.chaining_value.0)
            .build()
            .expect("serialize circuit should have 256 bit input");

        Ok(vm.call(call)?)
    }

    // Buffers incoming data into blocks.
    fn buffer_slice(&mut self, mut data: Slice) {
        if data.len() == 0 {
            return;
        }

        // Fill the last incomplete block first.
        if let Some(block) = self.blocks.last_mut()
            && block.len < BLOCK_SIZE
        {
            let diff = BLOCK_SIZE - block.len;
            let (left, right) = data.split_at(diff.min(data.len()));
            block.data.push(left);
            block.len += left.len();
            data = right;
        }

        // Partition the rest into full or partial blocks.
        while data.len() > 0 {
            let (left, right) = data.split_at(BLOCK_SIZE.min(data.len()));
            self.blocks.push(Block {
                data: vec![left],
                len: left.len(),
            });
            data = right;
        }
    }

    // Compresses all full blocks, keeping at least one block buffered.
    fn compress_full_blocks(&mut self, vm: &mut dyn Vm<Binary>) -> Result<(), Blake2sError> {
        while self.blocks.len() > 1 && self.blocks[0].len == BLOCK_SIZE {
            let block = self.blocks.remove(0);
            self.processed_bytes += block.len / 8;
            let t = self.processed_bytes as u64;
            self.compress_block(vm, block, t, false)?;
        }
        Ok(())
    }

    // Compresses a single block through the VM.
    //
    // Circuit inputs: h (256 bits) | m (512 bits) | v_upper (256 bits)
    fn compress_block(
        &mut self,
        vm: &mut dyn Vm<Binary>,
        block: Block,
        t: u64,
        is_last: bool,
    ) -> Result<(), Blake2sError> {
        let v_upper = Self::make_v_upper(vm, t, is_last)?;

        let mut builder = Call::builder(BLAKE2S_COMPRESS.clone())
            .arg(self.chaining_value.0);

        for slice in block.data {
            builder = builder.arg(slice);
        }

        // Zero-pad the last block to full block size.
        if block.len < BLOCK_SIZE {
            let pad_bits = BLOCK_SIZE - block.len;
            let padding = vm.alloc_raw(pad_bits)?;
            vm.mark_public_raw(padding)?;
            vm.assign_raw(padding, BitVec::repeat(false, pad_bits))?;
            vm.commit_raw(padding)?;
            builder = builder.arg(padding);
        }

        let call = builder
            .arg(v_upper)
            .build()
            .expect("blake2s compress circuit should have 1024 bit input");

        let new_state: Array<U32, 8> = vm.call(call)?;
        self.chaining_value.set(new_state);

        Ok(())
    }

    // Builds v_upper = [IV[0..4], IV[4]^t_low, IV[5]^t_high, IV[6]^f, IV[7]]
    // and allocates it as a public value in the VM.
    fn make_v_upper(
        vm: &mut dyn Vm<Binary>,
        t: u64,
        is_last: bool,
    ) -> Result<Array<U32, 8>, Blake2sError> {
        let v_upper_vals = [
            IV[0],
            IV[1],
            IV[2],
            IV[3],
            IV[4] ^ (t as u32),
            IV[5] ^ ((t >> 32) as u32),
            IV[6] ^ if is_last { 0xFFFF_FFFFu32 } else { 0u32 },
            IV[7],
        ];

        let v_upper: Array<U32, 8> = vm.alloc()?;
        vm.mark_public(v_upper)?;
        vm.assign(v_upper, v_upper_vals)?;
        vm.commit(v_upper)?;

        Ok(v_upper)
    }
}

/// Error for [`Blake2s`].
#[derive(Debug, thiserror::Error)]
#[error("blake2s error: {0}")]
pub struct Blake2sError(#[from] VmError);

#[cfg(test)]
mod tests {
    use blake2::{Blake2s256, Digest};
    use mpz_common::context::test_st_context;
    use mpz_ideal_vm::IdealVm;
    use mpz_vm_core::prelude::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::*;

    use super::*;

    async fn hash(data: &Vec<Vec<u8>>) -> [u8; 32] {
        let (vm_0, vm_1) = (IdealVm::default(), IdealVm::default());
        let (mut ctx_a, mut ctx_b) = test_st_context(8);

        let (a, b) = tokio::join!(
            async {
                let mut vm = vm_0;
                let mut hasher = Blake2s::new(&mut vm).unwrap();

                for chunk in data {
                    if chunk.is_empty() {
                        continue;
                    }
                    let data_ref = vm.alloc_vec::<U8>(chunk.len()).unwrap();
                    vm.mark_public(data_ref).unwrap();
                    vm.assign(data_ref, chunk.clone()).unwrap();
                    vm.commit(data_ref).unwrap();
                    hasher.update(&mut vm, &data_ref).unwrap();
                }

                let out = hasher.finalize(&mut vm).unwrap();
                let mut out = vm.decode(out).unwrap();
                vm.execute_all(&mut ctx_a).await.unwrap();
                out.try_recv().unwrap().unwrap()
            },
            async {
                let mut vm = vm_1;
                let mut hasher = Blake2s::new(&mut vm).unwrap();

                for chunk in data {
                    if chunk.is_empty() {
                        continue;
                    }
                    let data_ref = vm.alloc_vec::<U8>(chunk.len()).unwrap();
                    vm.mark_public(data_ref).unwrap();
                    vm.assign(data_ref, chunk.clone()).unwrap();
                    vm.commit(data_ref).unwrap();
                    hasher.update(&mut vm, &data_ref).unwrap();
                }

                let out = hasher.finalize(&mut vm).unwrap();
                let mut out = vm.decode(out).unwrap();
                vm.execute_all(&mut ctx_b).await.unwrap();
                out.try_recv().unwrap().unwrap()
            }
        );

        assert_eq!(a, b, "both VMs should produce the same hash");
        a
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::less_than_block(vec![1])]
    #[case::less_than_block_many(vec![1, 3, 9, 10])]
    #[case::exactly_one_block(vec![64])]
    #[case::multiple_blocks(vec![64, 64])]
    #[case::multiple_blocks_and_partial(vec![64, 64, 63])]
    #[tokio::test]
    async fn test_blake2s(#[case] lens: Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(0);
        let data: Vec<Vec<u8>> = lens
            .into_iter()
            .map(|len| (0..len).map(|_| rng.random::<u8>()).collect())
            .collect();

        let out = hash(&data).await;

        let mut ref_hasher = Blake2s256::new();
        for chunk in &data {
            ref_hasher.update(chunk);
        }
        let expected: [u8; 32] = ref_hasher.finalize().into();

        assert_eq!(out, expected, "hash should match reference Blake2s256");
    }

    // Official test vectors from RFC 7693 Appendix A.
    // https://www.rfc-editor.org/rfc/rfc7693#appendix-A
    #[tokio::test]
    async fn test_blake2s_official_vectors() {
        let vectors: &[(&[u8], &str)] = &[
            (
                b"abc",
                "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982",
            ),
            (
                b"",
                "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9",
            ),
        ];

        for (input, expected_hex) in vectors {
            let expected: Vec<u8> = (0..expected_hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&expected_hex[i..i + 2], 16).unwrap())
                .collect();

            let data = if input.is_empty() {
                vec![]
            } else {
                vec![input.to_vec()]
            };

            let result = hash(&data).await;

            assert_eq!(
                result,
                expected.as_slice(),
                "RFC vector mismatch for input {:?}",
                std::str::from_utf8(input).unwrap_or("<binary>")
            );
        }
    }
}
