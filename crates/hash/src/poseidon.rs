//! Poseidon2 hash function over M31.

use mpz_circuits::{POSEIDON2_ABSORB, POSEIDON2_PERMUTE};
use mpz_vm_core::{
    Call, CallableExt, Vm, VmError,
    memory::{
        Array, MemoryExt, ViewExt,
        binary::{Binary, U32},
    },
};

/// Number of M31 elements in the rate portion of the state.
pub const RATE: usize = 8;

/// Number of M31 elements in the full state.
const STATE_SIZE: usize = 16;

/// Initial value of the sponge state.
const IV: [u32; STATE_SIZE] = [0u32; STATE_SIZE];

/// Poseidon2 hasher over M31 using a sponge construction.
///
/// Absorbs M31 field elements (values in `[0, 2^31 - 1)`) and produces
/// a [`RATE`]-element digest.
#[derive(Debug, Clone)]
pub struct Poseidon2 {
    state: Option<Array<U32, STATE_SIZE>>,
    /// Buffered M31 elements not yet absorbed.
    buf: Vec<u32>,
}

impl Default for Poseidon2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Poseidon2 {
    /// Creates a new hasher.
    pub fn new() -> Self {
        Self {
            state: None,
            buf: Vec::new(),
        }
    }

    /// Creates a new hasher initialized with the IV in the VM.
    pub fn new_with_init(vm: &mut dyn Vm<Binary>) -> Result<Self, Poseidon2Error> {
        let mut hasher = Self::new();
        hasher.get_or_init_state(vm)?;
        Ok(hasher)
    }

    /// Absorbs M31 elements into the hasher.
    pub fn update(
        &mut self,
        vm: &mut dyn Vm<Binary>,
        input: &[u32],
    ) -> Result<(), Poseidon2Error> {
        self.buf.extend_from_slice(input);
        while self.buf.len() >= RATE {
            let chunk: [u32; RATE] = self.buf[..RATE].try_into().unwrap();
            self.buf.drain(..RATE);
            self.absorb_block(vm, chunk)?;
        }
        Ok(())
    }

    /// Finalizes the hash and returns the first `RATE` M31 elements of the state.
    pub fn finalize(mut self, vm: &mut dyn Vm<Binary>) -> Result<Array<U32, RATE>, Poseidon2Error> {
        // Absorb a zero-padded block when:
        //   - there is a partial (non-empty) buffer to flush, OR
        //   - no block was absorbed yet (empty input → one all-zero block).
        if self.state.is_none() || !self.buf.is_empty() {
            let mut padded = [0u32; RATE];
            for (i, &x) in self.buf.iter().enumerate() {
                padded[i] = x;
            }
            self.absorb_block(vm, padded)?;
        }

        Ok(self
            .state
            .expect("state was initialized")
            .get::<RATE>(0)
            .expect("state has 16 elements"))
    }

    /// Absorbs a single rate-sized block into the state.
    fn absorb_block(
        &mut self,
        vm: &mut dyn Vm<Binary>,
        block: [u32; RATE],
    ) -> Result<(), Poseidon2Error> {
        let state = self.get_or_init_state(vm)?;

        // Split state into rate and capacity.
        let rate = state.get::<RATE>(0).expect("state has 16 elements");
        let capacity = state.get::<RATE>(RATE).expect("state has 16 elements");

        // Add block into rate using M31 field addition.
        let block_arr: Array<U32, RATE> = vm.alloc()?;
        vm.mark_public(block_arr)?;
        vm.assign(block_arr, block)?;
        vm.commit(block_arr)?;

        let call = Call::builder(POSEIDON2_ABSORB.clone())
            .arg(rate)
            .arg(block_arr)
            .build()
            .expect("poseidon2 absorb circuit should have 512 bit input");
        let new_rate: Array<U32, RATE> = vm.call(call)?;

        // Permute the full state.
        let call = Call::builder(POSEIDON2_PERMUTE.clone())
            .arg(new_rate)
            .arg(capacity)
            .build()
            .expect("poseidon2 permute circuit should have 512 bit input");

        self.state = Some(vm.call(call)?);
        Ok(())
    }

    fn get_or_init_state(
        &mut self,
        vm: &mut dyn Vm<Binary>,
    ) -> Result<Array<U32, STATE_SIZE>, Poseidon2Error> {
        if let Some(state) = self.state {
            Ok(state)
        } else {
            let state: Array<U32, STATE_SIZE> = vm.alloc()?;
            vm.mark_public(state)?;
            vm.assign(state, IV)?;
            vm.commit(state)?;
            self.state = Some(state);
            Ok(state)
        }
    }
}

/// Error for [`Poseidon2`].
#[derive(Debug, thiserror::Error)]
#[error("poseidon2 error: {0}")]
pub struct Poseidon2Error(#[from] VmError);

#[cfg(test)]
mod tests {
    use mpz_common::context::test_st_context;
    use mpz_ideal_vm::IdealVm;
    use mpz_vm_core::prelude::*;

    use super::*;

    /// Reference sponge implementation using the 31-bit `permute()` circuit.
    fn native_hash(input: &[u32]) -> [u32; RATE] {
        use mpz_circuits::circuits::poseidon::permute;

        const P: u32 = 0x7FFF_FFFF;

        let circ = permute();

        let permute_state = |state: [u32; 16]| -> [u32; 16] {
            let bits: Vec<bool> = state
                .iter()
                .flat_map(|&x| (0..31).map(move |b| x & (1 << b) != 0))
                .collect();
            let out: Vec<bool> = circ.evaluate(bits.into_iter()).unwrap().into_iter().collect();
            std::array::from_fn(|i| {
                out[i * 31..(i + 1) * 31]
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (b, &bit)| acc | ((bit as u32) << b))
            })
        };

        let add = |a: u32, b: u32| -> u32 {
            let s = a + b;
            if s >= P { s - P } else { s }
        };

        let mut state = [0u32; 16];

        if input.is_empty() {
            state = permute_state(state);
        } else {
            for chunk in input.chunks(RATE) {
                let mut block = [0u32; RATE];
                block[..chunk.len()].copy_from_slice(chunk);
                for i in 0..RATE {
                    state[i] = add(state[i], block[i]);
                }
                state = permute_state(state);
            }
        }

        state[..RATE].try_into().unwrap()
    }

    async fn hash(input: &[u32]) -> [u32; RATE] {
        let (vm_0, vm_1) = (IdealVm::default(), IdealVm::default());
        let (mut ctx_a, mut ctx_b) = test_st_context(8);

        let (a, b) = tokio::join!(
            async {
                let mut vm = vm_0;
                let mut hasher = Poseidon2::new();
                hasher.update(&mut vm, input).unwrap();
                let out = hasher.finalize(&mut vm).unwrap();
                let mut out = vm.decode(out).unwrap();
                vm.execute_all(&mut ctx_a).await.unwrap();
                out.try_recv().unwrap().unwrap()
            },
            async {
                let mut vm = vm_1;
                let mut hasher = Poseidon2::new();
                hasher.update(&mut vm, input).unwrap();
                let out = hasher.finalize(&mut vm).unwrap();
                let mut out = vm.decode(out).unwrap();
                vm.execute_all(&mut ctx_b).await.unwrap();
                out.try_recv().unwrap().unwrap()
            }
        );

        assert_eq!(a, b, "both VMs should produce the same hash");
        a
    }

    #[tokio::test]
    async fn test_poseidon2_less_than_block() {
        let input: Vec<u32> = (0..4).collect();
        let out = hash(&input).await;
        assert_eq!(out, native_hash(&input));
    }

    #[tokio::test]
    async fn test_poseidon2_exactly_one_block() {
        let input: Vec<u32> = (0..8).collect();
        let out = hash(&input).await;
        assert_eq!(out, native_hash(&input));
    }

    #[tokio::test]
    async fn test_poseidon2_multiple_blocks() {
        let input: Vec<u32> = (0..20).map(|i| i % 0x7FFF_FFFF).collect();
        let out = hash(&input).await;
        assert_eq!(out, native_hash(&input));
    }

    // Test vectors: expected digest is output[0].
    #[rstest::rstest]
    #[case::empty(vec![], 0x4685cf30)]
    #[case::single_zero(vec![0], 0x4685cf30)]
    #[case::single_one(vec![1], 0x3278aa1c)]
    #[case::one_block(vec![0,1,2,3,4,5,6,7], 0x28c4b3a1)]
    #[case::two_blocks((0..16u32).collect(), 0x7502874b)]
    #[tokio::test]
    async fn test_poseidon2_vectors(#[case] input: Vec<u32>, #[case] expected: u32) {
        let out = hash(&input).await;
        assert_eq!(out[0], expected, "native also: {:08x}", native_hash(&input)[0]);
    }
}
