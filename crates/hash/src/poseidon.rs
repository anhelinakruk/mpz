use mpz_circuits::{POSEIDON2_ABSORB, POSEIDON2_PERMUTE};
use mpz_core::bitvec::BitVec;
use mpz_vm_core::{
    Call, CallableExt, Vm, VmError,
    memory::{
        Array, FromRaw, MemoryExt, Slice, ToRaw, Vector, ViewExt,
        binary::{Binary, U32, U8},
    },
};

pub const RATE: usize = 8;

const STATE_SIZE: usize = 16;

const IV: [u32; STATE_SIZE] = [0u32; STATE_SIZE];

#[derive(Debug, Clone)]
pub struct Poseidon2 {
    state: Option<Array<U32, STATE_SIZE>>,
    byte_buf: Vec<Slice>,
    permutation_count: usize,
}

impl Default for Poseidon2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Poseidon2 {
    pub fn new() -> Self {
        Self {
            state: None,
            byte_buf: Vec::new(),
            permutation_count: 0,
        }
    }

    pub fn permutation_count(&self) -> usize {
        self.permutation_count
    }

    pub fn update(
        &mut self,
        vm: &mut dyn Vm<Binary>,
        input: &Vector<U8>,
    ) -> Result<(), Poseidon2Error> {
        let raw = input.to_raw();
        let n_bytes = raw.len() / 8;
        for i in 0..n_bytes {
            let (_, tail) = raw.split_at(i * 8);
            let (byte_slice, _) = tail.split_at(8);
            self.byte_buf.push(byte_slice);
        }
        while self.byte_buf.len() >= RATE {
            let chunk: [Slice; RATE] = self.byte_buf[..RATE].try_into().unwrap();
            self.byte_buf.drain(..RATE);
            self.absorb_block(vm, chunk)?;
        }
        Ok(())
    }

    pub fn finalize(mut self, vm: &mut dyn Vm<Binary>) -> Result<Array<U8, 32>, Poseidon2Error> {
        if self.state.is_none() || !self.byte_buf.is_empty() {
            let padding = RATE - self.byte_buf.len();
            if padding > 0 {
                let zeros = vm.alloc_raw(padding * 8)?;
                vm.mark_public_raw(zeros)?;
                vm.assign_raw(zeros, BitVec::repeat(false, padding * 8))?;
                vm.commit_raw(zeros)?;
                for i in 0..padding {
                    let (_, tail) = zeros.split_at(i * 8);
                    let (zero_byte, _) = tail.split_at(8);
                    self.byte_buf.push(zero_byte);
                }
            }
            let chunk: [Slice; RATE] = self
                .byte_buf
                .drain(..RATE)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            self.absorb_block(vm, chunk)?;
        }

        let rate_out = self
            .state
            .expect("state was initialized")
            .get::<RATE>(0)
            .expect("state has 16 elements");
        Ok(<Array<U8, 32> as FromRaw<Binary>>::from_raw(rate_out.to_raw()))
    }

    fn absorb_block(
        &mut self,
        vm: &mut dyn Vm<Binary>,
        bytes: [Slice; RATE],
    ) -> Result<(), Poseidon2Error> {
        let state = self.get_or_init_state(vm)?;
        let rate = state.get::<RATE>(0).expect("state has 16 elements");
        let capacity = state.get::<RATE>(RATE).expect("state has 16 elements");

        let zeros = vm.alloc_raw(RATE * 24)?;
        vm.mark_public_raw(zeros)?;
        vm.assign_raw(zeros, BitVec::repeat(false, RATE * 24))?;
        vm.commit_raw(zeros)?;

        let mut call_builder = Call::builder(POSEIDON2_ABSORB.clone()).arg(rate);
        for (i, byte) in bytes.into_iter().enumerate() {
            let (_, tail) = zeros.split_at(i * 24);
            let (zero_chunk, _) = tail.split_at(24);
            call_builder = call_builder.arg(byte).arg(zero_chunk);
        }
        let new_rate: Array<U32, RATE> = vm.call(
            call_builder
                .build()
                .expect("poseidon2 absorb circuit should have 512 bit input"),
        )?;

        let call = Call::builder(POSEIDON2_PERMUTE.clone())
            .arg(new_rate)
            .arg(capacity)
            .build()
            .expect("poseidon2 permute circuit should have 512 bit input");

        self.state = Some(vm.call(call)?);
        self.permutation_count += 1;
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

#[derive(Debug, thiserror::Error)]
#[error("poseidon2 error: {0}")]
pub struct Poseidon2Error(#[from] VmError);

#[cfg(test)]
mod tests {
    use mpz_circuits::circuits::poseidon2::permute;
use mpz_common::context::test_st_context;
    use mpz_ideal_vm::IdealVm;
    use mpz_vm_core::prelude::*;

    use super::*;

    async fn hash(input: &[u8]) -> [u8; 32] {
        let (vm_0, vm_1) = (IdealVm::default(), IdealVm::default());
        let (mut ctx_a, mut ctx_b) = test_st_context(8);

        let (a, b) = tokio::join!(
            async {
                let mut vm = vm_0;
                let mut hasher = Poseidon2::new();
                if !input.is_empty() {
                    let data = vm.alloc_vec::<U8>(input.len()).unwrap();
                    vm.mark_public(data).unwrap();
                    vm.assign(data, input.to_vec()).unwrap();
                    vm.commit(data).unwrap();
                    hasher.update(&mut vm, &data).unwrap();
                }
                let out = hasher.finalize(&mut vm).unwrap();
                let mut out = vm.decode(out).unwrap();
                vm.execute_all(&mut ctx_a).await.unwrap();
                out.try_recv().unwrap().unwrap()
            },
            async {
                let mut vm = vm_1;
                let mut hasher = Poseidon2::new();
                if !input.is_empty() {
                    let data = vm.alloc_vec::<U8>(input.len()).unwrap();
                    vm.mark_public(data).unwrap();
                    vm.assign(data, input.to_vec()).unwrap();
                    vm.commit(data).unwrap();
                    hasher.update(&mut vm, &data).unwrap();
                }
                let out = hasher.finalize(&mut vm).unwrap();
                let mut out = vm.decode(out).unwrap();
                vm.execute_all(&mut ctx_b).await.unwrap();
                out.try_recv().unwrap().unwrap()
            }
        );

        assert_eq!(a, b);
        a
    }

    fn native_hash(input: &[u8]) -> [u8; 32] {

        const P: u32 = 0x7FFF_FFFF;

        let circ = permute();

        let permute_state = |state: [u32; 16]| -> [u32; 16] {
            let bits: Vec<bool> = state
                .iter()
                .flat_map(|&x| (0..31).map(move |b| x & (1 << b) != 0))
                .collect();
            let out: Vec<bool> = circ
                .evaluate(bits.into_iter())
                .unwrap()
                .into_iter()
                .collect();
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
        let input_u32: Vec<u32> = input.iter().map(|&b| b as u32).collect();

        if input_u32.is_empty() {
            state = permute_state(state);
        } else {
            for chunk in input_u32.chunks(RATE) {
                let mut block = [0u32; RATE];
                block[..chunk.len()].copy_from_slice(chunk);
                for i in 0..RATE {
                    state[i] = add(state[i], block[i]);
                }
                state = permute_state(state);
            }
        }

        let mut out = [0u8; 32];
        for (i, &word) in state[..RATE].iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    #[rstest::rstest]
    #[case::empty(&[])]
    #[case::less_than_block(&[1u8, 2, 3, 4])]
    #[case::exactly_one_block(&[0u8, 1, 2, 3, 4, 5, 6, 7])]
    #[case::multiple_blocks(&[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])]
    #[tokio::test]
    async fn test_poseidon2_matches_native(#[case] input: &[u8]) {
        let out = hash(input).await;
        assert_eq!(out, native_hash(input));
    }

}