//! GPU port of TFHE-rs core_crypto's extract_bits loop. Each round operates on
//! a whole public batch; only the rounds depend on one another. All key switches,
//! PBS calls and ciphertext linear operations use upstream CUDA APIs.
use tfhe::core_crypto::gpu::algorithms::*;
use tfhe::core_crypto::gpu::entities::{
    glwe_ciphertext_list::CudaGlweCiphertextList,
    lwe_bootstrap_key::CudaLweBootstrapKey,
    lwe_ciphertext_list::CudaLweCiphertextList,
    lwe_keyswitch_key::CudaLweKeyswitchKey,
};
use tfhe::core_crypto::gpu::vec::{CudaVec, GpuIndex};
use tfhe::core_crypto::gpu::CudaStreams;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::ciphertext::{Degree, NoiseLevel};
use tfhe::FheUint64;
use super::{assemble_phase, GPU_PARAMS, PHASE_BITS, SOURCE_DIM, SOURCE_T};

pub struct GpuBridge {
    source_to_big: CudaLweKeyswitchKey<u64>,
    big_to_small: CudaLweKeyswitchKey<u64>,
    precise_bsk: CudaLweBootstrapKey,
    output_bsk: CudaLweBootstrapKey,
    small_dimension: LweDimension,
}

fn upload(words: &[u64], streams: &CudaStreams) -> CudaVec<u64> {
    let mut out = CudaVec::new(words.len(), streams, 0);
    unsafe { out.copy_from_cpu_async(words, streams, 0); }
    streams.synchronize();
    out
}

fn lut(value: u64, streams: &CudaStreams) -> CudaGlweCiphertextList<u64> {
    let mut accumulator = GlweCiphertext::new(0u64, GPU_PARAMS.glwe_dimension.to_glwe_size(),
        GPU_PARAMS.polynomial_size, CiphertextModulus::new_native());
    accumulator.get_mut_body().as_mut().fill(value);
    CudaGlweCiphertextList::from_glwe_ciphertext(&accumulator, streams)
}

impl GpuBridge {
    // CLIENT SETUP: upload evaluation keys only; no GPU secret-key buffers.
    pub fn new(source: &LweKeyswitchKeyOwned<u64>, small: &LweKeyswitchKeyOwned<u64>,
        precise: &LweBootstrapKeyOwned<u64>, output: &LweBootstrapKeyOwned<u64>) -> Self {
        let streams = CudaStreams::new_single_gpu(GpuIndex::new(0));
        Self {
            source_to_big: CudaLweKeyswitchKey::from_lwe_keyswitch_key(source, &streams),
            big_to_small: CudaLweKeyswitchKey::from_lwe_keyswitch_key(small, &streams),
            precise_bsk: CudaLweBootstrapKey::from_lwe_bootstrap_key(precise, None, &streams),
            output_bsk: CudaLweBootstrapKey::from_lwe_bootstrap_key(output, None, &streams),
            small_dimension: small.output_key_lwe_dimension(),
        }
    }

    // SERVER GPU: batch all selected distances through 24 dependent rounds.
    // Each native keyswitch/PBS kernel handles the whole batch concurrently.
    pub fn phase_integers(&self, words: &[u64]) -> Vec<FheUint64> {
        let count = words.len() / (SOURCE_DIM+1);
        let ct_count = LweCiphertextCount(count);
        let native = CiphertextModulus::new_native();
        let big_dimension = GPU_PARAMS.glwe_dimension.to_equivalent_lwe_dimension(GPU_PARAMS.polynomial_size);
        let big_size = big_dimension.to_lwe_size();
        let streams = CudaStreams::new_single_gpu(GpuIndex::new(0));
        let host = LweCiphertextList::from_container(words, LweSize(SOURCE_DIM+1), native);
        let input = CudaLweCiphertextList::from_lwe_ciphertext_list(&host, &streams);
        let indexes = upload(&(0..count as u64).collect::<Vec<_>>(), &streams);
        let lut_indexes = upload(&vec![0; count], &streams);
        let mut working = CudaLweCiphertextList::new(big_dimension, ct_count, native, &streams);
        cuda_keyswitch_lwe_ciphertext(&self.source_to_big, &input, &mut working,
            &indexes, &indexes, true, &streams, false);
        let center = upload(&vec![((1u128 << 63) / SOURCE_T as u128) as u64; count], &streams);
        cuda_lwe_ciphertext_plaintext_add_assign(&mut working, &center, &streams);

        let mut scaled = CudaLweCiphertextList::new(big_dimension, ct_count, native, &streams);
        let mut small = CudaLweCiphertextList::new(self.small_dimension, ct_count, native, &streams);
        let mut correction = CudaLweCiphertextList::new(big_dimension, ct_count, native, &streams);
        let mut output_bit = CudaLweCiphertextList::new(big_dimension, ct_count, native, &streams);
        let mut digit = CudaLweCiphertextList::new(big_dimension, ct_count, native, &streams);
        let quarter = upload(&vec![1u64 << 62; count], &streams);
        let delta_half = upload(&vec![1u64 << 58; count], &streams);
        let two = upload(&vec![2u64; count], &streams);
        let output_lut = lut(0u64.wrapping_sub(1u64 << 58), &streams);
        let mut blocks = vec![Vec::with_capacity(32); count];
        for bit in 0..PHASE_BITS {
            // Identical to upstream extract_bits: lift the current bit into
            // the torus sign, key switch, then center at 1/4 or 3/4.
            let multiplier = upload(&vec![1u64 << (PHASE_BITS-1-bit); count], &streams);
            cuda_lwe_ciphertext_cleartext_mul(&mut scaled, &working, &multiplier, &streams);
            cuda_keyswitch_lwe_ciphertext(&self.big_to_small, &scaled, &mut small,
                &indexes, &indexes, true, &streams, false);
            cuda_lwe_ciphertext_plaintext_add_assign(&mut small, &quarter, &streams);

            // Native radix bit: sign PBS -> bit * 2^59 under the GPU's big key.
            cuda_programmable_bootstrap_lwe_ciphertext(&small, &mut output_bit, &output_lut,
                &lut_indexes, &indexes, &indexes, &self.output_bsk, &streams);
            cuda_lwe_ciphertext_plaintext_add_assign(&mut output_bit, &delta_half, &streams);
            if bit % 2 == 0 {
                std::mem::swap(&mut digit, &mut output_bit);
            } else {
                cuda_lwe_ciphertext_cleartext_mul_assign(&mut output_bit, &two, &streams);
                cuda_lwe_ciphertext_add_assign(&mut digit, &output_bit, &streams);
                // Download ciphertexts only to assemble the public Rust radix
                // container. Arithmetic and PBS have already completed on GPU.
                let host_digit = digit.to_lwe_ciphertext_list(&streams);
                for (row, ct) in blocks.iter_mut().zip(host_digit.iter()) {
                    row.push(tfhe::shortint::Ciphertext::new(
                        LweCiphertext::from_container(ct.as_ref().to_vec(), native),
                        Degree::new(3), NoiseLevel::NOMINAL * 3,
                        GPU_PARAMS.message_modulus, GPU_PARAMS.carry_modulus,
                        GPU_PARAMS.to_shortint_conformance_param().atomic_pattern));
                }
            }
            if bit+1 < PHASE_BITS {
                // Recover this bit at its original scale, then subtract it.
                // The next bit of the SAME distance depends on this operation;
                // different distances remain independent throughout the batch.
                let alpha = 1u64 << (64-PHASE_BITS-1+bit);
                let precise_lut = lut(0u64.wrapping_sub(alpha), &streams);
                cuda_programmable_bootstrap_lwe_ciphertext(&small, &mut correction, &precise_lut,
                    &lut_indexes, &indexes, &indexes, &self.precise_bsk, &streams);
                let offset = upload(&vec![alpha; count], &streams);
                cuda_lwe_ciphertext_plaintext_add_assign(&mut correction, &offset, &streams);
                cuda_lwe_ciphertext_negate_assign(&mut correction, &streams);
                cuda_lwe_ciphertext_add_assign(&mut working, &correction, &streams);
            }
            if (bit+1) % 6 == 0 {
                println!("GPU_BIT_EXTRACTION round={}/{} batch={count}", bit+1, PHASE_BITS);
            }
        }
        blocks.into_iter().map(|row| assemble_phase(row, big_size)).collect()
    }
}
