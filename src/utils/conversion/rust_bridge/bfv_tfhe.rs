//! Experimental BFV torus LWE -> native TFHE-rs radix conversion.
//! Client setup is the only entry point that receives secret keys. BridgeEval
//! contains evaluation keys only. Conversion never decrypts or re-encrypts.
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;
use std::time::Instant;
use rayon::prelude::*;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::ciphertext::{Degree, NoiseLevel};
use tfhe::shortint::parameters::{
    parameters_wopbs_message_carry::LEGACY_WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS as BIT_PARAMS,
    PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 as GPU_PARAMS,
};
use tfhe::{ClientKey, CudaServerKey, FheUint32, FheUint64};
use tfhe::prelude::FheTrivialEncrypt;

mod gpu;

const SOURCE_DIM: usize = 4096;
const SOURCE_T: u64 = 8_380_417;
const PHASE_BITS: usize = 24;
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }

#[repr(C)]
pub struct Buffer { pointer: *mut u8, length: usize }

// SERVER MATERIAL ONLY: deliberately no ClientKey or LweSecretKey fields.
pub struct BridgeEval {
    source_to_big: LweKeyswitchKeyOwned<u64>,
    big_to_small: LweKeyswitchKeyOwned<u64>,
    precise_bsk: FourierLweBootstrapKeyOwned,
    output_bsk: FourierLweBootstrapKeyOwned,
    gpu_key: CudaServerKey,
    pool: rayon::ThreadPool,
    gpu: Option<gpu::GpuBridge>,
    gpu_pool: rayon::ThreadPool,
}

fn guarded(f: impl FnOnce() -> Result<(), String>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        other => {
            let message = match other {
                Ok(Err(s)) => s,
                Err(p) => p.downcast_ref::<String>().cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "Rust conversion panicked".into()),
                _ => unreachable!(),
            };
            ERROR.with(|e| *e.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
            1
        }
    }
}

#[no_mangle]
pub extern "C" fn bfv_tfhe_last_error() -> *const c_char {
    ERROR.with(|e| e.borrow().as_ptr())
}

fn fourier(key: LweBootstrapKeyOwned<u64>) -> FourierLweBootstrapKeyOwned {
    let mut out = FourierLweBootstrapKeyOwned::new(key.input_lwe_dimension(), key.glwe_size(),
        key.polynomial_size(), key.decomposition_base_log(), key.decomposition_level_count());
    par_convert_standard_lwe_bootstrap_key_to_fourier(&key, &mut out);
    out
}

#[no_mangle]
pub unsafe extern "C" fn bfv_tfhe_key_new(
    source_secret: *const u64, dimension: usize, modulus: u64,
    native_client: *const u8, native_client_len: usize, result: *mut *mut c_void,
) -> i32 {
    guarded(|| {
        if dimension != SOURCE_DIM || modulus != SOURCE_T || source_secret.is_null()
            || native_client.is_null() || result.is_null() {
            return Err("Conversion requires BFV N=4096, t=8380417 and valid client setup buffers".into());
        }
        *result = std::ptr::null_mut();
        // CLIENT SETUP: import a borrowed BFV secret and a temporary serialized
        // native TFHE client key. Neither is retained in the evaluation object.
        let source_words = slice::from_raw_parts(source_secret, dimension);
        if source_words.iter().any(|&v| v != 0 && v != 1 && v != u64::MAX) {
            return Err("BFV source key must be ternary".into());
        }
        let source_key = LweSecretKey::from_container(source_words);
        let client: ClientKey = bincode::deserialize(slice::from_raw_parts(native_client, native_client_len))
            .map_err(|e| e.to_string())?;
        if client.computation_parameters() != GPU_PARAMS.into() {
            return Err("Native TFHE client does not use the pinned GPU parameter set".into());
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().map_err(|e| e.to_string())?;
        let use_gpu = match std::env::var("PRIVATEIMPL_CONVERSION_BACKEND").as_deref() {
            Ok("cpu") => false,
            Ok("gpu") | Err(_) => true,
            _ => return Err("PRIVATEIMPL_CONVERSION_BACKEND must be gpu or cpu".into()),
        };
        let gpu_pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().map_err(|e| e.to_string())?;
        let (source_to_big, big_to_small, precise_bsk, output_bsk, gpu_key, gpu) = pool.install(|| {
            let short_client = client.clone().into_raw_parts().0.into_raw_parts();
            let big_key = short_client.encryption_key();
            let glwe_key = GlweSecretKey::from_container(big_key.as_ref(), GPU_PARAMS.polynomial_size);
            let mut seeder = new_seeder();
            let mut secret_rng = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
            let mut enc_rng = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder.as_mut());
            let small_key = allocate_and_generate_new_binary_lwe_secret_key(BIT_PARAMS.lwe_dimension, &mut secret_rng);
            let native = CiphertextModulus::new_native();
            // Fine decomposition preserves the 23-bit BFV message during the
            // cross-key switch. Target GLWE noise matches the native GPU key.
            let source_to_big = allocate_and_generate_new_lwe_keyswitch_key(&source_key, &big_key,
                DecompositionBaseLog(8), DecompositionLevelCount(7),
                GPU_PARAMS.glwe_noise_distribution, native, &mut enc_rng);
            let big_to_small = allocate_and_generate_new_lwe_keyswitch_key(&big_key, &small_key,
                BIT_PARAMS.ks_base_log, BIT_PARAMS.ks_level,
                BIT_PARAMS.lwe_noise_distribution, native, &mut enc_rng);
            // Bit extraction subtracts small encoded bits from a high precision
            // message. Use finer PBS decomposition than a normal shortint PBS.
            // This custom bridge is experimental; no 2^-128 failure bound is claimed.
            let precise_standard = par_allocate_and_generate_new_lwe_bootstrap_key(&small_key, &glwe_key,
                DecompositionBaseLog(4), DecompositionLevelCount(12),
                GPU_PARAMS.glwe_noise_distribution, native, &mut enc_rng);
            let output_standard = par_allocate_and_generate_new_lwe_bootstrap_key(&small_key, &glwe_key,
                DecompositionBaseLog(8), DecompositionLevelCount(4),
                GPU_PARAMS.glwe_noise_distribution, native, &mut enc_rng);
            let gpu = use_gpu.then(|| gpu::GpuBridge::new(&source_to_big, &big_to_small,
                &precise_standard, &output_standard));
            let precise_bsk = fourier(precise_standard);
            let output_bsk = fourier(output_standard);
            let gpu_key = client.generate_compressed_server_key().decompress_to_gpu();
            (source_to_big, big_to_small, precise_bsk, output_bsk, gpu_key, gpu)
        });
        *result = Box::into_raw(Box::new(BridgeEval {
            source_to_big, big_to_small, precise_bsk, output_bsk, gpu_key, pool, gpu, gpu_pool,
        })).cast();
        Ok(())
    })
}

impl BridgeEval {
    // SERVER CPU: public RNS extraction already supplied [a_0,...,a_N-1,b].
    // Key switch, then use TFHE-rs's own bit extraction and PBS primitives.
    fn phase_integer(&self, words: &[u64]) -> FheUint64 {
        let native = CiphertextModulus::new_native();
        let input = LweCiphertext::from_container(words, native);
        let big_size = self.source_to_big.output_key_lwe_dimension().to_lwe_size();
        let mut big = LweCiphertext::new(0u64, big_size, native);
        keyswitch_lwe_ciphertext(&self.source_to_big, &input, &mut big);
        // Center each BFV integer at (m+1/2)/t. Extract z approximately equal to
        // round(2^24*(m+1/2)/t). Then floor(z*t/2^24)=m if |z-z_exact|<1;
        // t<2^23 leaves a half-integer margin for the quantization error.
        let body = big.get_mut_body().data;
        *body = body.wrapping_add(((1u128 << 63) / SOURCE_T as u128) as u64);
        let mut bits = LweCiphertextList::new(0u64, self.big_to_small.output_key_lwe_dimension().to_lwe_size(),
            LweCiphertextCount(PHASE_BITS), native);
        let fft = Fft::new(self.precise_bsk.polynomial_size());
        let fft = fft.as_view();
        let mut buffers = ComputationBuffers::new();
        buffers.resize(extract_bits_from_lwe_ciphertext_mem_optimized_requirement::<u64>(
            big_size.to_lwe_dimension(), self.big_to_small.output_key_lwe_dimension(),
            self.precise_bsk.glwe_size(), self.precise_bsk.polynomial_size(), fft).unaligned_bytes_required());
        extract_bits_from_lwe_ciphertext_mem_optimized(&big, &mut bits, &self.precise_bsk,
            &self.big_to_small, DeltaLog(64-PHASE_BITS), ExtractedBitsCount(PHASE_BITS), fft, buffers.stack());

        let delta = 1u64 << 59; // Native message=4, carry=4, one padding bit.
        let mut acc = GlweCiphertext::new(0u64, self.output_bsk.glwe_size(), self.output_bsk.polynomial_size(), native);
        acc.get_mut_body().as_mut().fill(0u64.wrapping_sub(delta / 2));
        let mut blocks = Vec::with_capacity(32);
        // extract_bits returns MSB first; native radix blocks are LSB first.
        let extracted: Vec<_> = bits.iter().collect();
        for pair in (0..PHASE_BITS).step_by(2) {
            let mut digit = LweCiphertext::new(0u64, big_size, native);
            for bit in 0..2 {
                let mut shifted = LweCiphertext::from_container(extracted[PHASE_BITS-1-pair-bit].as_ref().to_vec(), native);
                let body = shifted.get_mut_body().data;
                *body = body.wrapping_add(1u64 << 62);
                let mut output = LweCiphertext::new(0u64, big_size, native);
                // The constant negacyclic LUT maps phases 1/4 and 3/4 to
                // -delta/2 and +delta/2. The body offset gives bit*delta.
                programmable_bootstrap_lwe_ciphertext(&shifted, &mut output, &acc, &self.output_bsk);
                let body = output.get_mut_body().data;
                *body = body.wrapping_add(delta / 2);
                for (d, &v) in digit.as_mut().iter_mut().zip(output.as_ref()) {
                    *d = d.wrapping_add(v.wrapping_mul(1 << bit));
                }
            }
            blocks.push(tfhe::shortint::Ciphertext::new(digit, Degree::new(3), NoiseLevel::NOMINAL * 3,
                GPU_PARAMS.message_modulus, GPU_PARAMS.carry_modulus,
                GPU_PARAMS.to_shortint_conformance_param().atomic_pattern));
        }
        assemble_phase(blocks, big_size)
    }
}

fn assemble_phase(mut blocks: Vec<tfhe::shortint::Ciphertext>, big_size: LweSize) -> FheUint64 {
    let native = CiphertextModulus::new_native();
    while blocks.len() < 32 {
        blocks.push(tfhe::shortint::Ciphertext::new(LweCiphertext::new(0u64, big_size, native),
        Degree::new(0), NoiseLevel::ZERO, GPU_PARAMS.message_modulus, GPU_PARAMS.carry_modulus,
        GPU_PARAMS.to_shortint_conformance_param().atomic_pattern));
    }
    FheUint64::from_raw_parts(tfhe::integer::RadixCiphertext::from(blocks),
        Default::default(), Default::default(), Default::default())
}

#[no_mangle]
pub unsafe extern "C" fn bfv_tfhe_convert_split(
    key: *const c_void, words: *const u64, count: usize, ids: *const u64,
    drop_bits: u32, id_bits: u32, result: *mut Buffer,
) -> i32 {
    guarded(|| {
        if key.is_null() || words.is_null() || ids.is_null() || result.is_null()
            || count == 0 || count > SOURCE_DIM || ![0,7].contains(&drop_bits)
            || id_bits == 0 || id_bits > 32 {
            return Err("Invalid BFV conversion request (drop_bits must be 0 or 7)".into());
        }
        (*result).pointer = std::ptr::null_mut(); (*result).length = 0;
        let eval = &*key.cast::<BridgeEval>();
        let words = slice::from_raw_parts(words, count*(SOURCE_DIM+1));
        let ids = slice::from_raw_parts(ids, count);
        if ids.iter().any(|&id| id >= (1u64 << id_bits)) {
            return Err("Conversion ID exceeds its public bit width".into());
        }
        let mut packet = Vec::new();
        let start = Instant::now();
        let phases = if let Some(gpu) = &eval.gpu {
            gpu.phase_integers(words)
        } else {
            // Explicit CPU reference backend; never a silent GPU fallback.
            eval.pool.install(|| words.par_chunks_exact(SOURCE_DIM+1)
                .map(|sample| eval.phase_integer(sample)).collect::<Vec<_>>())
        };
        println!("CONVERSION_PHASE backend={} samples={count} keyswitch_bits_ms={:.3}",
            if eval.gpu.is_some() { "gpu" } else { "cpu" }, start.elapsed().as_secs_f64()*1000.0);
        let start = Instant::now();
        // Four host workers drive independent CUDA streams for native integer
        // arithmetic. Each worker installs its server key once for its chunk.
        let chunk_size = count.div_ceil(4);
        let parts = eval.gpu_pool.install(|| phases.par_chunks(chunk_size).zip(ids.par_chunks(chunk_size))
            .map(|(phases, ids)| {
                tfhe::set_server_key(eval.gpu_key.clone());
                let result = phases.iter().zip(ids).map(|(phase, &id)| {
                    // SERVER GPU: exact modulus correction; truncation is explicit.
                    let score: FheUint64 = (phase * SOURCE_T) >> (PHASE_BITS as u64 + drop_bits as u64);
                    // Distance and ID are separate native ciphertexts. IDs
                    // are public at this input stage; later secret selection
                    // makes the selected ID confidential without decryption.
                    let distance: FheUint32 = score.cast_into();
                    let record_id = FheUint32::encrypt_trivial(id as u32);
                    Ok([
                        bincode::serialize(&distance).map_err(|e| e.to_string())?,
                        bincode::serialize(&record_id).map_err(|e| e.to_string())?,
                    ])
                }).collect::<Result<Vec<_>, String>>();
                tfhe::unset_server_key();
                result
            }).collect::<Vec<_>>());
        for part in parts {
            for fields in part? {
                for blob in fields {
                    packet.extend_from_slice(&(blob.len() as u64).to_le_bytes());
                    packet.extend_from_slice(&blob);
                }
            }
        }
        println!("CONVERSION_RADIX samples={count} gpu_workers=4 ms={:.3}", start.elapsed().as_secs_f64()*1000.0);
        let mut packet = packet.into_boxed_slice();
        (*result).pointer = packet.as_mut_ptr(); (*result).length = packet.len();
        std::mem::forget(packet);
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn bfv_tfhe_key_destroy(key: *mut c_void) {
    if !key.is_null() { drop(Box::from_raw(key.cast::<BridgeEval>())); }
}
#[no_mangle]
pub unsafe extern "C" fn bfv_tfhe_buffer_destroy(buffer: Buffer) {
    if !buffer.pointer.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(buffer.pointer, buffer.length)));
    }
}
