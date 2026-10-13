//! Size probe for the bridge's pinned key parameters, using fresh probe keys.
//! Seeded generation is experimental here; the production bridge is unchanged.
use std::{fs, mem::size_of_val};
use tfhe::core_crypto::gpu::{CudaStreams, vec::GpuIndex};
use tfhe::core_crypto::gpu::entities::{
    lwe_bootstrap_key::CudaLweBootstrapKey, lwe_keyswitch_key::CudaLweKeyswitchKey,
};
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::{
    parameters_wopbs_message_carry::LEGACY_WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS as BIT,
    PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 as GPU,
};

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}
fn free_gpu() -> usize {
    let (mut free, mut total) = (0, 0);
    unsafe {
        assert_eq!(cudaDeviceSynchronize(), 0);
        assert_eq!(cudaMemGetInfo(&mut free, &mut total), 0);
    }
    free
}

struct Size { name: &'static str, expanded: usize, seeded: usize, serialized: u64 }
impl Size {
    fn json(&self) -> String {
        format!("{{\"name\":\"{}\",\"expanded_payload_bytes\":{},\"seeded_payload_bytes\":{},\"seeded_serialized_bytes\":{}}}",
            self.name, self.expanded, self.seeded, self.serialized)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() <= 2, "Usage: bfv_tfhe_key_size [REPORT_JSON]");
    let report = args.get(1).map(String::as_str).unwrap_or("logs/bfv_tfhe_key_size.json");
    unsafe { assert_eq!(cudaSetDevice(0), 0); }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
    let json = pool.install(|| {
        let client = tfhe::ClientKey::generate(tfhe::ConfigBuilder::with_custom_parameters(GPU).build());
        let short = client.clone().into_raw_parts().0.into_raw_parts();
        let big = short.encryption_key();
        let glwe = GlweSecretKey::from_container(big.as_ref(), GPU.polynomial_size);
        let mut seeder = new_seeder();
        let mut rng = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
        // Sample a ternary source for measurement; dimensions and noise/
        // decomposition parameters match bfv_tfhe.rs. No secret is exported.
        let a = allocate_and_generate_new_binary_lwe_secret_key::<u64, _>(LweDimension(4096), &mut rng);
        let b = allocate_and_generate_new_binary_lwe_secret_key::<u64, _>(LweDimension(4096), &mut rng);
        let source = LweSecretKey::from_container(a.as_ref().iter().zip(b.as_ref())
            .map(|(&x, &y)| x.wrapping_sub(y)).collect::<Vec<_>>());
        let small = allocate_and_generate_new_binary_lwe_secret_key::<u64, _>(BIT.lwe_dimension, &mut rng);
        let native = CiphertextModulus::new_native();
        let mut sizes = Vec::new();
        eprintln!("Generating seeded source-to-big keyswitch key...");
        let seeded_source = allocate_and_generate_new_seeded_lwe_keyswitch_key(&source, &big,
            DecompositionBaseLog(8), DecompositionLevelCount(7),
            GPU.glwe_noise_distribution, native, seeder.as_mut());
        let seeded = size_of_val(seeded_source.as_ref());
        let serialized = bincode::serialized_size(&seeded_source).unwrap();
        let source_to_big = seeded_source.decompress_into_lwe_keyswitch_key();
        sizes.push(Size { name: "source_to_big", expanded: size_of_val(source_to_big.as_ref()), seeded, serialized });

        let seeded_small = allocate_and_generate_new_seeded_lwe_keyswitch_key(&big, &small,
            BIT.ks_base_log, BIT.ks_level, BIT.lwe_noise_distribution, native, seeder.as_mut());
        let seeded = size_of_val(seeded_small.as_ref());
        let serialized = bincode::serialized_size(&seeded_small).unwrap();
        let big_to_small = seeded_small.decompress_into_lwe_keyswitch_key();
        sizes.push(Size { name: "big_to_small", expanded: size_of_val(big_to_small.as_ref()), seeded, serialized });

        let mut bootstrap_keys = Vec::new();
        for (name, base, level) in [("precise_bsk", 4, 12), ("output_bsk", 8, 4)] {
            eprintln!("Generating seeded {name}...");
            let seeded_key = par_allocate_and_generate_new_seeded_lwe_bootstrap_key(&small, &glwe,
                DecompositionBaseLog(base), DecompositionLevelCount(level),
                GPU.glwe_noise_distribution, native, seeder.as_mut());
            let seeded = size_of_val(seeded_key.as_ref());
            let serialized = bincode::serialized_size(&seeded_key).unwrap();
            let expanded = seeded_key.par_decompress_into_lwe_bootstrap_key();
            sizes.push(Size { name, expanded: size_of_val(expanded.as_ref()), seeded, serialized });
            bootstrap_keys.push(expanded);
        }
        // The production bridge retains a CPU Fourier copy in addition to its
        // GPU copy. Measure these buffers as well; report workspace separately.
        let mut fourier_bytes = 0;
        for bsk in &bootstrap_keys {
            let mut fourier = FourierLweBootstrapKeyOwned::new(bsk.input_lwe_dimension(),
                bsk.glwe_size(), bsk.polynomial_size(), bsk.decomposition_base_log(),
                bsk.decomposition_level_count());
            par_convert_standard_lwe_bootstrap_key_to_fourier(bsk, &mut fourier);
            fourier_bytes += size_of_val(fourier.as_view().data());
        }
        let streams = CudaStreams::new_single_gpu(GpuIndex::new(0));
        let before = free_gpu();
        let d_source = CudaLweKeyswitchKey::from_lwe_keyswitch_key(&source_to_big, &streams);
        let d_small = CudaLweKeyswitchKey::from_lwe_keyswitch_key(&big_to_small, &streams);
        let d_precise = CudaLweBootstrapKey::from_lwe_bootstrap_key(&bootstrap_keys[0], None, &streams);
        let d_output = CudaLweBootstrapKey::from_lwe_bootstrap_key(&bootstrap_keys[1], None, &streams);
        let bridge_gpu_allocation = before.saturating_sub(free_gpu());

        eprintln!("Measuring the separate native TFHE server key...");
        let compressed_native = client.generate_compressed_server_key();
        let native_serialized = bincode::serialized_size(&compressed_native).unwrap();
        let before = free_gpu();
        let native_gpu = compressed_native.decompress_to_gpu();
        let native_gpu_allocation = before.saturating_sub(free_gpu());
        let expanded: usize = sizes.iter().map(|s| s.expanded).sum();
        let seeded: usize = sizes.iter().map(|s| s.seeded).sum();
        let serialized: u64 = sizes.iter().map(|s| s.serialized).sum();
        let cpu_retained = sizes[0].expanded + sizes[1].expanded + fourier_bytes;
        let json = format!(concat!(
            "{{\n  \"status\": \"PASS\",\n  \"source_lwe_dimension\": 4096,\n",
            "  \"target_lwe_dimension\": {}, \"auxiliary_lwe_dimension\": {},\n",
            "  \"keys\": [{}],\n",
            "  \"bridge_expanded_payload_bytes\": {},\n",
            "  \"bridge_seeded_payload_bytes\": {},\n",
            "  \"bridge_seeded_serialized_bytes\": {},\n",
            "  \"production_bridge_cpu_retained_payload_bytes\": {},\n",
            "  \"observed_bridge_gpu_allocation_bytes\": {},\n",
            "  \"native_tfhe_compressed_server_key_serialized_bytes\": {},\n",
            "  \"observed_native_tfhe_gpu_allocation_bytes\": {},\n",
            "  \"seeded_bridge_plus_native_serialized_bytes\": {},\n",
            "  \"production_seeded_bridge_enabled\": false,\n",
            "  \"scope\": \"Fresh keys under current bridge parameters; seeded generation, decompression, Fourier conversion and GPU upload only. No conversion accuracy or throughput test. GPU deltas include allocator overhead and may include library initialization; exclude ciphertexts and evaluation workspaces.\"\n}}\n"),
            big.lwe_dimension().0, small.lwe_dimension().0,
            sizes.iter().map(Size::json).collect::<Vec<_>>().join(",\n    "),
            expanded, seeded, serialized, cpu_retained, bridge_gpu_allocation,
            native_serialized, native_gpu_allocation, serialized + native_serialized);
        drop((native_gpu, d_source, d_small, d_precise, d_output));
        json
    });
    fs::write(report, &json).unwrap();
    print!("{json}");
}
