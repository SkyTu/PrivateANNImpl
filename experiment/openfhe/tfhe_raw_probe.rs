use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;
use tfhe::conformance::ParameterSetConformant;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::ciphertext::{Degree, NoiseLevel};
use tfhe::shortint::parameters::{
    CarryModulus, MessageModulus,
    PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128 as GPU_PARAMS,
};
use tfhe::{FheUint128, FheUint128ConformanceParams};

fn read_u64(reader: &mut impl Read) -> u64 {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes).expect("Truncated fixture");
    u64::from_le_bytes(bytes)
}

fn main() {
    let filename = std::env::args().nth(1).expect("Usage: tfhe_raw_probe PRIVATE_FIXTURE");
    let mut reader = BufReader::new(File::open(filename).expect("Cannot open fixture"));
    let mut magic = [0; 8];
    reader.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"OFHETFH1");
    let count = read_u64(&mut reader) as usize;
    let dimension = read_u64(&mut reader) as usize;
    let q = read_u64(&mut reader);
    let p = read_u64(&mut reader);
    assert_eq!(count, 500);
    assert_eq!(dimension, 1305);
    assert!(q.is_power_of_two() && p.is_power_of_two() && p <= q);

    // CLIENT DIAGNOSTIC KEY IMPORT ONLY: map ternary -1 to its u64 residue.
    // This key is used only to verify raw phases, never for server evaluation.
    let secret: Vec<u64> = (0..dimension).map(|_| read_u64(&mut reader)).collect();
    assert!(secret.iter().all(|&x| x == 0 || x == 1 || x == u64::MAX));
    let has_negative_coefficient = secret.contains(&u64::MAX);
    let key = LweSecretKey::from_container(secret);

    let expected_block = GPU_PARAMS.to_shortint_conformance_param();
    println!("TFHE_GPU_PARAMS: small_lwe_dimension={} input_lwe_dimension={} message_modulus={} carry_modulus={} native_modulus={} FheUint128_blocks={}",
        GPU_PARAMS.lwe_dimension.0, expected_block.ct_params.lwe_dim.0,
        GPU_PARAMS.message_modulus.0, GPU_PARAMS.carry_modulus.0,
        GPU_PARAMS.ciphertext_modulus.is_native_modulus(),
        128 / GPU_PARAMS.message_modulus.0.ilog2());
    println!("OPENFHE_INPUT: dimension={dimension} q={q} p={p} ternary_key_has_minus_one={has_negative_coefficient}");

    let shift = 64 - q.ilog2();
    let delta = (1u128 << 64) / u128::from(p);
    let shortint_delta = (1u128 << 63)
        / u128::from(GPU_PARAMS.message_modulus.0 * GPU_PARAMS.carry_modulus.0);
    println!("ENCODING: imported_delta={delta} GPU_shortint_delta={shortint_delta} (includes padding and carry space)");
    let mut phase_matches = 0;
    let mut full_matches = 0;
    let mut raw_conformant = 0;
    let mut integer_conformant = 0;
    let mut import_ms = 0.0;
    let mut verify_ms = 0.0;

    for _ in 0..count {
        let original = read_u64(&mut reader);
        let openfhe_decoded = read_u64(&mut reader);
        let coefficients: Vec<u64> = (0..=dimension).map(|_| read_u64(&mut reader)).collect();
        assert!(coefficients.iter().all(|&x| x < q));

        // PUBLIC CIPHERTEXT ADAPTATION: lift a and b from Z_q to the u64 torus.
        // No decryption, encryption, key switch or radix decomposition occurs here.
        let start = Instant::now();
        let lifted = coefficients.into_iter().map(|x| x << shift).collect::<Vec<_>>();
        let raw = LweCiphertext::from_container(lifted, CiphertextModulus::new_native());
        import_ms += start.elapsed().as_secs_f64() * 1000.0;

        // CLIENT VERIFICATION: TFHE-rs computes the LWE phase with the SAME key.
        // Decode using OpenFHE's p; this is not TFHE-rs shortint encoding.
        let start = Instant::now();
        let phase: Plaintext<u64> = decrypt_lwe_ciphertext(&key, &raw);
        let decoded = ((u128::from(phase.0) + delta / 2) / delta % u128::from(p)) as u64;
        phase_matches += usize::from(decoded == openfhe_decoded);
        full_matches += usize::from(decoded == original);
        verify_ms += start.elapsed().as_secs_f64() * 1000.0;

        // SERVER INPUT VALIDATION: use the real TFHE-rs conformance interfaces.
        // The diagnostic wrapper records the large message modulus honestly;
        // adding metadata cannot manufacture the absent radix digits or keys.
        raw_conformant += usize::from(raw.is_conformant(&expected_block.ct_params));
        let block = tfhe::shortint::Ciphertext::new(
            raw, Degree::new(p - 1), NoiseLevel::UNKNOWN,
            MessageModulus(p), CarryModulus(1), expected_block.atomic_pattern,
        );
        let integer = FheUint128::from_raw_parts(
            tfhe::integer::RadixCiphertext::from(vec![block]),
            Default::default(), Default::default(), Default::default(),
        );
        integer_conformant += usize::from(integer.is_conformant(
            &FheUint128ConformanceParams::from(GPU_PARAMS),
        ));
    }
    assert_eq!(reader.read(&mut [0u8; 1]).unwrap(), 0, "Unexpected trailing bytes");
    println!("RAW_IMPORT: phase_matches={phase_matches}/{count} original_distance_matches={full_matches}/{count} import_ms={import_ms:.6} client_verify_ms={verify_ms:.6}");
    println!("GPU_CONFORMANCE: raw_lwe_accepted={raw_conformant}/{count} FheUint128_accepted={integer_conformant}/{count}");
    assert_eq!(phase_matches, count, "The ciphertext adapter changed the LWE message");
    assert_eq!(raw_conformant, 0, "Expected dimension incompatibility disappeared");
    assert_eq!(integer_conformant, 0, "Expected radix incompatibility disappeared");
    println!("BLOCKED: raw LWE import works; existing TFHE-rs GPU integer input is incompatible. GPU min/max was not executed.");
    // 2 explicitly means an incomplete bridge; a passing raw import is not success.
    std::process::exit(2);
}
