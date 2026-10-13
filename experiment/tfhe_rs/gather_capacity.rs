// Measure a real ciphertext from the same cached TFHE-rs build as the C API.
// This is a capacity check, not a completed million-record gather benchmark.
use std::fs;
use tfhe::integer::IntegerCiphertext;
use tfhe::prelude::*;
use tfhe::{ClientKey, ConfigBuilder, FheUint32};

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() <= 2 || args.len() == 7,
        "Usage: tfhe_gather_capacity [REPORT [N C u B k]]");
    let report = args.get(1).cloned()
        .unwrap_or_else(|| "logs/tfhe_gather_1m_split_capacity.json".into());
    let parse = |index: usize, fallback: usize| args.get(index)
        .map(|arg| arg.parse::<usize>().expect("Expected an unsigned integer"))
        .unwrap_or(fallback);
    let records = parse(2, 1_000_000);
    let clusters = parse(3, 500);
    let selected = parse(4, 100);
    let buckets = parse(5, 10_000);
    let k = parse(6, 10);
    assert!(records > 0 && records <= (1 << 20) && clusters > 0 && clusters <= records
        && records % clusters == 0 && selected > 0 && selected <= clusters
        && buckets > 0 && buckets <= 65536 && k > 0 && k <= buckets,
        "Invalid workload dimensions for 20-bit IDs");
    let rows = records / clusters;
    let candidates = selected * rows;
    let capacity = candidates.div_ceil(buckets);
    // CLIENT: measure both fields under the same GPU-compatible parameters.
    // Distances use 23 value bits; IDs use 20. Each has a native uint32 container.
    let key = ClientKey::generate(ConfigBuilder::default().build());
    let distance = FheUint32::encrypt(8_323_200u32, &key);
    let id = FheUint32::encrypt(records as u32 - 1, &key);
    let decoded_distance: u32 = distance.decrypt(&key);
    let decoded_id: u32 = id.decrypt(&key);
    assert_eq!(decoded_distance, 8_323_200);
    assert_eq!(decoded_id, records as u32 - 1);
    let serialized_bytes = bincode::serialized_size(&distance).unwrap()
        + bincode::serialized_size(&id).unwrap();
    let (raw, _, _, _) = distance.into_raw_parts();
    let (raw_id, _, _, _) = id.into_raw_parts();
    let blocks = raw.blocks();
    let lwe_words = blocks[0].ct.as_ref().len();
    assert_eq!(blocks.len(), raw_id.blocks().len());
    assert!(blocks.iter().chain(raw_id.blocks()).all(|block| block.ct.as_ref().len() == lwe_words));
    let payload_bytes: usize = blocks.iter().chain(raw_id.blocks())
        .map(|block| std::mem::size_of_val(block.ct.as_ref())).sum();
    // Raw LWE words alone are a lower bound; object/allocator metadata is extra.
    let input_bytes = payload_bytes * records;
    let output_bytes = payload_bytes * selected * rows;
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap();
    let available: usize = meminfo.lines().find_map(|line| {
        line.strip_prefix("MemAvailable:")
            .map(|v| v.split_whitespace().next().unwrap().parse::<usize>().unwrap() * 1024)
    }).unwrap();
    let (mut gpu_free, mut gpu_total) = (0, 0);
    unsafe {
        assert_eq!(cudaSetDevice(0), 0);
        assert_eq!(cudaMemGetInfo(&mut gpu_free, &mut gpu_total), 0);
    }
    let blocked = input_bytes > available;
    let status = if blocked { "CAPACITY_BLOCKED" } else { "CAPACITY_CHECK_ONLY" };
    let json = format!(concat!(
        "{{\n  \"status\": \"{}\",\n  \"full_gather_executed\": false,\n",
        "  \"gather_evaluation_ms\": null,\n",
        "  \"records\": {}, \"clusters\": {}, \"rows_per_cluster\": {}, \"selected_clusters\": {},\n",
        "  \"candidates\": {}, \"buckets\": {}, \"bucket_capacity\": {}, \"k\": {},\n",
        "  \"score_bits\": 23, \"id_bits\": 20, \"record_type\": \"FheUint32 distance + FheUint32 ID\",\n",
        "  \"measured_radix_blocks_per_field\": {}, \"measured_lwe_words_per_block\": {},\n",
        "  \"measured_payload_bytes_per_record\": {}, \"measured_serialized_bytes_per_record\": {},\n",
        "  \"input_payload_bytes_lower_bound\": {}, \"output_payload_bytes_lower_bound\": {},\n",
        "  \"input_plus_gather_payload_bytes_lower_bound\": {}, \"centroid_payload_bytes_lower_bound\": {},\n",
        "  \"host_available_bytes\": {}, \"gpu_free_bytes\": {}, \"gpu_total_bytes\": {},\n",
        "  \"equality_calls_required\": {}, \"field_cmux_calls_required\": {},\n",
        "  \"reason\": \"Current gather holds every independent input ciphertext in memory; lower bound excludes keys, metadata and workspaces.\"\n}}\n"),
        status, records, clusters, rows, selected, candidates, buckets, capacity, k,
        blocks.len(), lwe_words,
        payload_bytes, serialized_bytes, input_bytes, output_bytes,
        input_bytes + output_bytes, payload_bytes * clusters, available,
        gpu_free, gpu_total, selected * clusters, 2 * selected * records);
    fs::write(report, &json).unwrap();
    print!("{json}");
    if blocked { std::process::exit(77); }
}
