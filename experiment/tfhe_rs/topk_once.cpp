#include "he/detail/selection_network.hpp"
#include "he/tfhe_topk.hpp"
#include "utils/tfhe_crypto.hpp"
#include "utils/check_support.hpp"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
namespace tfhe = privateann::he::tfhe_rs;

void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class Function>
double measure(const char* label, Function&& function) {
    cuda_check(cudaDeviceSynchronize());
    const auto start = Clock::now();
    function();
    cuda_check(cudaDeviceSynchronize());
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << label << "_ms=" << ms << std::endl;
    return ms;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("Usage: tfhe_topk_once [OUTPUT_CSV]");
        constexpr unsigned score_bits = 23, id_bits = 7;
        std::cout << std::fixed << std::setprecision(6);
        cuda_check(cudaSetDevice(0));
        cudaDeviceProp device{};
        cuda_check(cudaGetDeviceProperties(&device, 0));
        constexpr std::size_t count = 100, k = 10, dimensions = 128;
        std::cout << "GPU: " << device.name << '\n'
                  << "100 candidates, k=10, 128D uint8 squared L2, separate FheUint32 distance and ID\n"
                  << "Encoding: " << score_bits << " score bits + " << id_bits
                  << " ID bits; distance-only comparison, arbitrary ties, no truncation\n"
                  << "Exactly one Top-k evaluation; no warmup or repeated trials.\n";

        // Public fixture: identical plaintext distances to the OpenFHE CPU run.
        std::mt19937 rng(20261012);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> fixture;
        for (std::size_t i = 0; i < count; ++i) {
            std::uint64_t distance = 0;
            for (std::size_t j = 0; j < dimensions; ++j) {
                const std::int64_t query = (j % 2) ? 255 : 0;
                std::int64_t record = rng() & 255;
                if (i == 0) record = query;
                if (i == 1) record = 255 - query;
                const auto difference = query - record;
                distance += difference * difference;
            }
            fixture.emplace_back(distance, i);
        }
        std::size_t full_gates = 0, min_only_gates = 0;
        const auto network = privateann::he::topk_schedule(count, k);
        for (const auto& layer : network.layers)
            for (const auto& gate : layer)
                if (gate.min_only) ++min_only_gates;
                else ++full_gates;
        std::cout << "Grouped network: padded=" << network.size
                  << ", layers=" << network.layers.size()
                  << ", compare/select gates=" << full_gates
                  << ", min-only gates=" << min_only_gates << '\n'
                  << "Native calls: distance_gt=" << full_gates + min_only_gates
                  << ", field_if_then_else=" << 4 * full_gates + 2 * min_only_gates << std::endl;

        // CLIENT: generate the secret key and compressed evaluation key once.
        std::unique_ptr<privateann::entity::TfheKeys> client;
        const double keygen_ms = measure("client_keygen", [&] {
            client = std::make_unique<privateann::entity::TfheKeys>();
        });

        privateann::utils::TfheCrypto crypto(*client);

        // SERVER: load the evaluation key onto the GPU once; no secret key.
        std::unique_ptr<tfhe::Evaluator> evaluator;
        const double setup_ms = measure("server_gpu_key_setup", [&] {
            evaluator = std::make_unique<tfhe::Evaluator>(client->evaluation_key());
        });
        std::cout << "Requested layer workers=" << evaluator->workers() << std::endl;

        // CLIENT: encrypt integer distances directly in TFHE-rs, not via CKKS.
        std::vector<tfhe::Record> encrypted;
        encrypted.reserve(count);
        const double encryption_ms = measure("client_encryption", [&] {
            for (const auto& [distance, id] : fixture)
                encrypted.push_back(crypto.encrypt(distance, id, score_bits, id_bits));
        });
        for (const auto& record : encrypted)
            if (record.distance_ciphertext_bits() != 32 || record.id_ciphertext_bits() != 32) throw std::runtime_error("Incorrect ciphertext width");

        // SERVER: use the same selection network as he::topk exactly once.
        // This includes padding, native GPU comparison/selection and transfers.
        std::cout << "Starting the single Top-k evaluation..." << std::endl;
        std::vector<tfhe::Record> result;
        const double evaluation_ms = measure("topk_evaluation", [&] {
            result = privateann::he::topk(*evaluator, std::move(encrypted), k);
        });
        std::cout << "Layer workers used=" << evaluator->last_workers_used()
                  << ", peak in-flight gates=" << evaluator->last_peak_inflight() << std::endl;
        if (evaluator->last_workers_used() < 2 || evaluator->last_peak_inflight() < 2)
            throw std::runtime_error("Layer parallelism was not exercised");

        // CLIENT: decrypt only after evaluation and verify every selected score/ID.
        const double verification_ms = measure("client_decryption_verification", [&] {
            std::vector<privateann::he::PlainRecord> values;
            for (const auto& record : result) values.push_back(crypto.decrypt(record));
            privateann::checks::require_distance_topk(values, fixture, k);
            for (std::size_t i = 0; i < k; ++i) {
                const auto& actual = values[i];
                std::cout << "rank=" << i << " id=" << actual.id
                          << " distance=" << actual.score << '\n';
            }
        });
        const std::string report_path = argc > 1 ? argv[1] : "logs/tfhe_rs_topk_100_10_split_once.csv";
        std::ofstream report(report_path);
        if (!report) throw std::runtime_error("Cannot write " + report_path);
        report << std::fixed << std::setprecision(6)
               << "candidates,k,dimensions,element_bits,distance_ciphertext_bits,id_ciphertext_bits,evaluations,comparators,"
                  "compare_select_gates,min_only_gates,workers_used,peak_inflight_gates,"
                  "client_keygen_ms,server_gpu_key_setup_ms,client_encryption_ms,"
                  "topk_evaluation_ms,client_decryption_verification_ms,correct\n"
               << count << ',' << k << ',' << dimensions << ",8,32,32,1,"
               << full_gates + min_only_gates << ',' << full_gates << ',' << min_only_gates << ','
               << evaluator->last_workers_used() << ',' << evaluator->last_peak_inflight() << ','
               << keygen_ms << ',' << setup_ms << ',' << encryption_ms << ','
               << evaluation_ms << ',' << verification_ms << ",10\n";
        report.close();
        if (!report) throw std::runtime_error("Failed writing " + report_path);
        std::cout << "PASS: 10/10 distances and matching IDs; report=" << report_path << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
