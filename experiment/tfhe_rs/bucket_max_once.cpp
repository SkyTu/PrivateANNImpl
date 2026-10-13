#include "he/tfhe_select.hpp"
#include "utils/tfhe_crypto.hpp"
#include "utils/check_support.hpp"
#include <cuda_profiler_api.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>

// Single-trial paired argmax benchmark. All inputs are valid; no padding is
// used. Reuse the production reduction schedule and native distance gt/CMUX.
int main(int argc, char** argv) {
    using namespace privateann;
    using namespace privateann::checks;
    if (argc != 5) {
        std::cerr << "Usage: bucket_max_once BUCKETS CAPACITY WORKERS OUTPUT_CSV\n";
        return 1;
    }
    const std::filesystem::path report = argv[4];
    if (report.has_parent_path()) std::filesystem::create_directories(report.parent_path());
    std::ofstream csv(report);
    if (!csv) { std::cerr << "Cannot open report\n"; return 1; }
    const auto result = run("bucket_max_once", [&](Result& result) {
        const auto count = std::stoul(argv[1]), capacity = std::stoul(argv[2]);
        const auto workers = std::stoul(argv[3]);
        if (!count || count > 100 || capacity < 2 || capacity > 65536 ||
            !workers || workers > 32) throw std::invalid_argument("Invalid public dimensions");
        const auto* arena_env = std::getenv("PRIVATEIMPL_TFHE_ARENA_MB");
        const std::string arena = arena_env ? arena_env : "0";
        const auto* profile_env = std::getenv("PRIVATEIMPL_PROFILE_BUCKETS");
        const bool profile = profile_env && std::string(profile_env) == "1";
        require_gpu();
        std::cout << "CONFIG buckets=" << count << " capacity=" << capacity
                  << " workers=" << workers << " arena_mb=" << arena
                  << " operation=maximum_with_id" << std::endl;
        std::mt19937 random(20261013);
        std::vector<he::PlainRecord> fixture;
        for (std::uint32_t i = 0; i < count * capacity; ++i) {
            std::uint32_t distance = 0;
            for (unsigned d = 0; d < 128; ++d) {
                const auto delta = std::int64_t(d % 2 ? 255 : 0) - std::int64_t(random() & 255);
                distance += delta * delta;
            }
            fixture.push_back({distance, i * 7919u % 1000000u, true});
        }
        // Cover an exact tie at the 128D uint8 L2 upper bound and zero, while
        // other buckets keep independent random maxima and associated IDs.
        fixture[0].score = 8323200;
        fixture[capacity - 1].score = 8323200;
        fixture[1].score = 0;

        // CLIENT KEY GENERATION: server receives only evaluation material.
        auto client = measure(result, Phase::setup, "client_keys", [&] {
            return std::make_unique<entity::TfheKeys>();
        });
        utils::TfheCrypto crypto(*client);
        auto evaluator = measure(result, Phase::setup, "server_setup", [&] {
            return std::make_unique<he::tfhe_rs::Evaluator>(client->evaluation_key(), workers);
        });
        // CLIENT ENCRYPTION: each process starts with freshly encrypted inputs.
        auto values = measure(result, Phase::encryption, "input_encryption", [&] {
            std::vector<he::tfhe_rs::Record> encrypted;
            encrypted.reserve(fixture.size());
            for (const auto& value : fixture)
                encrypted.push_back(crypto.encrypt(value.score, value.id, 23, 20));
            return encrypted;
        });
        const auto local = he::topk_schedule(capacity, 1);
        he::TopkSchedule schedule{count * capacity, {}};
        schedule.layers.resize(local.layers.size());
        for (std::size_t b = 0; b < count; ++b)
            for (std::size_t layer = 0; layer < local.layers.size(); ++layer)
                for (const auto& gate : local.layers[layer])
                    schedule.layers[layer].push_back({b * capacity + gate.a, b * capacity + gate.b, true});
        if (profile) cuda_check(cudaProfilerStart(), "cudaProfilerStart");
        // SERVER EVALUATION: fixed addresses and layer barriers. Every node
        // compares distance only, then selects distance and ID with that bit.
        measure(result, Phase::evaluation, "bucket_maxima", [&] {
            evaluator->execute_layers(schedule, [&](const he::TopkGate& gate) {
                values[gate.a] = evaluator->maximum(values[gate.a], values[gate.b]);
            });
        });
        if (profile) cuda_check(cudaProfilerStop(), "cudaProfilerStop");
        // CLIENT DECRYPTION: independent integer argmax oracle; either tied
        // record is accepted only if its original distance and ID match.
        measure(result, Phase::decryption, "verify_all_winners", [&] {
            for (std::size_t b = 0; b < count; ++b) {
                const auto winner = crypto.decrypt(values[b * capacity]);
                std::uint32_t maximum = 0;
                bool association = false;
                for (std::size_t row = 0; row < capacity; ++row) {
                    const auto& input = fixture[b * capacity + row];
                    maximum = std::max(maximum, input.score);
                    association |= winner.score == input.score && winner.id == input.id;
                }
                if (!winner.valid || winner.score != maximum || !association)
                    throw std::runtime_error("Incorrect maximum or distance/ID association");
            }
        });
        csv << "buckets,capacity,workers,workers_used,peak_inflight,arena_mb,layers,gates,"
               "evaluation_ms,setup_ms,encryption_ms,verification_ms,verified\n"
            << std::fixed << std::setprecision(3)
            << count << ',' << capacity << ',' << workers << ',' << evaluator->last_workers_used()
            << ',' << evaluator->last_peak_inflight() << ',' << arena << ',' << schedule.layers.size()
            << ',' << count * (capacity - 1) << ',' << result.evaluation_ms << ',' << result.setup_ms
            << ',' << result.encryption_ms << ',' << result.decryption_ms << ',' << count << '\n';
        std::cout << "VERIFIED maxima=" << count << " workers_used=" << evaluator->last_workers_used()
                  << " peak_host_calls=" << evaluator->last_peak_inflight() << std::endl;
    });
    csv.close();
    if (!csv) { std::cerr << "Failed to write report\n"; return 1; }
    return exit_code(result);
}
