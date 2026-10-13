#include "he/tfhe_select.hpp"
#include "utils/tfhe_crypto.hpp"
#include "utils/check_support.hpp"
#include <cuda_profiler_api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>

// Compare one 300-to-1 bucket against a wide network with similar total work.
// Each public configuration runs once. This is a scheduling diagnostic, not
// a statistically repeated benchmark or a prediction of the full 30k workload.
int main(int argc, char** argv) {
    using namespace privateann;
    using namespace privateann::checks;
    const bool two_buckets = argc >= 2 && std::string(argv[1]) == "--two-buckets";
    const bool launch_check = argc >= 2 && std::string(argv[1]) == "--launch-check";
    const bool selected_case = two_buckets || launch_check;
    const auto* profile_env = std::getenv("PRIVATEIMPL_PROFILE_BUCKETS");
    const bool profile = profile_env && std::string(profile_env) == "1";
    if (argc > (selected_case ? 3 : 2)) {
        std::cerr << "Usage: bucket_parallel_once [--two-buckets | --launch-check] [OUTPUT_CSV]\n";
        return 1;
    }
    const auto output_index = selected_case ? 2 : 1;
    const std::filesystem::path report = argc > output_index ? argv[output_index] :
        (two_buckets ? "logs/bucket_two_300_once.csv" :
         launch_check ? "logs/bucket_launch_check.csv" : "logs/bucket_parallel_once.csv");
    if (report.has_parent_path()) std::filesystem::create_directories(report.parent_path());
    std::ofstream csv(report);
    if (!csv) { std::cerr << "Cannot open report\n"; return 1; }
    csv << "case,buckets,capacity,layers,minimum_gates,requested_workers,workers_used,"
           "peak_inflight,evaluation_ms,encryption_ms,verification_ms,correct\n"
        << std::fixed << std::setprecision(3);
    struct Case { const char* name; std::size_t buckets, capacity; unsigned workers; };
    const std::vector<Case> cases = two_buckets ? std::vector<Case>{
        {"two_300_w4", 2, 300, 4},
    } : launch_check ? std::vector<Case>{
        {"launch_check_2x4_w4", 2, 4, 4},
    } : std::vector<Case>{
        {"single_300_w4", 1, 300, 4},
        {"wide_100x4_w4", 100, 4, 4},
        {"wide_100x4_w16", 100, 4, 16},
    };
    const auto result = run("bucket_parallel_once", [&](Result& result) {
        require_gpu();
        std::cout << "SCOPE: one trial per configuration; GPU native gt + paired field CMUX.\n"
                     "Distances are 128D uint8 L2, stored with IDs as separate FheUint32.\n";
        std::mt19937 random(20261013);
        std::size_t fixture_size = 0;
        for (const auto& config : cases)
            fixture_size = std::max(fixture_size, config.buckets * config.capacity);
        std::vector<he::PlainRecord> fixture;
        for (std::uint32_t i = 0; i < fixture_size; ++i) {
            std::uint32_t distance = 0;
            for (std::size_t d = 0; d < 128; ++d) {
                const auto delta = std::int64_t(d % 2 ? 255 : 0) - std::int64_t(random() & 255);
                distance += delta * delta;
            }
            fixture.push_back({distance, i * 7919u % 1000000u, true});
        }
        // CLIENT KEY GENERATION: one private key; only evaluation material is
        // passed to each server evaluator. Setup is outside evaluation timing.
        auto client = measure(result, Phase::setup, "client_keys", [&] {
            return std::make_unique<entity::TfheKeys>();
        });
        utils::TfheCrypto crypto(*client);
        for (const auto& config : cases) {
            const auto schedule = he::topk_schedule(config.capacity, 1);
            std::cout << "CONFIG " << config.name << " buckets=" << config.buckets
                      << " capacity=" << config.capacity << " workers=" << config.workers
                      << " layers=" << schedule.layers.size()
                      << " minimum_gates=" << config.buckets * (config.capacity - 1) << std::endl;
            auto evaluator = measure(result, Phase::setup, "server_setup", [&] {
                return std::make_unique<he::tfhe_rs::Evaluator>(client->evaluation_key(), config.workers);
            });
            // CLIENT ENCRYPTION: fresh inputs for each case avoid reusing any
            // device residency that a previous case might have populated.
            const auto encryption_before = result.encryption_ms;
            auto buckets = measure(result, Phase::encryption, "input_encryption", [&] {
                std::vector<std::vector<he::tfhe_rs::Record>> encrypted(config.buckets);
                for (std::size_t b = 0; b < config.buckets; ++b)
                    for (std::size_t row = 0; row < config.capacity; ++row) {
                        const auto& value = fixture[b * config.capacity + row];
                        encrypted[b].push_back(crypto.encrypt(value.score, value.id, 23, 20));
                    }
                return encrypted;
            });
            const auto encryption_ms = result.encryption_ms - encryption_before;
            // SERVER: the same production executor and minimum operation as
            // the large experiment. Only public shape/worker count changes.
            // Optional Nsight capture excludes key setup, input encryption and
            // result verification. Profiled timings are not baseline timings.
            if (profile) cuda_check(cudaProfilerStart(), "cudaProfilerStart");
            const auto before = result.evaluation_ms;
            const auto winners = measure(result, Phase::evaluation, config.name, [&] {
                return evaluator->bucket_minima(buckets);
            });
            const auto evaluation_ms = result.evaluation_ms - before;
            if (profile) cuda_check(cudaProfilerStop(), "cudaProfilerStop");
            // CLIENT DECRYPTION: verify each winner against an integer oracle;
            // decrypted output is never fed into another server operation.
            const auto verification_before = result.decryption_ms;
            measure(result, Phase::decryption, "verify_winners", [&] {
                if (winners.size() != config.buckets) throw std::runtime_error("Wrong bucket count");
                for (std::size_t b = 0; b < config.buckets; ++b) {
                    const auto first = fixture.begin() + b * config.capacity;
                    const std::vector<he::PlainRecord> expected(first, first + config.capacity);
                    require_distance_topk({crypto.decrypt(winners[b])}, expected, 1);
                }
            });
            csv << config.name << ',' << config.buckets << ',' << config.capacity << ','
                << schedule.layers.size() << ',' << config.buckets * (config.capacity - 1) << ','
                << config.workers << ',' << evaluator->last_workers_used() << ','
                << evaluator->last_peak_inflight() << ',' << evaluation_ms << ',' << encryption_ms << ','
                << result.decryption_ms - verification_before << ',' << winners.size() << '\n';
            csv.flush();
            std::cout << "VERIFIED " << config.name << " winners=" << winners.size()
                      << " workers_used=" << evaluator->last_workers_used()
                      << " peak_inflight=" << evaluator->last_peak_inflight() << std::endl;
        }
    });
    csv.close();
    if (!csv) { std::cerr << "Failed to write report\n"; return 1; }
    return exit_code(result);
}
