#include "he/entity/client.hpp"
#include "he/entity/server.hpp"
#include "utils/bfv_crypto.hpp"
#include "utils/tfhe_crypto.hpp"
#include "utils/convert.hpp"
#include "utils/check_support.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>

// This is a single integration measurement, not the full regression suite.
// Public synthetic balanced clusters replace offline dataset clustering.
// Output sanitization and network transport are outside this experiment.
int main(int argc, char** argv) {
    using namespace privateann;
    using namespace privateann::checks;
    if (argc > 2) {
        std::cerr << "Usage: selection_pipeline_once [OUTPUT_CSV]\n";
        return 1;
    }
    const std::filesystem::path report = argc == 2 ? argv[1] : "logs/selection_pipeline_split_once.csv";
    if (report.has_parent_path()) std::filesystem::create_directories(report.parent_path());
    std::ofstream csv(report);
    if (!csv) { std::cerr << "Cannot open report: " << report << '\n'; return 1; }

    constexpr std::size_t n = 100, dimensions = 128, c = 10, u = 4, b = 16, k = 10;
    constexpr std::size_t rows = n / c, converted_count = n + c;
    constexpr unsigned id_bits = 20, drop_bits = 0;
    constexpr std::uint64_t data_seed = 20261012, bucket_seed = 20261013;
    std::vector<std::pair<std::string, double>> timings;
    double pipeline_wall_ms = 0, recall = 0;
    unsigned bucket_workers = 0, bucket_peak = 0, topk_workers = 0, topk_peak = 0;
    std::vector<std::uint64_t> output_ids;

    const auto result = run("selection_pipeline_once", [&](Result& result) {
        require_gpu();
        std::cout << "CONFIG N=" << n << " dimensions=" << dimensions << " element_bits=8 C=" << c
                  << " L=" << rows << " u=" << u << " B=" << b << " k=" << k
                  << " score_bits=23 id_bits=" << id_bits << " drop_bits=" << drop_bits
                  << " distance_ciphertext_bits=32 id_ciphertext_bits=32 data_seed=" << data_seed << " bucket_seed=" << bucket_seed
                  << "\nExactly one encrypted pipeline; no intermediate client decryption.\n";
        // Record each completed server stage, including synchronization and
        // its host scheduling/transfers. Key generation and client work are separate.
        const auto stage = [&](const char* label, auto&& operation) {
            const auto before = result.evaluation_ms;
            auto output = measure(result, Phase::evaluation, label, operation);
            timings.emplace_back(label, result.evaluation_ms - before);
            return output;
        };

        // PUBLIC FIXTURE: ten synthetic clusters, ten uint8 vectors each.
        // Centroids are rounded integer means of their own public records.
        // These are supplied clusters, not a k-means/recall dataset benchmark.
        BfvParameters parameters;
        parameters.validate_distance_dimension(dimensions);
        std::mt19937 random(data_seed);
        std::vector<std::vector<std::uint64_t>> columns(
            dimensions, std::vector<std::uint64_t>(parameters.slots, 0));
        std::vector<std::uint64_t> q(dimensions), ids(converted_count);
        for (std::size_t d = 0; d < dimensions; ++d) {
            for (std::size_t cluster = 0; cluster < c; ++cluster) {
                const auto center = static_cast<int>(random() & 255);
                std::uint64_t sum = 0;
                for (std::size_t row = 0; row < rows; ++row) {
                    const auto value = std::clamp(center + static_cast<int>(random() % 17) - 8, 0, 255);
                    columns[d][cluster * rows + row] = value;
                    sum += value;
                }
                columns[d][n + cluster] = (sum + rows / 2) / rows;
            }
            q[d] = std::clamp(static_cast<int>(columns[d][n + 7]) +
                             static_cast<int>(random() % 17) - 8, 0, 255);
        }
        for (std::size_t i = 0; i < n; ++i) ids[i] = 10000 + i * 7919;
        for (std::size_t i = 0; i < c; ++i) ids[n + i] = i;
        std::vector<std::size_t> positions(converted_count);
        std::iota(positions.begin(), positions.end(), 0);

        // CLIENT KEY GENERATION: cross-scheme material is created once using
        // client secrets. Server and converter receive only evaluation keys.
        auto client = measure(result, Phase::setup, "client_keys", [&] {
            return std::make_unique<entity::Client>(parameters, true);
        });
        measure(result, Phase::setup, "conversion_keys", [&] { client->enable_bfv_tfhe_conversion(); });
        utils::BfvCrypto bfv(*client);
        utils::TfheCrypto tfhe(client->tfhe_keys());
        const auto keys = client->evaluation_keys();
        auto server = measure(result, Phase::setup, "server_setup", [&] {
            return std::make_unique<entity::Server>(keys);
        });
        utils::BfvToTfheConverter converter(keys);

        // CLIENT ENCRYPTION: one query is encrypted once for all 110 distances.
        auto query = measure(result, Phase::encryption, "query_encryption", [&] {
            return bfv.encrypt_bfv_query(q);
        });
        client.reset();

        // SERVER PIPELINE: all stages consume the preceding real ciphertexts.
        // No plaintext oracle, decryption, or client re-encryption occurs here.
        synchronize_gpu();
        const auto pipeline_start = Clock::now();
        auto distance = stage("bfv_distance", [&] { return server->squared_l2(query, columns); });
        const auto converted = stage("bfv_to_tfhe", [&] {
            return converter.convert(distance, positions, ids, id_bits, drop_bits);
        });
        const std::vector<TfheRecord> centroids(converted.begin() + n, converted.end());
        std::vector<std::vector<TfheRecord>> clusters(c);
        for (std::size_t cluster = 0; cluster < c; ++cluster)
            clusters[cluster].assign(converted.begin() + cluster * rows,
                                     converted.begin() + (cluster + 1) * rows);
        const auto selected = stage("centroid_topu", [&] { return server->topk(centroids, u); });
        const auto gathered = stage("gather", [&] { return server->gather_clusters(selected, clusters); });
        const auto buckets = stage("random_buckets", [&] {
            return server->random_buckets(gathered, b, bucket_seed);
        });
        const auto winners = stage("bucket_minima", [&] { return server->bucket_minima(buckets); });
        bucket_workers = server->tfhe_last_workers_used();
        bucket_peak = server->tfhe_last_peak_inflight();
        const auto nearest = stage("winner_topk", [&] { return server->topk(winners, k); });
        topk_workers = server->tfhe_last_workers_used();
        topk_peak = server->tfhe_last_peak_inflight();
        synchronize_gpu();
        pipeline_wall_ms = elapsed_ms(pipeline_start);
        std::cout << "HOST_CONCURRENCY bucket_workers=" << bucket_workers << " bucket_peak=" << bucket_peak
                  << " winner_topk_workers=" << topk_workers << " winner_topk_peak=" << topk_peak << '\n';
        if (bucket_workers < 2 || bucket_peak < 2 || topk_workers < 2 || topk_peak < 2)
            throw std::runtime_error("Expected host call concurrency was not exercised");

        // CLIENT DECRYPTION AND ORACLE: first inspect results only after the
        // final Top-k has completed. Intermediate checks are test diagnostics;
        // production response sanitization and ID-only output are not implemented.
        measure(result, Phase::decryption, "client_verification", [&] {
            using Plain = std::pair<std::uint64_t, std::uint64_t>;
            const Plain invalid{UINT64_MAX, UINT64_MAX};
            std::vector<std::uint64_t> distances(parameters.slots, 0);
            for (std::size_t i = 0; i < distances.size(); ++i)
                for (std::size_t d = 0; d < dimensions; ++d) {
                    const auto delta = std::int64_t(q[d]) - std::int64_t(columns[d][i]);
                    distances[i] += delta * delta;
                }
            if (bfv.decrypt_bfv_coefficients(distance) != distances)
                throw std::runtime_error("BFV distance differs from integer reference");
            const auto verify = [&](const std::vector<TfheRecord>& actual, const std::vector<Plain>& expected) {
                if (actual.size() != expected.size()) throw std::runtime_error("Wrong encrypted output length");
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    const auto value = tfhe.decrypt(actual[i]);
                    if (expected[i] == invalid) {
                        if (value.valid) throw std::runtime_error("Padding became a valid record");
                    } else if (!value.valid || Plain{value.score, value.id} != expected[i]) {
                        throw std::runtime_error("Encrypted pipeline differs from plaintext bucket oracle");
                    }
                }
            };
            std::vector<Plain> all, centroid_oracle, gather_oracle;
            for (std::size_t i = 0; i < converted_count; ++i) all.emplace_back(distances[i], ids[i]);
            verify(converted, all);
            centroid_oracle.assign(all.begin() + n, all.end());
            const auto decrypt = [&](const std::vector<TfheRecord>& records) {
                std::vector<he::PlainRecord> values;
                for (const auto& record : records) values.push_back(tfhe.decrypt(record));
                return values;
            };
            const auto chosen = decrypt(selected);
            require_distance_topk(chosen, centroid_oracle, u);
            for (const auto& centroid : chosen)
                for (std::size_t row = 0; row < rows; ++row)
                    gather_oracle.push_back(all[centroid.id * rows + row]);
            verify(gathered, gather_oracle);

            // The public random layout is reproduced independently from the
            // seed. The oracle checks distances and matching IDs with arbitrary ties.
            std::vector<std::size_t> permutation(gather_oracle.size());
            std::iota(permutation.begin(), permutation.end(), 0);
            std::mt19937_64 layout_random(bucket_seed);
            std::shuffle(permutation.begin(), permutation.end(), layout_random);
            const auto capacity = (gather_oracle.size() + b - 1) / b;
            std::vector<std::vector<Plain>> bucket_oracle(b, std::vector<Plain>(capacity, invalid));
            for (std::size_t i = 0; i < permutation.size(); ++i)
                bucket_oracle[i % b][i / b] = gather_oracle[permutation[i]];
            const auto plain_winners = decrypt(winners);
            if (buckets.size() != b || plain_winners.size() != b)
                throw std::runtime_error("Wrong public bucket count");
            for (std::size_t bucket = 0; bucket < b; ++bucket) {
                verify(buckets[bucket], bucket_oracle[bucket]);
                std::vector<he::PlainRecord> input;
                for (const auto& value : bucket_oracle[bucket])
                    input.push_back(value == invalid ? he::PlainRecord{0, 0, false} :
                        he::PlainRecord{static_cast<std::uint32_t>(value.first),
                                        static_cast<std::uint32_t>(value.second), true});
                require_distance_topk({plain_winners[bucket]}, input, 1);
            }
            const auto plain_nearest = decrypt(nearest);
            require_distance_topk(plain_nearest, plain_winners, k);
            std::vector<Plain> exact(all.begin(), all.begin() + n);
            std::stable_sort(exact.begin(), exact.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            exact.resize(k);
            std::size_t hits = 0;
            for (const auto& value : plain_nearest) {
                if (!value.valid) throw std::runtime_error("Fixture should have k valid bucket winners");
                output_ids.push_back(value.id);
                hits += std::find(exact.begin(), exact.end(), Plain{value.score, value.id}) != exact.end();
            }
            recall = static_cast<double>(hits) / k;
            std::cout << "VERIFIED converted=" << converted_count << " gathered=" << gathered.size()
                      << " buckets=" << buckets.size() << " capacity=" << capacity
                      << " winners=" << winners.size() << " final_ids=" << nearest.size()
                      << " recall_at_k=" << recall << "\nFINAL_IDS";
            for (auto id : output_ids) std::cout << ' ' << id;
            std::cout << '\n';
        });
        std::cout << "SERVER_STAGE_SUM_MS=" << result.evaluation_ms
                  << " SERVER_PIPELINE_WALL_MS=" << pipeline_wall_ms
                  << "\nSCOPE: through encrypted bucket-winner Top-k; no output sanitization or network transport.\n";
    });

    csv << "metric,value\n" << std::fixed << std::setprecision(3)
        << "status," << status_name(result.status) << '\n'
        << "records," << n << "\ndimensions," << dimensions << "\nclusters," << c
        << "\nselected_clusters," << u << "\nbuckets," << b << "\nk," << k
        << "\nscore_bits,23\nid_bits," << id_bits << "\ndistance_ciphertext_bits,32\nid_ciphertext_bits,32\ndrop_bits," << drop_bits
        << "\ndata_seed," << data_seed << "\nbucket_seed," << bucket_seed << '\n';
    for (const auto& [name, ms] : timings) csv << name << "_ms," << ms << '\n';
    csv << "server_stage_sum_ms," << result.evaluation_ms << "\nserver_pipeline_wall_ms," << pipeline_wall_ms
        << "\nsetup_ms," << result.setup_ms << "\nencryption_ms," << result.encryption_ms
        << "\nverification_ms," << result.decryption_ms << "\nprocess_wall_ms," << result.wall_ms
        << "\nbucket_workers," << bucket_workers << "\nbucket_peak_inflight," << bucket_peak
        << "\nwinner_topk_workers," << topk_workers << "\nwinner_topk_peak_inflight," << topk_peak
        << "\nrecall_at_k," << recall << "\noutput_sanitization_implemented,0\n";
    csv.close();
    if (!csv) { std::cerr << "Failed to write report\n"; return 1; }
    return exit_code(result);
}
