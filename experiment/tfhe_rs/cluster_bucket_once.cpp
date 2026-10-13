#include "he/entity/client.hpp"
#include "he/entity/server.hpp"
#include "he/detail/topk_schedule.hpp"
#include "utils/bfv_crypto.hpp"
#include "utils/tfhe_crypto.hpp"
#include "utils/convert.hpp"
#include "utils/check_support.hpp"

#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

namespace {
using namespace privateann;
using namespace privateann::checks;

// Report public phase elapsed time during long operations. This observer never
// inspects a ciphertext and does not change the production network or scheduler.
class Progress {
    std::mutex mutex_;
    std::condition_variable changed_;
    bool stopped_ = false;
    std::thread thread_;
public:
    explicit Progress(const char* label) : thread_([this, label] {
        const auto start = Clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        while (!changed_.wait_for(lock, std::chrono::seconds(30), [&] { return stopped_; }))
            std::cout << "PROGRESS " << label << " elapsed_seconds="
                      << elapsed_ms(start) / 1000 << std::endl;
    }) {}
    ~Progress() {
        { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; }
        changed_.notify_one();
        thread_.join();
    }
};

void describe_network(const char* name, std::size_t n, std::size_t k, std::size_t copies = 1) {
    const auto schedule = he::topk_schedule(n, k);
    std::size_t full = 0, minimum = 0;
    for (const auto& layer : schedule.layers)
        for (const auto& gate : layer) gate.min_only ? ++minimum : ++full;
    std::cout << "NETWORK " << name << " layers=" << schedule.layers.size()
              << " minmax_gates=" << full * copies << " minimum_gates=" << minimum * copies
              << std::endl;
}
} // namespace

// A single measurement of the requested large components, NOT a continuous
// private ANN query. The old continuous experiment converts every record before
// TFHE gather. Selecting BFV records with TFHE selectors is still missing.
// Here candidate ciphertexts are prepared by the client from a PUBLIC fixture.
// Decrypted selectors are never used to route records or continue evaluation.
int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "Usage: cluster_bucket_once [OUTPUT_CSV]\n";
        return 1;
    }
    const std::filesystem::path report = argc == 2 ? argv[1] : "logs/cluster_bucket_once.csv";
    if (report.has_parent_path()) std::filesystem::create_directories(report.parent_path());
    std::ofstream csv(report);
    if (!csv) { std::cerr << "Cannot open report\n"; return 1; }
    csv << "metric,value\nfull_pipeline_executed,0\noblivious_bfv_gather_implemented,0\n"
           "candidate_conversion_executed,0\ncandidate_fixture_client_prepared,1\n";
    csv << std::fixed << std::setprecision(3);
    constexpr std::size_t n = 1000000, dimensions = 128, c = 500, rows = n / c;
    constexpr std::size_t u = 15, m = u * rows, b = 100, k = 10;
    constexpr unsigned score_bits = 23, id_bits = 20;
    constexpr std::uint64_t data_seed = 20261013, bucket_seed = 20261013;
    csv << "records," << n << "\ndimensions," << dimensions << "\nelement_bits,8\nclusters," << c
        << "\ncluster_capacity," << rows << "\nselected_clusters," << u << "\ncandidates," << m
        << "\nbuckets," << b << "\nbucket_capacity," << m / b << "\nk," << k
        << "\nscore_bits," << score_bits << "\nid_bits," << id_bits
        << "\ndistance_ciphertext_bits,32\nid_ciphertext_bits,32\ndrop_bits,0\ndata_seed,"
        << data_seed << "\nbucket_seed," << bucket_seed << '\n';
    csv.flush();

    const auto result = run("cluster_bucket_once", [&](Result& result) {
        require_gpu();
        std::cout << "CONFIG N=" << n << " dimensions=" << dimensions << " C=" << c
                  << " u=" << u << " candidates=" << m << " B=" << b << " capacity=" << m / b
                  << " k=" << k << " score_bits=23 id_bits=20 fields=32+32 drop_bits=0\n"
                     "SCOPE: component measurements; no oblivious BFV gather or candidate conversion.\n"
                     "Candidate TFHE inputs come from a public synthetic fixture, not encrypted Top-u.\n";
        describe_network("centroid_topu", c, u);
        describe_network("bucket_minima", m / b, 1, b);
        describe_network("winner_topk", b, k);
        const auto stage = [&](const char* name, auto&& operation) {
            Progress progress(name);
            const auto before = result.evaluation_ms;
            auto value = measure(result, Phase::evaluation, name, operation);
            csv << name << "_ms," << result.evaluation_ms - before << '\n';
            csv.flush();
            return value;
        };

        // PUBLIC TEST DATA: 500 supplied uint8 representative vectors and
        // 2,000 nearby records per cluster. This does not benchmark clustering.
        // Plaintext distances are independent integer oracles, not server input.
        BfvParameters parameters;
        parameters.validate_distance_dimension(dimensions);
        std::mt19937 random(data_seed);
        std::vector<std::uint64_t> q(dimensions);
        for (std::size_t d = 0; d < dimensions; ++d) q[d] = d % 2 ? 255 : 0;
        std::vector<std::vector<std::uint64_t>> centroid_columns(
            dimensions, std::vector<std::uint64_t>(parameters.slots, 0));
        std::vector<he::PlainRecord> centroid_oracle(c), all(n);
        for (std::size_t cluster = 0; cluster < c; ++cluster) {
            centroid_oracle[cluster] = {0, static_cast<std::uint32_t>(cluster), true};
            for (std::size_t d = 0; d < dimensions; ++d) {
                const auto value = random() & 255;
                centroid_columns[d][cluster] = value;
                const auto delta = std::int64_t(q[d]) - std::int64_t(value);
                centroid_oracle[cluster].score += delta * delta;
            }
        }
        std::vector<std::uint8_t> data(n * dimensions);
        for (std::size_t i = 0; i < n; ++i) {
            all[i] = {0, static_cast<std::uint32_t>(i), true};
            for (std::size_t d = 0; d < dimensions; ++d) {
                const auto value = std::clamp(static_cast<int>(centroid_columns[d][i / rows]) +
                    static_cast<int>(random() % 17) - 8, 0, 255);
                data[i * dimensions + d] = value;
                const auto delta = std::int64_t(q[d]) - value;
                all[i].score += delta * delta;
            }
        }
        // PUBLIC FIXTURE BOUNDARY: choose 15 fixture clusters before any server
        // computation. This is NOT an implementation of secret cluster routing.
        auto fixture_centroids = centroid_oracle;
        std::stable_sort(fixture_centroids.begin(), fixture_centroids.end(),
            [](const auto& a, const auto& b) { return a.score < b.score; });
        std::vector<he::PlainRecord> candidate_oracle;
        candidate_oracle.reserve(m);
        for (std::size_t j = 0; j < u; ++j) {
            const auto begin = all.begin() + fixture_centroids[j].id * rows;
            candidate_oracle.insert(candidate_oracle.end(), begin, begin + rows);
        }

        // CLIENT KEY GENERATION: only evaluation keys are shared with Server.
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

        // CLIENT ENCRYPTION: IDs are TFHE ciphertexts from the beginning.
        // Direct candidate encryption is outside server evaluation timings.
        auto query = measure(result, Phase::encryption, "query_encryption", [&] {
            return bfv.encrypt_bfv_query(q);
        });
        auto candidates = measure(result, Phase::encryption, "candidate_fixture_encryption", [&] {
            Progress progress("candidate_fixture_encryption");
            std::vector<TfheRecord> encrypted;
            encrypted.reserve(m);
            for (const auto& value : candidate_oracle)
                encrypted.push_back(tfhe.encrypt(value.score, value.id, score_bits, id_bits));
            return encrypted;
        });
        client.reset();

        // SERVER BFV DISTANCE: all 1M real records, 245 batches, one query.
        // Column preparation is reported separately; each distance call includes
        // the existing encoding, transfers, GPU arithmetic and synchronization.
        std::vector<BfvCiphertext> record_distances;
        const auto batches = (n + parameters.slots - 1) / parameters.slots;
        record_distances.reserve(batches);
        std::vector<std::vector<std::uint64_t>> columns(
            dimensions, std::vector<std::uint64_t>(parameters.slots, 0));
        const auto before_distance = result.evaluation_ms;
        double prepare_ms = 0;
        for (std::size_t offset = 0; offset < n; offset += parameters.slots) {
            const auto start = Clock::now();
            const auto count = std::min(parameters.slots, n - offset);
            for (std::size_t d = 0; d < dimensions; ++d) {
                std::fill(columns[d].begin(), columns[d].end(), 0);
                for (std::size_t i = 0; i < count; ++i)
                    columns[d][i] = data[(offset + i) * dimensions + d];
            }
            prepare_ms += elapsed_ms(start);
            record_distances.push_back(measure(result, Phase::evaluation, "bfv_batch", [&] {
                return server->squared_l2(query, columns);
            }, false));
        }
        const auto distance_ms = result.evaluation_ms - before_distance;
        csv << "bfv_records_distance_ms," << distance_ms << "\nbfv_column_preparation_ms," << prepare_ms
            << "\nbfv_record_batches," << batches << '\n';
        csv.flush();
        std::cout << "TIME bfv_records_distance = " << distance_ms << " ms; column_preparation = "
                  << prepare_ms << " ms; batches=" << batches << std::endl;

        // SERVER CENTROID COMPONENT: actual BFV -> TFHE -> Top-15, no decrypt.
        auto centroid_distance = stage("bfv_centroid_distance", [&] {
            return server->squared_l2(query, centroid_columns);
        });
        std::vector<std::size_t> positions(c);
        std::vector<std::uint64_t> centroid_ids(c);
        std::iota(positions.begin(), positions.end(), 0);
        std::iota(centroid_ids.begin(), centroid_ids.end(), 0);
        const auto converted = stage("centroid_conversion", [&] {
            return converter.convert(centroid_distance, positions, centroid_ids, id_bits, 0);
        });
        const auto selected = stage("centroid_topu", [&] { return server->topk(converted, u); });

        // SERVER CANDIDATE COMPONENT: reuse the real production bucket/min/Top-k
        // functions on prepared encrypted inputs. The preceding encrypted
        // selectors do not feed this component: that missing edge is explicit.
        const auto buckets = stage("random_buckets", [&] {
            return server->random_buckets(candidates, b, bucket_seed);
        });
        const auto winners = stage("bucket_minima", [&] { return server->bucket_minima(buckets); });
        csv << "bucket_workers," << server->tfhe_last_workers_used()
            << "\nbucket_peak_inflight," << server->tfhe_last_peak_inflight() << '\n';
        const auto nearest = stage("winner_topk", [&] { return server->topk(winners, k); });
        csv << "winner_topk_workers," << server->tfhe_last_workers_used()
            << "\nwinner_topk_peak_inflight," << server->tfhe_last_peak_inflight() << '\n';

        // CLIENT DECRYPTION / VERIFICATION: run only after all server work.
        // Verify all BFV distances, every converted centroid, every bucket
        // winner and the final Top-10, preserving distance/ID associations.
        measure(result, Phase::decryption, "client_verification", [&] {
            const auto decode = [&](const std::vector<TfheRecord>& records) {
                std::vector<he::PlainRecord> plain;
                for (const auto& value : records) plain.push_back(tfhe.decrypt(value));
                return plain;
            };
            std::uint64_t padding_distance = 0;
            for (const auto value : q) padding_distance += value * value;
            for (std::size_t batch = 0; batch < record_distances.size(); ++batch) {
                const auto plain = bfv.decrypt_bfv_coefficients(record_distances[batch]);
                for (std::size_t i = 0; i < parameters.slots; ++i) {
                    const auto index = batch * parameters.slots + i;
                    const auto expected = index < n ? all[index].score : padding_distance;
                    if (plain.at(i) != expected) throw std::runtime_error("BFV distance mismatch");
                }
            }
            const auto centroid_plain = decode(converted);
            for (std::size_t i = 0; i < c; ++i)
                if (!centroid_plain[i].valid || centroid_plain[i].score != centroid_oracle[i].score ||
                    centroid_plain[i].id != centroid_oracle[i].id)
                    throw std::runtime_error("Centroid conversion mismatch");
            require_distance_topk(decode(selected), centroid_oracle, u);
            std::vector<std::size_t> permutation(m);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::mt19937_64 random_layout(bucket_seed);
            std::shuffle(permutation.begin(), permutation.end(), random_layout);
            std::vector<std::vector<he::PlainRecord>> bucket_oracle(b);
            for (std::size_t i = 0; i < m; ++i)
                bucket_oracle[i % b].push_back(candidate_oracle[permutation[i]]);
            const auto plain_winners = decode(winners), plain_nearest = decode(nearest);
            if (plain_winners.size() != b) throw std::runtime_error("Wrong bucket count");
            for (std::size_t j = 0; j < b; ++j)
                require_distance_topk({plain_winners[j]}, bucket_oracle[j], 1);
            require_distance_topk(plain_nearest, plain_winners, k);

            // Distance-based recall accepts arbitrary IDs at a tied cutoff.
            // It measures bucket loss only on this synthetic candidate fixture.
            auto sorted = candidate_oracle;
            std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
                return a.score < b.score;
            });
            const auto cutoff = sorted[k - 1].score;
            std::size_t hits = 0;
            std::cout << "FINAL_IDS";
            for (const auto& value : plain_nearest) {
                hits += value.valid && value.score <= cutoff;
                std::cout << ' ' << value.id;
            }
            const double recall = static_cast<double>(hits) / k;
            std::cout << "\nVERIFIED bfv_records=" << n << " converted_centroids=" << c
                      << " selected_centroids=" << u << " bucket_winners=" << b << " final_k=" << k
                      << " candidate_distance_recall_at_10=" << recall << std::endl;
            csv << "candidate_distance_recall_at_10," << recall << '\n';
        });
        std::cout << "MEASURED_COMPONENT_SUM_MS=" << result.evaluation_ms
                  << " (excludes oblivious BFV gather and candidate conversion; not full query latency)\n";
    });
    csv << "status," << status_name(result.status) << "\nmeasured_component_sum_ms," << result.evaluation_ms
        << "\nsetup_ms," << result.setup_ms << "\nclient_encryption_ms," << result.encryption_ms
        << "\nclient_verification_ms," << result.decryption_ms << "\nprocess_wall_ms," << result.wall_ms << '\n';
    csv.close();
    if (!csv) { std::cerr << "Failed to write report\n"; return 1; }
    return exit_code(result);
}
