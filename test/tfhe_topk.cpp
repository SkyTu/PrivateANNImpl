#include "utils/check_support.hpp"
#include "utils/detail/tfhe_rs_api.hpp"

#if PRIVATEIMPL_HAS_GPU
#include "he/entity/client.hpp"
#include "utils/tfhe_crypto.hpp"
#include "he/entity/server.hpp"
#include "he/tfhe_select.hpp"
#include <algorithm>
#include <memory>
#include <vector>
#endif

namespace privateann::checks {
Result tfhe_topk() {
    return run("tfhe_topk", [](Result& result) {
        he::tfhe_rs::Api::instance();
        require_gpu();
#if PRIVATEIMPL_HAS_GPU
        // CLIENT: generate keys once; the entity also initializes its BFV context.
        std::cout << "Generating BFV/TFHE-rs client keys..." << std::endl;
        auto client = measure(result, Phase::setup, "BFV/TFHE client setup/key generation", [&] {
            return std::make_unique<entity::Client>(BfvParameters{}, true);
        });
        utils::TfheCrypto tfhe(client->tfhe_keys());
        const auto tfhe_key = client->tfhe_keys().evaluation_key();
        // SERVER: decompress the evaluation key to GPU without secret keys.
        std::cout << "Initializing TFHE-rs CUDA server key..." << std::endl;
        auto server = measure(result, Phase::setup, "TFHE CUDA server setup", [&] {
            return std::make_unique<entity::Server>(client->evaluation_keys());
        });

        // CLIENT: crypto retains its key; Server retains evaluation material only.
        client.reset();

        using Plain = std::pair<std::uint64_t, std::uint64_t>;
        auto encrypt = [&](const std::vector<Plain>& input, unsigned score_bits, unsigned id_bits) {
            // CLIENT: direct TFHE fixtures; these do not test BFV conversion.
            return measure(result, Phase::encryption, "TFHE fixture encryption", [&] {
                std::vector<TfheRecord> encrypted;
                for (auto [score, id] : input)
                    encrypted.push_back(tfhe.encrypt(score, id, score_bits, id_bits));
                return encrypted;
            });
        };
        auto verify = [&](const std::vector<TfheRecord>& encrypted,
                          const std::vector<Plain>& expected) {
            // CLIENT: decrypt after all server GPU work has completed.
            auto actual = measure(result, Phase::decryption, "TFHE verification decryption", [&] {
                std::vector<he::PlainRecord> values;
                for (const auto& record : encrypted) values.push_back(tfhe.decrypt(record));
                return values;
            });
            if (actual.size() != expected.size()) throw std::runtime_error("Wrong Top-k output count");
            for (std::size_t i = 0; i < actual.size(); ++i)
                if (!actual[i].valid || actual[i].score != expected[i].first || actual[i].id != expected[i].second)
                    throw std::runtime_error("Encrypted records differ from independent reference");
        };
        const auto verify_selection = [&](const std::vector<TfheRecord>& output,
                                           const std::vector<Plain>& input, std::size_t k) {
            return measure(result, Phase::decryption, "Distance-only selection verification", [&] {
                std::vector<he::PlainRecord> actual;
                for (const auto& record : output) actual.push_back(tfhe.decrypt(record));
                require_distance_topk(actual, input, k);
                return actual;
            });
        };
        auto evaluate = [&](const std::vector<TfheRecord>& input, std::size_t k,
                            const std::vector<Plain>& expected, const char* label) {
            std::cout << "Evaluating " << label << "..." << std::endl;
            auto output = measure(result, Phase::evaluation, label, [&] { return server->topk(input, k); });
            verify_selection(output, expected, k);
            std::cout << "PASS: " << label << std::endl;
        };

        const std::vector<Plain> fixture{{9, 2}, {4, 3}, {4, 1}, {0, 7}, {15, 15}};
        auto encrypted = encrypt(fixture, 4, 4);
        for (const auto& record : encrypted)
            if (record.distance_ciphertext_bits() != 32) throw std::runtime_error("Small distance must use FheUint32");
        evaluate(encrypted, 1, fixture, "TFHE Top-1 (n=5)");
        evaluate(encrypted, 3, fixture, "TFHE Top-3 (n=5)");
        evaluate(encrypted, 5, fixture, "TFHE Top-5 (n=5)");
        for (auto bad_k : {0U, 6U}) {
            bool rejected = false;
            try { server->topk(encrypted, bad_k); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Invalid k accepted");
        }
        evaluate({encrypted.back()}, 1, {fixture.back()}, "TFHE singleton");

        std::vector<Plain> centroids;
        for (unsigned i = 0; i < 17; ++i) centroids.emplace_back((i * 7) % 11, 16 - i);
        auto encrypted_centroids = encrypt(centroids, 4, 5);
        evaluate(encrypted_centroids, 16, centroids, "TFHE Top-u (n=17,u=16)");

        // Check that invalid padding is larger than every valid uint32 distance.
        const std::vector<Plain> boundaries{
            {UINT32_MAX, UINT32_MAX}, {0, UINT32_MAX}, {UINT32_MAX, 0}};
        auto encrypted_boundaries = encrypt(boundaries, 32, 32);
        for (const auto& record : encrypted_boundaries)
            if (record.distance_ciphertext_bits() != 64) throw std::runtime_error("Full uint32 distance needs a uint64 sentinel");
        evaluate(encrypted_boundaries, 3, boundaries, "TFHE uint32 boundaries");
        // Exercise unsigned bit-31 padding and ties in the 32-bit distance path.
        const std::vector<Plain> compact_boundaries{
            {(1u << 23) - 1, 255}, {7, 254}, {7, 0}, {0, 255}, {(1u << 23) - 1, 0}};
        auto compact = encrypt(compact_boundaries, 31, 8);
        evaluate(compact, 5, compact_boundaries, "TFHE uint32 distance with bit-31 padding");
        // The same key cannot make different public layouts comparable.
        bool mixed_rejected = false;
        try { server->topk({encrypted.front(), compact.front()}, 1); }
        catch (const std::invalid_argument&) { mixed_rejected = true; }
        if (!mixed_rejected) throw std::runtime_error("Mixed TFHE record layouts were accepted");
        verify(encrypted, fixture);
        verify(compact, compact_boundaries);
        std::cout << "PASS: arbitrary ties, distance/ID association, invalid padding, invalid k, and unchanged inputs\n";

        // CLIENT FIXTURE: three balanced clusters, with deliberately unrelated
        // record IDs. Copying a public cluster index into the output would fail.
        using Clusters = std::vector<std::vector<TfheRecord>>;
        const std::vector<std::vector<Plain>> cluster_values{
            {{9, 101}, {0, 105}}, {{4, 17}, {2, 9}}, {{2, 8}, {7, 2}}};
        Clusters clusters;
        for (const auto& values : cluster_values) clusters.push_back(encrypt(values, 4, 7));
        const std::vector<Plain> centroid_a{{7, 0}, {1, 1}, {1, 2}};
        const std::vector<Plain> centroid_b{{0, 0}, {6, 1}, {1, 2}};
        const auto encrypted_a = encrypt(centroid_a, 4, 2);
        const auto encrypted_b = encrypt(centroid_b, 4, 2);
        using Trace = he::tfhe_rs::GatherTrace;

        const auto check_gather = [&](const std::vector<TfheRecord>& centroid_cts,
            std::vector<Plain> centroid_values, const Clusters& source,
            const std::vector<std::vector<Plain>>& plain_source, std::size_t u,
            const char* label, Trace& trace) {
            // SERVER: Top-u -> secret gather -> candidate Top-k. No client
            // decryption, plaintext selector, or secret address between stages.
            const auto selected = measure(result, Phase::evaluation, "Gather centroid Top-u", [&] {
                return server->topk(centroid_cts, u);
            });
            const auto gathered = measure(result, Phase::evaluation, label, [&] {
                return server->gather_clusters(selected, source, &trace);
            });
            const auto nearest = measure(result, Phase::evaluation, "Gathered candidate Top-k", [&] {
                return server->topk(gathered, std::min<std::size_t>(3, gathered.size()));
            });

            // CLIENT DECRYPTION: inspect outputs only after the full server chain.
            const auto chosen = verify_selection(selected, centroid_values, u);
            // Any tied centroid may survive. Derive the expected routing from
            // its validated output only AFTER the entire encrypted chain ends.
            std::vector<Plain> expected;
            for (const auto& centroid : chosen)
                expected.insert(expected.end(), plain_source[centroid.id].begin(), plain_source[centroid.id].end());
            verify(gathered, expected);
            verify_selection(nearest, expected, nearest.size());
            for (std::size_t c = 0; c < source.size(); ++c) verify(source[c], plain_source[c]);
            if (gathered.size() != u * source.front().size())
                throw std::runtime_error("Secret gather changed its public output capacity");
            // Every source record must be read exactly once for each selected
            // output cluster, even when that source cluster is not selected.
            std::vector<std::size_t> visits(source.size() * source.front().size(), 0);
            std::vector<std::size_t> writes(gathered.size(), 0);
            for (const auto& access : trace.accesses) {
                if (access.cluster >= source.size() || access.row >= source.front().size() ||
                    access.output >= gathered.size()) throw std::runtime_error("Invalid gather trace");
                ++visits[access.cluster * source.front().size() + access.row];
                ++writes[access.output];
            }
            if (std::any_of(visits.begin(), visits.end(), [=](auto n) { return n != u; }) ||
                std::any_of(writes.begin(), writes.end(), [&](auto n) { return n != source.size(); }))
                throw std::runtime_error("Gather skipped an encrypted input or output update");
            std::cout << "PASS: " << label << " C=" << source.size() << " u=" << u
                      << " L=" << source.front().size() << " output=" << gathered.size()
                      << " logical_accesses=" << trace.accesses.size() << '\n';
        };
        Trace trace_a, trace_b, edge_trace;
        check_gather(encrypted_a, centroid_a, clusters, cluster_values, 2,
                     "TFHE gather compact A", trace_a);
        check_gather(encrypted_b, centroid_b, clusters, cluster_values, 2,
                     "TFHE gather compact B", trace_b);
        if (trace_a.accesses != trace_b.accesses)
            throw std::runtime_error("Secret cluster choices changed the logical access trace");
        check_gather(encrypted_b, centroid_b, clusters, cluster_values, 1,
                     "TFHE gather u=1", edge_trace);
        check_gather(encrypted_a, centroid_a, clusters, cluster_values, 3,
                     "TFHE gather u=C", edge_trace);

        // Widths are public and independent: uint64 centroid distances can select
        // compact records, and uint32 centroid distances can select records with 20-bit IDs.
        const auto wide_centroids = encrypt(centroid_a, 32, 32);
        check_gather(wide_centroids, centroid_a, clusters, cluster_values, 1,
                     "TFHE gather wide centroid", edge_trace);
        auto wide_values = cluster_values;
        wide_values[0][0] = {(1u << 23) - 1, (1u << 20) - 1};
        Clusters wide_clusters;
        for (const auto& values : wide_values) wide_clusters.push_back(encrypt(values, 23, 20));
        check_gather(encrypted_b, centroid_b, wide_clusters, wide_values, 1,
                     "TFHE gather wide records", edge_trace);

        // Invalid selectors remain encrypted and produce invalid output. In
        // particular padding must never alias cluster 0 through ID zero.
        he::tfhe_rs::Evaluator diagnostics(tfhe_key, 4);
        const auto tied = measure(result, Phase::evaluation, "Distance tie ignores larger ID", [&] {
            return diagnostics.minimum(encrypted[1], encrypted[2]);
        });
        verify({tied}, {fixture[1]});
        const auto invalid_selector = diagnostics.padding(wide_centroids.front());
        const auto unmatched_selector = encrypt({{0, 3}}, 4, 2).front();
        for (const auto& selector : {invalid_selector, unmatched_selector}) {
            const auto invalid = measure(result, Phase::evaluation, "TFHE gather invalid selector", [&] {
                return server->gather_clusters({selector}, clusters);
            });
            measure(result, Phase::decryption, "Invalid gather verification", [&] {
                for (const auto& record : invalid)
                    if (tfhe.decrypt(record).valid) throw std::runtime_error("Invalid selector gathered real data");
            });
        }

        // Public malformed shapes/layouts are rejected before secret selection.
        const auto rejects_gather = [](auto&& operation) {
            try { operation(); }
            catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("Malformed gather input was accepted");
        };
        rejects_gather([&] { server->gather_clusters({}, clusters); });
        rejects_gather([&] { server->gather_clusters(encrypted_a, {}); });
        rejects_gather([&] { server->gather_clusters(encrypted_a, {{}}); });
        rejects_gather([&] { server->gather_clusters(encrypted_a, {clusters[0]}); });
        auto malformed = clusters;
        malformed[1].pop_back();
        rejects_gather([&] { server->gather_clusters(encrypted_a, malformed); });
        malformed = clusters;
        malformed[1] = wide_clusters[1];
        rejects_gather([&] { server->gather_clusters(encrypted_a, malformed); });
        malformed = clusters;
        malformed[1][0] = {};
        rejects_gather([&] { server->gather_clusters(encrypted_a, malformed); });
        verify(encrypted_a, centroid_a);
        verify(encrypted_b, centroid_b);
        std::cout << "PASS: encrypted Top-u -> gather -> Top-k, ties, both widths, invalid selectors, "
                     "fixed output size, unchanged inputs and identical logical traces for different choices\n";

        // CLIENT FIXTURE: fixed public capacities; encrypted invalid records
        // represent missing entries. Buckets cover empty, singleton, tied and
        // full occupancy. Capacity five also exercises an unpaired survivor.
        const auto invalid = diagnostics.padding(encrypted.front());
        const Clusters buckets{
            {invalid, invalid, invalid, invalid, invalid},
            {encrypted[0], encrypted[1], invalid, encrypted[2], invalid},
            {invalid, invalid, encrypted[4], invalid, invalid},
            {encrypted[4], encrypted[2], encrypted[0], encrypted[3], encrypted[1]}};
        const auto verify_winners = [&](const std::vector<TfheRecord>& winners,
                                        const std::vector<he::PlainRecord>& expected) {
            measure(result, Phase::decryption, "Bucket winner verification", [&] {
                if (winners.size() != expected.size())
                    throw std::runtime_error("Bucket minima changed its public output count");
                for (std::size_t b = 0; b < winners.size(); ++b) {
                    const auto actual = tfhe.decrypt(winners[b]);
                    const auto& ref = expected[b];
                    if (actual.valid != ref.valid ||
                        (ref.valid && (actual.score != ref.score || actual.id != ref.id)))
                        throw std::runtime_error("Wrong encrypted bucket winner");
                }
            });
        };

        // SERVER: no plaintext occupancy information is supplied. Reuse the
        // existing Top-1 reduction and then the existing global Top-k network.
        const auto winners = measure(result, Phase::evaluation, "TFHE bucket minima B=4 capacity=5", [&] {
            return server->bucket_minima(buckets);
        });
        const auto nearest = measure(result, Phase::evaluation, "TFHE bucket winners Top-3", [&] {
            return server->topk(winners, 3);
        });
        // CLIENT: decrypt only after both server stages finish.
        measure(result, Phase::decryption, "Bucket ties and winner Top-k verification", [&] {
            std::vector<he::PlainRecord> plain_winners, plain_nearest;
            for (std::size_t b = 0; b < buckets.size(); ++b) {
                std::vector<he::PlainRecord> input;
                for (const auto& record : buckets[b]) input.push_back(tfhe.decrypt(record));
                plain_winners.push_back(tfhe.decrypt(winners[b]));
                require_distance_topk({plain_winners.back()}, input, 1);
            }
            for (const auto& record : nearest) plain_nearest.push_back(tfhe.decrypt(record));
            require_distance_topk(plain_nearest, plain_winners, 3);
        });
        verify(encrypted, fixture);
        verify_winners(buckets[0], std::vector<he::PlainRecord>(5, {0, 0, false}));
        verify_winners(buckets[1], {{9, 2, true}, {4, 3, true}, {0, 0, false},
                                    {4, 1, true}, {0, 0, false}});
        verify_winners(buckets[2], {{0, 0, false}, {0, 0, false}, {15, 15, true},
                                    {0, 0, false}, {0, 0, false}});
        verify(buckets[3], {fixture[4], fixture[2], fixture[0], fixture[3], fixture[1]});

        // Same public shape, different secret occupancy: keep all four outputs.
        const auto empty_winners = measure(result, Phase::evaluation, "TFHE all-invalid buckets B=4 capacity=5", [&] {
            return server->bucket_minima(Clusters(4, std::vector<TfheRecord>(5, invalid)));
        });
        verify_winners(empty_winners, std::vector<he::PlainRecord>(4, {0, 0, false}));

        // Two entries per bucket leave exactly one gate per bucket. Any
        // observed overlap here comes from different buckets sharing a layer.
        // Also verify the bit-32 distance sentinel exceeds UINT32_MAX pairs.
        const auto wide_invalid = diagnostics.padding(encrypted_boundaries.front());
        const Clusters wide_buckets{
            {wide_invalid, encrypted_boundaries[0]},
            {encrypted_boundaries[2], encrypted_boundaries[1]},
            {wide_invalid, wide_invalid}};
        const auto wide_winners = measure(result, Phase::evaluation, "TFHE wide bucket minima B=3 capacity=2", [&] {
            return diagnostics.bucket_minima(wide_buckets);
        });
        std::cout << "BUCKET_PARALLEL workers=" << diagnostics.last_workers_used()
                  << " peak_inflight=" << diagnostics.last_peak_inflight()
                  << " layers=1 distance_gt_calls=3 field_cmux_calls=6\n";
        verify_winners(wide_winners,
            {{UINT32_MAX, UINT32_MAX, true}, {0, UINT32_MAX, true}, {0, 0, false}});

        const auto singletons = measure(result, Phase::evaluation, "TFHE bucket minima capacity=1", [&] {
            return server->bucket_minima({{invalid}, {encrypted[3]}});
        });
        verify_winners(singletons, {{0, 0, false}, {0, 7, true}});

        // Bad public shapes/layouts are rejected before any GPU comparisons.
        const auto rejects_buckets = [&](const Clusters& input) {
            try { server->bucket_minima(input); }
            catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("Malformed bucket input was accepted");
        };
        rejects_buckets({});
        rejects_buckets({{}});
        rejects_buckets({{encrypted[0]}, {}});
        rejects_buckets({{encrypted[0]}, {encrypted[1], encrypted[2]}});
        rejects_buckets({{encrypted[0]}, {encrypted_boundaries[0]}});
        rejects_buckets({{encrypted[0]}, {TfheRecord{}}});
        std::cout << "PASS: parallel bucket minima -> Top-k, empty/full/singleton buckets, "
                     "non-power-of-two capacity, distance/ID ties, both widths and unchanged inputs\n";
#endif
    });
}
} // namespace privateann::checks

#ifndef PRIVATEIMPL_CHECK_NO_MAIN
int main() { return privateann::checks::exit_code(privateann::checks::tfhe_topk()); }
#endif
