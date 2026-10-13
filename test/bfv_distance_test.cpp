#include "utils/check_support.hpp"

#if PRIVATEIMPL_HAS_GPU
#include "he/entity/client.hpp"
#include "utils/bfv_crypto.hpp"
#include "he/entity/server.hpp"
#include <algorithm>
#include <memory>
#include <random>
#include <vector>
#endif

namespace privateann::checks {
Result bfv_distance() {
    return run("bfv_distance", [](Result& result) {
        require_gpu();
#if PRIVATEIMPL_HAS_GPU
        constexpr std::size_t record_count = 1'000'000;
        constexpr std::size_t dimensions = 128;
        BfvParameters parameters;
        // 128 unsigned byte coordinates can produce 128 * 255^2 = 8,323,200.
        // Use a plaintext modulus above this bound to prevent modular wraparound.
        parameters.validate_distance_dimension(dimensions);
        // CLIENT: create the context and generate keys once for all cases.
        auto client = measure(result, Phase::setup, "BFV client setup/key generation", [&] {
            return std::make_unique<entity::Client>(parameters);
        });
        auto bfv = measure(result, Phase::setup, "BFV crypto utilities setup", [&] {
            return std::make_unique<utils::BfvCrypto>(*client);
        });
        // SERVER: initialize evaluation objects using evaluation keys only.
        auto server = measure(result, Phase::setup, "BFV server setup", [&] {
            return std::make_unique<entity::Server>(client->evaluation_keys());
        });
        if (client->evaluation_keys().relin_key || client->evaluation_keys().rotation_key)
            throw std::runtime_error("Linear distance should not generate relin or rotation keys");
        // CLIENT: utilities keep their key material alive after the owner exits.
        client.reset();
        auto verify_query = [&](BfvQuery& query, const std::vector<std::uint64_t>& q) {
            std::vector<std::uint64_t> constant(parameters.slots, 0);
            std::uint64_t norm = 0;
            for (std::size_t d = 0; d < q.size(); ++d) {
                constant[0] = q[d];
                norm += q[d] * q[d];
                if (bfv->decrypt_bfv_coefficients(query.coordinates[d]) != constant)
                    throw std::runtime_error("Server changed an input query coordinate");
            }
            constant[0] = norm;
            if (bfv->decrypt_bfv_coefficients(query.squared_norm) != constant)
                throw std::runtime_error("Server changed the encrypted query norm");
        };

        // Reject malformed public inputs before doing server evaluation.
        auto rejects = [](auto&& call) {
            try { call(); }
            catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("Invalid BFV input was accepted");
        };
        rejects([&] { bfv->encrypt_bfv_query({}); });
        rejects([&] { bfv->encrypt_bfv_query({256}); });
        rejects([&] { bfv->encrypt_bfv_query(std::vector<std::uint64_t>(129, 0)); });
        auto valid_query = bfv->encrypt_bfv_query({5});
        rejects([&] { server->squared_l2(valid_query, {}); });
        rejects([&] { server->squared_l2(valid_query, {{3}}); });
        rejects([&] { server->squared_l2(valid_query,
                       {std::vector<std::uint64_t>(parameters.slots, 256)}); });
        std::cout << "PASS: query/record bounds, dimension guards, no relin or rotation keys\n";

        auto verify = [&](const std::vector<std::uint64_t>& q,
                          const std::vector<std::vector<std::uint64_t>>& records) {
            std::vector<std::uint64_t> expected(parameters.slots, 0);
            for (std::size_t d = 0; d < q.size(); ++d)
                for (std::size_t slot = 0; slot < parameters.slots; ++slot) {
                    const auto delta = std::int64_t(q[d]) - std::int64_t(records[d][slot]);
                    expected[slot] += delta * delta;
                }
            // CLIENT: encrypt q[d] and ||q||^2 before timing server evaluation.
            auto query = measure(result, Phase::encryption, "BFV query encryption", [&] {
                return bfv->encrypt_bfv_query(q);
            });
            // SERVER: ||q||^2 - 2*q.x + ||x||^2, using plaintext multiplication.
            auto distance = measure(result, Phase::evaluation, "BFV squared L2", [&] {
                return server->squared_l2(query, records);
            });
            // CLIENT: decrypt only after the evaluation timer has stopped.
            auto actual = measure(result, Phase::decryption, "BFV result decryption", [&] {
                return bfv->decrypt_bfv_coefficients(distance);
            });
            if (actual != expected)
                throw std::runtime_error("Squared L2 differs from the signed integer reference");
            measure(result, Phase::decryption, "BFV input verification decryption", [&] {
                verify_query(query, q);
            });
        };

        verify({5}, {std::vector<std::uint64_t>(parameters.slots, 3)});
        std::cout << "PASS: 5^2 - 2*5*3 + 3^2 = 4 in every BFV coefficient\n";
        for (unsigned trial = 0; trial < 3; ++trial) {
            std::vector<std::uint64_t> q;
            std::vector<std::vector<std::uint64_t>> records;
            for (unsigned d = 0; d < 3; ++d) {
                q.push_back(trial == 0 ? 0 : trial == 1 ? 255 : 17 + d);
                records.emplace_back(parameters.slots);
                for (std::size_t slot = 0; slot < parameters.slots; ++slot)
                    records.back()[slot] = (slot * 37 + d * 19) % 256;
            }
            verify(q, records);
        }
        std::cout << "PASS: entity reuse, 3D L2, negative differences, all coefficients, unchanged inputs\n";
        verify(std::vector<std::uint64_t>(dimensions, 0),
               std::vector<std::vector<std::uint64_t>>(
                   dimensions, std::vector<std::uint64_t>(parameters.slots, 255)));
        verify(std::vector<std::uint64_t>(dimensions, 255),
               std::vector<std::vector<std::uint64_t>>(
                   dimensions, std::vector<std::uint64_t>(parameters.slots, 0)));
        verify(std::vector<std::uint64_t>(dimensions, 255),
               std::vector<std::vector<std::uint64_t>>(
                   dimensions, std::vector<std::uint64_t>(parameters.slots, 255)));
        std::cout << "PASS: maximum 128D uint8 distance = 8,323,200; maximum norms cancel to zero\n";

        // SERVER DATA: one million synthetic 128D uint8 records, streamed in BFV batches.
        // Each polynomial coefficient holds one candidate; coordinates are 0..255.
        // The final batch is padded with zero coordinates. Padding is checked
        // but is never counted as a real record in the workload or throughput.
        const std::size_t batch_count = (record_count + parameters.slots - 1) / parameters.slots;
        result.distance_dimensions = dimensions;
        std::vector<std::uint64_t> q(dimensions);
        for (std::size_t d = 0; d < dimensions; ++d) q[d] = (d * 37 + 19) & 255;
        q[0] = 0;
        q[1] = 255;
        std::cout << "BFV workload: records=" << record_count << ", dimensions=" << dimensions
                  << ", coordinate_bits=8, plaintext_modulus=" << parameters.plaintext_modulus
                  << ", coefficients_per_batch=" << parameters.slots << ", batches=" << batch_count
                  << ", final_batch_records=" << (record_count - (batch_count - 1) * parameters.slots)
                  << std::endl;

        // CLIENT: encrypt this query only once, then reuse it across all batches.
        auto query = measure(result, Phase::encryption, "BFV 1M query encryption", [&] {
            return bfv->encrypt_bfv_query(q);
        });
        std::vector<std::vector<std::uint64_t>> records(
            dimensions, std::vector<std::uint64_t>(parameters.slots));
        // A fixed seed makes the public test dataset reproducible across runs.
        std::mt19937 generator(0x5a17c3U);
        for (std::size_t offset = 0; offset < record_count; offset += parameters.slots) {
            const std::size_t active = std::min(parameters.slots, record_count - offset);
            for (auto& column : records) {
                for (std::size_t slot = 0; slot < active; ++slot)
                    column[slot] = generator() & 255U;
                std::fill(column.begin() + active, column.end(), 0);
            }

            // SERVER: measure completed distance evaluation only. Dataset
            // generation and the client's verification are outside this timer.
            const double before = result.evaluation_ms;
            auto distance = measure(result, Phase::evaluation, "BFV 1M batch", [&] {
                return server->squared_l2(query, records);
            }, false);
            result.distance_evaluation_ms += result.evaluation_ms - before;
            ++result.distance_batches;

            // CLIENT: check every real coefficient, plus final-batch padding.
            auto actual = measure(result, Phase::decryption, "BFV 1M batch decryption", [&] {
                return bfv->decrypt_bfv_coefficients(distance);
            }, false);
            if (actual.size() != parameters.slots)
                throw std::runtime_error("Unexpected BFV output coefficient count");
            for (std::size_t slot = 0; slot < parameters.slots; ++slot) {
                std::uint64_t expected = 0;
                for (std::size_t d = 0; d < dimensions; ++d) {
                    const auto delta = std::int64_t(q[d]) - std::int64_t(records[d][slot]);
                    expected += delta * delta;
                }
                if (actual[slot] != expected)
                    throw std::runtime_error("BFV 1M distance mismatch at batch offset " +
                                             std::to_string(offset) + ", coefficient " + std::to_string(slot));
            }
            result.distance_records += active;
            if (result.distance_batches % 16 == 0 || result.distance_records == record_count)
                std::cout << "BFV verified " << result.distance_records << '/' << record_count
                          << " records (" << result.distance_batches << '/' << batch_count
                          << " batches)" << std::endl;
        }
        if (result.distance_records != record_count || result.distance_batches != batch_count)
            throw std::runtime_error("BFV workload did not cover exactly one million records");
        measure(result, Phase::decryption, "BFV 1M reused query verification", [&] {
            verify_query(query, q);
        });
        std::cout << "PASS: 1,000,000 BFV distances (128D uint8), final-batch padding, and reused query\n";
#endif
    });
}
} // namespace privateann::checks

#ifndef PRIVATEIMPL_CHECK_NO_MAIN
int main() { return privateann::checks::exit_code(privateann::checks::bfv_distance()); }
#endif
