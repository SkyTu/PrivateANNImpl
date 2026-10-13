#include "utils/check_support.hpp"

#if PRIVATEIMPL_HAS_GPU
#include "he/entity/client.hpp"
#include "utils/tfhe_crypto.hpp"
#include "utils/bfv_crypto.hpp"
#include "he/entity/server.hpp"
#include "utils/conversion/bfv_lwe.hpp"
#include "utils/convert.hpp"
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#endif

namespace privateann::checks {
Result bfc_tfhe_conversion(std::size_t count, unsigned drop_bits) {
    return run("bfv_to_tfhe_conversion", [=](Result& result) {
        if ((count != 11 && count != 500) || (drop_bits != 0 && drop_bits != 7))
            throw std::invalid_argument("Use count=11 or 500 and drop_bits=0 or 7");
        require_gpu();
#if PRIVATEIMPL_HAS_GPU
        constexpr std::size_t dimensions = 128;
        BfvParameters parameters;
        parameters.validate_distance_dimension(dimensions);
        const auto n = parameters.slots;
        // CLIENT KEY GENERATION: coefficient extraction needs neither rotation
        // nor relin keys. Client creates cross-scheme evaluation material from
        // both secrets; Server receives only that evaluation material.
        auto client = measure(result, Phase::setup, "BFV extraction client keys", [&] {
            return std::make_unique<entity::Client>(parameters, true);
        });
        measure(result, Phase::setup, "BFV-to-TFHE evaluation key generation", [&] {
            client->enable_bfv_tfhe_conversion();
        });
        auto bfv = measure(result, Phase::setup, "BFV crypto utilities setup", [&] {
            return std::make_unique<utils::BfvCrypto>(*client);
        });
        utils::TfheCrypto tfhe(client->tfhe_keys());
        const auto keys = client->evaluation_keys();
        if (keys.relin_key || keys.rotation_key)
            throw std::runtime_error("Coefficient extraction should need no auxiliary BFV keys");
        auto server = measure(result, Phase::setup, "BFV/TFHE server setup", [&] {
            return std::make_unique<entity::Server>(keys);
        });
        utils::BfvLweExtractor extractor(keys);
        utils::BfvToTfheConverter converter(keys);

        // PUBLIC TEST FIXTURE: real 128D uint8 vectors with a nonzero query.
        // Boundary distances are constructed as sums of squared differences;
        // the independent reference below always uses (q[d]-x[d])^2.
        std::vector<std::uint64_t> q(dimensions), expected(n, 0);
        std::vector<std::vector<std::uint64_t>> records(dimensions, std::vector<std::uint64_t>(n));
        std::mt19937 random(20261012);
        for (std::size_t d = 0; d < dimensions; ++d) {
            q[d] = (d & 1) ? 255 : 0;
            for (auto& x : records[d]) x = random() & 255;
        }
        const std::vector<std::uint64_t> boundaries{
            0, 1, 8323200, 127, 128, 129, 65535, 65536, 255, 256, 257};
        for (std::size_t i = 0; i < boundaries.size(); ++i) {
            auto remaining = boundaries[i];
            for (std::size_t d = 0; d < dimensions; ++d) {
                std::uint64_t delta = 255;
                while (delta * delta > remaining) --delta;
                remaining -= delta * delta;
                records[d][i] = q[d] ? 255 - delta : delta;
            }
            if (remaining) throw std::runtime_error("Invalid boundary fixture");
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t d = 0; d < dimensions; ++d) {
                const auto delta = std::int64_t(q[d]) - std::int64_t(records[d][i]);
                expected[i] += delta * delta;
            }
        for (std::size_t i = 0; i < boundaries.size(); ++i)
            if (expected[i] != boundaries[i]) throw std::runtime_error("Boundary fixture mismatch");

        // CLIENT ENCRYPTION: 128 constant coordinates plus their encrypted norm.
        auto query = measure(result, Phase::encryption, "BFV conversion query encryption", [&] {
            return bfv->encrypt_bfv_query(q);
        });
        // SERVER EVALUATION: use the same distance function as the 1M workload.
        double before = result.evaluation_ms;
        auto distance = measure(result, Phase::evaluation, "BFV conversion source squared L2", [&] {
            return server->squared_l2(query, records);
        });
        result.conversion_source_ms = result.evaluation_ms - before;
        // CLIENT DIAGNOSTIC: verify the source independently of LWE extraction.
        measure(result, Phase::decryption, "BFV source verification", [&] {
            if (bfv->decrypt_bfv_coefficients(distance) != expected)
                throw std::runtime_error("BFV source distance is incorrect");
            std::cout << "SOURCE coefficient distance: N=" << n
                      << " t=" << parameters.plaintext_modulus
                      << " correct=" << n << '/' << n
                      << " library_noise_budget_bits=" << bfv->bfv_noise_budget(distance) << '\n';
        });

        // SERVER EVALUATION: extract 500 candidate distances, covering index 0,
        // both sides of the ring midpoint and the last coefficient. No mask,
        // rotation, decryption or client re-encryption is part of extraction.
        std::vector<std::size_t> positions;
        for (std::size_t i = 0; i < 497; ++i) positions.push_back(i);
        positions.insert(positions.end(), {n / 2 - 1, n / 2, n - 1});
        positions.resize(count);
        auto samples = measure(result, Phase::evaluation, "BFV coefficients to torus LWE diagnostic", [&] {
            std::vector<utils::BfvLweSample> out;
            out.reserve(positions.size());
            for (auto index : positions) out.push_back(extractor.extract_coefficient(distance, index));
            return out;
        });
        // CLIENT DECRYPTION: check results only after server evaluation ends.
        // The >>7 check is diagnostic plaintext arithmetic, not encrypted bit
        // truncation. No decoded value feeds back into the server or TFHE-rs.
        measure(result, Phase::decryption, "LWE verification decryptions", [&] {
            std::size_t exact = 0, high16 = 0;
            for (std::size_t j = 0; j < positions.size(); ++j) {
                const auto decoded = bfv->decrypt_bfv_lwe(samples[j]);
                const auto want = expected[positions[j]];
                exact += decoded == want;
                high16 += (decoded >> 7) == (want >> 7);
                if (j < boundaries.size())
                    std::cout << "BOUNDARY coefficient=" << positions[j] << " expected=" << want
                              << " actual=" << decoded << " high16=" << (decoded >> 7) << '\n';
            }
            std::cout << "EXTRACTION_SUMMARY exact=" << exact << '/' << count << " high16=" << high16
                      << '/' << count << " no_rotation_key=true no_relin_key=true\n";
            if (exact != count || high16 != count)
                throw std::runtime_error("Coefficient-to-LWE extraction failed");
            if (bfv->decrypt_bfv_coefficients(distance) != expected)
                throw std::runtime_error("Extraction modified the source distance");
        });
        // SERVER CONVERSION: the real public entry point performs extraction,
        // key switching, encrypted message decomposition, modulus correction and
        // native radix assembly. It accepts no plaintext distance or client key.
        std::vector<std::uint64_t> ids(count);
        std::iota(ids.begin(), ids.end(), 0);
        constexpr unsigned id_bits = 9;
        std::cout << "CONVERSION count=" << count << " drop_bits=" << drop_bits
                  << " score_bits=" << 23-drop_bits << " id_bits=" << id_bits << std::endl;
        before = result.evaluation_ms;
        auto converted = measure(result, Phase::evaluation, "Full BFV-to-native-TFHE conversion", [&] {
            return converter.convert(distance, positions, ids, id_bits, drop_bits);
        });
        result.conversion_evaluation_ms = result.evaluation_ms - before;
        result.conversion_records = count;
        result.conversion_drop_bits = drop_bits;
        // CLIENT DECRYPTION: the first observation of converted values happens
        // after the conversion utility has completed. It cannot repair conversion.
        measure(result, Phase::decryption, "Native TFHE conversion verification", [&] {
            if (converted.size() != count) throw std::runtime_error("Conversion result count mismatch");
            for (std::size_t i = 0; i < count; ++i) {
                const auto actual = tfhe.decrypt(converted[i]);
                if (!actual.valid || actual.score != (expected[positions[i]] >> drop_bits) || actual.id != ids[i])
                    throw std::runtime_error("Native TFHE conversion mismatch at ID " + std::to_string(ids[i]) +
                        ": expected score=" + std::to_string(expected[positions[i]] >> drop_bits) +
                        ", actual score=" + std::to_string(actual.score));
            }
        });
        std::cout << "PASS: native TFHE-rs conversion " << count << '/' << count
                  << " distance_bits=" << converted.front().distance_ciphertext_bits()
                  << " id_bits=" << converted.front().id_ciphertext_bits() << std::endl;

        // SERVER INTEROPERABILITY: consume converted ciphertexts with the same
        // GPU Top-k as ordinary native inputs. Eleven boundary records suffice
        // to check comparisons, ties and IDs without a second large benchmark.
        std::vector<TfheRecord> subset(converted.begin(), converted.begin()+11);
        before = result.evaluation_ms;
        auto selected = measure(result, Phase::evaluation, "Converted native GPU Top-3 of 11", [&] {
            return server->topk(subset, 3);
        });
        result.conversion_topk_ms = result.evaluation_ms - before;
        measure(result, Phase::decryption, "Converted Top-k verification", [&] {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> oracle;
            for (std::size_t i = 0; i < 11; ++i)
                oracle.emplace_back(expected[positions[i]] >> drop_bits, ids[i]);
            std::vector<he::PlainRecord> actual;
            for (const auto& record : selected) actual.push_back(tfhe.decrypt(record));
            require_distance_topk(actual, oracle, 3);
            if (bfv->decrypt_bfv_coefficients(distance) != expected)
                throw std::runtime_error("Conversion changed its BFV source");
        });
        std::cout << "PASS: converted ciphertexts consumed by native GPU Top-k\n";

        // SERVER ROUTING: use three converted distances as centroid records
        // (IDs 0,1,2), and the next six as three balanced candidate clusters.
        // All inputs came from BFV evaluation and real ciphertext conversion.
        // No plaintext selector or client decryption occurs between these calls.
        const std::vector<TfheRecord> centroids(converted.begin(), converted.begin() + 3);
        const std::vector<std::vector<TfheRecord>> clusters{
            {converted[3], converted[4]}, {converted[5], converted[6]}, {converted[7], converted[8]}};
        before = result.evaluation_ms;
        const auto selected_centroids = measure(result, Phase::evaluation, "Converted centroid Top-2", [&] {
            return server->topk(centroids, 2);
        });
        const auto gathered = measure(result, Phase::evaluation, "Converted TFHE cluster gather", [&] {
            return server->gather_clusters(selected_centroids, clusters);
        });
        const auto nearest = measure(result, Phase::evaluation, "Converted gathered candidate Top-3", [&] {
            return server->topk(gathered, 3);
        });
        std::cout << "CONVERTED_ROUTING C=3 u=2 L=2 candidates=" << gathered.size()
                  << " topu_gather_topk_ms=" << result.evaluation_ms - before << '\n';
        // CLIENT VERIFICATION: allow any tied centroid, then check its exact
        // gathered records. Decryption occurs only after all server operations.
        measure(result, Phase::decryption, "Converted cluster routing verification", [&] {
            using Plain = std::pair<std::uint64_t, std::uint64_t>;
            std::vector<Plain> centroid_oracle, gather_oracle;
            for (std::size_t c = 0; c < 3; ++c)
                centroid_oracle.emplace_back(expected[positions[c]] >> drop_bits, c);
            std::vector<he::PlainRecord> chosen;
            for (const auto& record : selected_centroids) chosen.push_back(tfhe.decrypt(record));
            require_distance_topk(chosen, centroid_oracle, 2);
            for (const auto& centroid : chosen)
                for (std::size_t r = 0; r < 2; ++r) {
                    const auto index = 3 + 2 * centroid.id + r;
                    gather_oracle.emplace_back(expected[positions[index]] >> drop_bits, ids[index]);
                }
            const auto check = [&](const std::vector<TfheRecord>& actual, const std::vector<Plain>& oracle) {
                if (actual.size() != oracle.size()) throw std::runtime_error("Converted gather output size mismatch");
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    const auto value = tfhe.decrypt(actual[i]);
                    if (!value.valid || value.score != oracle[i].first || value.id != oracle[i].second)
                        throw std::runtime_error("Converted gather score/ID differs from independent oracle");
                }
            };
            check(gathered, gather_oracle);
            std::vector<he::PlainRecord> nearest_plain;
            for (const auto& record : nearest) nearest_plain.push_back(tfhe.decrypt(record));
            require_distance_topk(nearest_plain, gather_oracle, 3);
            if (bfv->decrypt_bfv_coefficients(distance) != expected)
                throw std::runtime_error("Cluster routing changed the BFV source");
        });
        std::cout << "PASS: BFV distances -> TFHE conversion -> centroid Top-u -> secret gather -> candidate Top-k\n";
#endif
    });
}
} // namespace privateann::checks

#ifndef PRIVATEIMPL_CHECK_NO_MAIN
int main(int argc, char** argv) {
    std::size_t count = 500;
    unsigned drop_bits = 7; // Explicit CPU-reference semantics for this fixture.
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--smoke") count = 11;
        else if (arg == "--full-precision") drop_bits = 0;
        else { std::cerr << "Usage: bfc_tfhe_conversion [--smoke] [--full-precision]\n"; return 1; }
    }
    return privateann::checks::exit_code(privateann::checks::bfc_tfhe_conversion(count, drop_bits));
}
#endif
