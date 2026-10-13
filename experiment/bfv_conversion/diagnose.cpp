#include "he/entity/client.hpp"
#include "utils/bfv_crypto.hpp"
#include "he/entity/server.hpp"
#include "utils/conversion/bfv_lwe.hpp"
#include "utils/check_support.hpp"
#include <gmpxx.h>
#include <algorithm>
#include <random>

// Diagnostic harness, not a server protocol: the CLIENT checks intermediate
// ciphertexts after each completed operation. No decoded value is used to
// construct or repair a later server ciphertext.
using namespace privateann;
using namespace privateann::checks;

namespace {
using Words = std::vector<std::uint64_t>;

void trace_isolation(utils::BfvCrypto& crypto, const EvaluationKeys& keys,
                     BfvCiphertext& source, const Words& source_expected,
                     std::size_t slot, const char* label) {
    const auto n = source_expected.size();
    const std::uint64_t t = keys.bfv_parameters.plaintext_modulus;
    heongpu::HEEncoder<bfv_scheme> encoder(keys.bfv_context);
    heongpu::HEArithmeticOperator<bfv_scheme> evaluator(keys.bfv_context, encoder);
    // This reference tracks plaintext slots independently of encrypted outputs.
    auto inspect = [&](BfvCiphertext& ct, const Words& expected, const std::string& stage) {
        const auto values = crypto.decrypt_bfv_slots(ct);
        std::size_t matched = 0, first_bad = n;
        for (std::size_t i = 0; i < n; ++i) {
            matched += values[i] == expected[i];
            if (values[i] != expected[i] && first_bad == n) first_bad = i;
        }
        std::cout << "TRACE " << label << " stage=" << stage
                  << " correct=" << matched << '/' << n
                  << " library_noise_bits=" << crypto.bfv_noise_budget(ct);
        if (first_bad != n)
            std::cout << " first_bad_slot=" << first_bad << " expected=" << expected[first_bad]
                      << " actual=" << values[first_bad];
        std::cout << '\n';
    };
    inspect(source, source_expected, "source");
    Words mask(n, 0), expected(n, 0);
    mask[slot] = 1;
    expected[slot] = source_expected[slot];
    heongpu::Plaintext<bfv_scheme> plain(keys.bfv_context);
    encoder.encode(plain, mask);
    Words coefficients(n);
    cuda_check(cudaMemcpy(coefficients.data(), plain.data(), n * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost), "read public mask coefficients");
    std::uint64_t maximum = 0;
    for (auto value : coefficients) maximum = std::max(maximum, std::min(value, t - value));
    std::cout << "MASK " << label << " slot=" << slot
              << " max_abs_centered_coefficient=" << maximum << '\n';
    BfvCiphertext current(keys.bfv_context), rotated(keys.bfv_context);
    evaluator.multiply_plain(source, plain, current);
    inspect(current, expected, "multiply_plain_mask");
    for (std::size_t shift = 1; shift < n / 2; shift *= 2) {
        Words rotated_expected(n);
        for (std::size_t i = 0; i < n; ++i)
            rotated_expected[i] = expected[(i / (n / 2)) * (n / 2) + (i + shift) % (n / 2)];
        evaluator.rotate_rows(current, rotated, *keys.rotation_key, static_cast<int>(shift));
        inspect(rotated, rotated_expected, "rotate_" + std::to_string(shift));
        evaluator.add_inplace(current, rotated);
        for (std::size_t i = 0; i < n; ++i) expected[i] = (expected[i] + rotated_expected[i]) % t;
        inspect(current, expected, "add_" + std::to_string(shift));
    }
    Words rotated_expected(n);
    for (std::size_t i = 0; i < n; ++i) rotated_expected[i] = expected[(i + n / 2) % n];
    evaluator.rotate_columns(current, rotated, *keys.rotation_key);
    inspect(rotated, rotated_expected, "swap_rows");
    evaluator.add_inplace(current, rotated);
    for (std::size_t i = 0; i < n; ++i) expected[i] = (expected[i] + rotated_expected[i]) % t;
    inspect(current, expected, "final_sum");
    utils::BfvLweExtractor extractor(keys);
    auto actual = extractor.isolate_slot(source, slot);
    // The diagnostic trace must reproduce the production extractor exactly.
    Words traced_words, production_words;
    current.get_data(traced_words);
    actual.get_data(production_words);
    if (traced_words != production_words) throw std::runtime_error("Trace differs from production extraction");
}

// Historical CRT paths retained only to reproduce the original noise failure.
// Production distance uses Server::squared_l2 and coefficient packing below.
BfvCiphertext square_crt_distance(const EvaluationKeys& keys, BfvQuery& query,
                                  const std::vector<Words>& records) {
    heongpu::HEEncoder<bfv_scheme> encoder(keys.bfv_context);
    heongpu::HEArithmeticOperator<bfv_scheme> evaluator(keys.bfv_context, encoder);
    BfvCiphertext total(keys.bfv_context);
    for (std::size_t d = 0; d < query.coordinates.size(); ++d) {
        heongpu::Plaintext<bfv_scheme> plain(keys.bfv_context);
        encoder.encode(plain, records[d]);
        BfvCiphertext delta(keys.bfv_context), square(keys.bfv_context);
        evaluator.sub_plain(query.coordinates[d], plain, delta);
        evaluator.multiply(delta, delta, square);
        evaluator.relinearize_inplace(square, *keys.relin_key);
        if (d == 0) total = std::move(square);
        else evaluator.add_inplace(total, square);
    }
    return total;
}

BfvCiphertext linear_crt_distance(const EvaluationKeys& keys, BfvQuery& query,
                                  const std::vector<Words>& records) {
    const auto n = keys.bfv_parameters.slots;
    const std::uint64_t t = keys.bfv_parameters.plaintext_modulus;
    heongpu::HEEncoder<bfv_scheme> encoder(keys.bfv_context);
    heongpu::HEArithmeticOperator<bfv_scheme> evaluator(keys.bfv_context, encoder);
    auto total = query.squared_norm;
    Words column(n), norms(n, 0);
    for (std::size_t d = 0; d < query.coordinates.size(); ++d) {
        for (std::size_t i = 0; i < n; ++i) {
            column[i] = (t - 2 * records[d][i]) % t;
            norms[i] += records[d][i] * records[d][i];
        }
        heongpu::Plaintext<bfv_scheme> plain(keys.bfv_context);
        encoder.encode(plain, column);
        BfvCiphertext term(keys.bfv_context);
        evaluator.multiply_plain(query.coordinates[d], plain, term);
        evaluator.add_inplace(total, term);
    }
    heongpu::Plaintext<bfv_scheme> norm_plain(keys.bfv_context);
    encoder.encode(norm_plain, norms);
    evaluator.add_plain_inplace(total, norm_plain);
    return total;
}
} // namespace

int main(int argc, char** argv) try {
    const bool wide = argc == 2 && std::string(argv[1]) == "--wide-control";
    if (argc != 1 && !wide) throw std::invalid_argument("Usage: bfv_conversion_diagnose [--wide-control]");
    require_gpu();
    BfvParameters parameters;
    if (wide) { parameters.slots = 8192; parameters.plaintext_modulus = 8404993; }
    parameters.validate_distance_dimension(128);
    entity::Client client(parameters);
    client.enable_bfv_lwe_extraction();
    client.enable_bfv_relinearization();
    utils::BfvCrypto crypto(client);
    const auto keys = client.evaluation_keys();
    const auto n = parameters.slots;
    mpz_class Q = 1, QP = 1;
    const auto primes = keys.bfv_context->get_key_modulus();
    for (std::size_t i = 0; i < primes.size(); ++i) {
        QP *= primes[i].value;
        if (i < static_cast<std::size_t>(keys.bfv_context->get_ciphertext_modulus_count())) Q *= primes[i].value;
    }
    std::cout << "PARAMETERS N=" << n << " t=" << parameters.plaintext_modulus
              << " ciphertext_Q_bits=" << mpz_sizeinbase(Q.get_mpz_t(), 2)
              << " key_QP_bits=" << mpz_sizeinbase(QP.get_mpz_t(), 2) << '\n';
    constexpr std::size_t dimensions = 128;
    Words q(dimensions), expected(n, 0), fresh_values(n);
    std::vector<Words> records(dimensions, Words(n));
    std::mt19937 random(20261012);
    std::uint64_t norm = 0;
    for (std::size_t d = 0; d < dimensions; ++d) {
        q[d] = (d & 1) ? 255 : 0;
        norm += q[d] * q[d];
        for (std::size_t i = 0; i < n; ++i) records[d][i] = random() & 255;
    }
    // Real distance vectors with arbitrary boundary values, even with a nonzero
    // query: choose each |q[d]-x[d]| by a sum-of-squares decomposition.
    const Words boundaries{0, 1, 8323200, 127, 128, 129, 65535, 65536, 255, 256, 257};
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        auto remaining = boundaries[i];
        for (std::size_t d = 0; d < dimensions; ++d) {
            std::uint64_t difference = 255;
            while (difference * difference > remaining) --difference;
            remaining -= difference * difference;
            records[d][i] = q[d] ? 255 - difference : difference;
        }
        if (remaining) throw std::runtime_error("Boundary fixture needs more dimensions");
    }
    for (std::size_t i = 0; i < n; ++i) {
        fresh_values[i] = (i * 37 + 19) & 255;
        for (std::size_t d = 0; d < dimensions; ++d) {
            const auto delta = std::int64_t(q[d]) - std::int64_t(records[d][i]);
            expected[i] += delta * delta;
        }
    }
    // CLIENT: q coordinates are constant polynomials; this is also exactly the
    // plaintext produced by batching a repeated query coordinate in all slots.
    auto query = crypto.encrypt_bfv_query(q);
    auto fresh = crypto.encrypt_bfv_slots(fresh_values);
    trace_isolation(crypto, keys, fresh, fresh_values, 2, "fresh_crt");
    entity::Server server(keys);
    auto squared = square_crt_distance(keys, query, records);
    if (crypto.decrypt_bfv_slots(squared) != expected) throw std::runtime_error("Squared distance failed");
    trace_isolation(crypto, keys, squared, expected, 2, "square_crt");
    auto linear = linear_crt_distance(keys, query, records);
    if (crypto.decrypt_bfv_slots(linear) != expected) throw std::runtime_error("Linear CRT distance failed");
    trace_isolation(crypto, keys, linear, expected, 2, "linear_crt");

    Result timing;
    timing.name = "coefficient_distance_and_lwe_only";
    auto coefficient = measure(timing, Phase::evaluation, "Coefficient-packed BFV distance", [&] {
        return server.squared_l2(query, records);
    });
    const auto decoded = crypto.decrypt_bfv_coefficients(coefficient);
    if (decoded != expected) throw std::runtime_error("Coefficient-packed distance failed");
    std::cout << "COEFFICIENT_DISTANCE correct=" << n << '/' << n
              << " library_noise_bits=" << crypto.bfv_noise_budget(coefficient) << '\n';
    // Prove this path needs no rotations by removing the Galois evaluation key.
    auto coefficient_keys = keys;
    coefficient_keys.rotation_key.reset();
    coefficient_keys.relin_key.reset();
    utils::BfvLweExtractor extractor(coefficient_keys);
    // 500 centroid samples, including both ends and the midpoint of the ring.
    std::vector<std::size_t> positions;
    for (std::size_t i = 0; i < 497; ++i) positions.push_back(i);
    positions.insert(positions.end(), {n / 2 - 1, n / 2, n - 1});
    auto samples = measure(timing, Phase::evaluation, "500 coefficient-to-LWE extractions", [&] {
        std::vector<utils::BfvLweSample> out;
        for (auto i : positions) out.push_back(extractor.extract_coefficient(coefficient, i));
        return out;
    });
    std::size_t full = 0, high16 = 0;
    for (std::size_t j = 0; j < positions.size(); ++j) {
        const auto value = crypto.decrypt_bfv_lwe(samples[j]);
        full += value == expected[positions[j]];
        high16 += (value >> 7) == (expected[positions[j]] >> 7);
        if (j < boundaries.size())
            std::cout << "BOUNDARY coefficient=" << positions[j] << " expected=" << expected[positions[j]]
                      << " actual=" << value << " high16=" << (value >> 7) << '\n';
    }
    std::cout << "COEFFICIENT_EXTRACTION exact=" << full << "/500 high16=" << high16
              << "/500 no_rotation_key=true\n";
    if (full != 500 || high16 != 500) throw std::runtime_error("Coefficient LWE extraction failed");
    for (std::size_t d = 0; d < dimensions; ++d) {
        Words want(n, 0); want[0] = q[d];
        if (crypto.decrypt_bfv_coefficients(query.coordinates[d]) != want) throw std::runtime_error("Query was modified");
    }
    Words want_norm(n, 0); want_norm[0] = norm;
    if (crypto.decrypt_bfv_coefficients(query.squared_norm) != want_norm)
        throw std::runtime_error("Query norm was modified");
    std::cout << "DIAGNOSIS COMPLETE: coefficient packing and LWE extraction passed; "
                 "TFHE-rs key switching/radix conversion and encrypted truncation were NOT executed.\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
