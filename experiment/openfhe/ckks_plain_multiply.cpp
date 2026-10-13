// Standalone experiment using OpenFHE's native APIs, without entity wrappers.
// One encrypted uint8 query is multiplied elementwise by plaintext uint8 records.
// CKKS represents these integers approximately; every rounded product is checked.
#include "openfhe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) try {
    std::uint64_t vector_count = 1'000'000;
    constexpr std::uint64_t dimensions = 128;
    std::filesystem::path report = "logs/openfhe_plain_multiply.csv";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--vectors" && i + 1 < argc) {
            const std::string value = argv[++i];
            std::size_t consumed = 0;
            vector_count = std::stoull(value, &consumed);
            if (value.empty() || value[0] == '-' || consumed != value.size())
                throw std::invalid_argument("Invalid vector count");
        } else if (arg == "--report" && i + 1 < argc) {
            report = argv[++i];
        } else {
            throw std::invalid_argument("Usage: ckks_plain_multiply [--vectors N] [--report PATH]");
        }
    }
    if (!vector_count || vector_count > UINT64_MAX / dimensions)
        throw std::invalid_argument("Vector count is zero or overflows the element count");
    const std::uint64_t elements = vector_count * dimensions;
    const auto wall_start = Clock::now();
    auto elapsed_ms = [](Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };

    // CLIENT: initialize CKKS and generate its encryption/decryption key pair once.
    // Plaintext multiplication does not require a relinearization or rotation key.
    auto start = Clock::now();
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(1);
    parameters.SetFirstModSize(60);
    // Leave integer headroom for the full 16-bit product. The upstream example's
    // 50-bit scale targets small real inputs and is too large for this depth/range.
    parameters.SetScalingModSize(40);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetSecurityLevel(HEStd_128_classic);
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();
    const double setup_ms = elapsed_ms(start);
    const std::size_t slots = cc->GetRingDimension() / 2;
    if (slots % dimensions != 0)
        throw std::runtime_error("Packing must contain complete 128D vectors");
    const auto batches = (elements - 1) / slots + 1;
    std::cout << "OpenFHE " << EXPERIMENT_OPENFHE_VERSION
              << "; backend=CPU; omp_threads=" << omp_get_max_threads()
              << "; operation=CKKS_ciphertext_times_plaintext\n"
              << "vectors=" << vector_count << "; dimensions=" << dimensions
              << "; input_bits=8; output_range=0..65025; scalar_products=" << elements
              << "; ring_dimension=" << cc->GetRingDimension() << "; slots=" << slots
              << "; vectors_per_batch=" << slots / dimensions << "; batches=" << batches
              << "; security=HEStd_128_classic; depth=1; scaling=FLEXIBLEAUTO"
              << "; first_modulus_bits=60; scaling_modulus_bits=40\n" << std::flush;

    // CLIENT: repeat the same 128D query for all records packed in a ciphertext.
    constexpr double boundary_values[] = {255, 0, 1, 127, 128, 254};
    std::vector<double> query(slots);
    for (std::size_t i = 0; i < slots; ++i) {
        const auto d = i % dimensions;
        query[i] = (d * 37 + 19) & 255;
        if (d < 6) query[i] = boundary_values[d];
    }
    start = Clock::now();
    auto query_plain = cc->MakeCKKSPackedPlaintext(query);
    auto query_encrypted = cc->Encrypt(keys.publicKey, query_plain);
    const double query_encryption_ms = elapsed_ms(start);

    std::mt19937 generator(0x5a17c3U);
    std::vector<double> records(slots);
    double encoding_ms = 0, multiplication_ms = 0, decryption_ms = 0, verification_ms = 0;
    double max_absolute_error = 0;
    std::uint64_t verified_elements = 0, completed_batches = 0;
    for (std::uint64_t offset = 0; offset < elements; offset += slots) {
        const auto active = std::min<std::uint64_t>(slots, elements - offset);
        // SERVER DATA: generate unsigned bytes outside every cryptographic timer.
        for (std::size_t i = 0; i < active; ++i) {
            records[i] = generator() & 255U;
            if (offset + i < dimensions) records[i] = 255;
            else if (offset + i < 2 * dimensions) records[i] = 0;
        }
        std::fill(records.begin() + active, records.end(), 0);

        // SERVER: encode this plaintext batch; report encoding separately.
        start = Clock::now();
        auto record_plain = cc->MakeCKKSPackedPlaintext(records);
        encoding_ms += elapsed_ms(start);

        // SERVER: native ciphertext-plaintext multiplication, with no custom circuit.
        // This timer includes only EvalMult; no explicit rescale is added afterward.
        start = Clock::now();
        auto product = cc->EvalMult(query_encrypted, record_plain);
        multiplication_ms += elapsed_ms(start);
        ++completed_batches;

        // CLIENT: decrypt and decode after the multiplication timer has stopped.
        start = Clock::now();
        Plaintext decrypted;
        cc->Decrypt(keys.secretKey, product, &decrypted);
        decrypted->SetLength(slots);
        const auto& actual = decrypted->GetRealPackedValue();
        decryption_ms += elapsed_ms(start);

        // CLIENT: check every product, including zero padding in the final batch.
        start = Clock::now();
        if (actual.size() != slots) throw std::runtime_error("Unexpected output slot count");
        for (std::size_t i = 0; i < slots; ++i) {
            const auto expected = static_cast<std::uint64_t>(query[i]) *
                                  static_cast<std::uint64_t>(records[i]);
            const auto error = std::abs(actual[i] - static_cast<double>(expected));
            if (!std::isfinite(actual[i]) || error >= 0.5 ||
                std::llround(actual[i]) != static_cast<long long>(expected))
                throw std::runtime_error("Product mismatch at scalar index " + std::to_string(offset + i) +
                                         ": expected=" + std::to_string(expected) +
                                         ", actual=" + std::to_string(actual[i]) +
                                         ", absolute_error=" + std::to_string(error));
            max_absolute_error = std::max(max_absolute_error, error);
        }
        verified_elements += active;
        verification_ms += elapsed_ms(start);
        if (completed_batches % 1024 == 0 || completed_batches == batches)
            std::cout << "Verified vectors=" << verified_elements / dimensions << '/' << vector_count
                      << "; batches=" << completed_batches << '/' << batches
                      << "; multiplication_ms=" << multiplication_ms << std::endl;
    }

    // CLIENT: confirm all repeated evaluations preserved the encrypted query.
    start = Clock::now();
    Plaintext query_after;
    cc->Decrypt(keys.secretKey, query_encrypted, &query_after);
    query_after->SetLength(slots);
    const auto& unchanged = query_after->GetRealPackedValue();
    decryption_ms += elapsed_ms(start);
    if (unchanged.size() != slots) throw std::runtime_error("Unexpected query slot count");
    for (std::size_t i = 0; i < slots; ++i)
        if (!std::isfinite(unchanged[i]) || std::abs(unchanged[i] - query[i]) >= 0.5)
            throw std::runtime_error("Evaluation changed the encrypted query");
    if (verified_elements != elements || completed_batches != batches)
        throw std::runtime_error("Incomplete workload");

    const double wall_ms = elapsed_ms(wall_start);
    const double server_ms = encoding_ms + multiplication_ms;
    if (!report.parent_path().empty()) std::filesystem::create_directories(report.parent_path());
    std::ofstream csv(report);
    if (!csv) throw std::runtime_error("Cannot write report");
    csv << "status,backend,openfhe_version,vectors,dimensions,input_bits,scalar_products,ring_dimension,slots,batches,omp_threads,"
           "setup_ms,query_encoding_encryption_ms,record_encoding_ms,multiplication_ms,server_encoding_plus_multiplication_ms,"
           "decryption_ms,verification_ms,wall_ms,max_absolute_error,scalar_products_per_second\n"
        << std::setprecision(12) << "PASS,CPU," << EXPERIMENT_OPENFHE_VERSION << ','
        << vector_count << ',' << dimensions << ",8," << verified_elements << ','
        << cc->GetRingDimension() << ',' << slots << ',' << completed_batches << ',' << omp_get_max_threads() << ','
        << setup_ms << ',' << query_encryption_ms << ',' << encoding_ms << ',' << multiplication_ms << ','
        << server_ms << ',' << decryption_ms << ',' << verification_ms << ',' << wall_ms << ','
        << max_absolute_error << ',' << elements * 1000.0 / multiplication_ms << '\n';
    csv.close();
    if (!csv) throw std::runtime_error("Failed to finish report");
    std::cout << std::fixed << std::setprecision(6)
              << "PASS: all " << verified_elements << " rounded uint8 products match full integer results; query unchanged\n"
              << "setup_ms=" << setup_ms << "; query_encoding_encryption_ms=" << query_encryption_ms << '\n'
              << "record_encoding_ms=" << encoding_ms << "; multiplication_ms=" << multiplication_ms
              << "; server_encoding_plus_multiplication_ms=" << server_ms << '\n'
              << "decryption_ms=" << decryption_ms << "; verification_ms=" << verification_ms
              << "; wall_ms=" << wall_ms << "; max_absolute_error=" << max_absolute_error << '\n'
              << "CSV report: " << std::filesystem::absolute(report) << std::endl;
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
}
