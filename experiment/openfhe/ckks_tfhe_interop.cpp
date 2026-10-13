#include "openfhe.h"
#include "binfhecontext.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Explicit little-endian format for a local diagnostic fixture, not serialization
// for deployment. The runner creates a private temporary directory and deletes it.
static void write_u64(std::ostream& out, uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
        out.put(static_cast<char>(value >> (8 * byte)));
}

int main(int argc, char** argv) try {
    if (argc != 3)
        throw std::runtime_error("Usage: ckks_tfhe_interop PRIVATE_FIXTURE_PREFIX OUTPUT_CSV");
    std::cout << std::unitbuf << std::fixed << std::setprecision(6);
    const auto wall_start = Clock::now();
    constexpr size_t count = 500, dimensions = 128, slots = 512;
    constexpr uint64_t log_q = 25, q = uint64_t{1} << log_q;
    constexpr uint64_t full_p = uint64_t{1} << 23;

    // PLAINTEXT DISTANCE COMPUTATION: uint8 coordinates, exact squared L2.
    // Boundary records exercise zero, maximum, ties and adjacent distances.
    auto start = Clock::now();
    std::mt19937 rng(20261012);
    std::array<int64_t, dimensions> query{};
    for (size_t j = 0; j < dimensions; ++j)
        query[j] = (j % 2) ? 255 : 0;
    std::vector<uint64_t> distances(count);
    std::vector<double> values(slots, 0.0);
    for (size_t i = 0; i < count; ++i) {
        uint64_t distance = 0;
        for (size_t j = 0; j < dimensions; ++j) {
            int64_t coordinate = rng() & 255;
            if (i == 0 || i == 3 || i == 4) coordinate = query[j];
            if (i == 1 || i == 2) coordinate = 255 - query[j];
            if ((i == 2 || i == 3 || i == 4) && j == 0) coordinate = 1;
            const auto difference = query[j] - coordinate;
            distance += static_cast<uint64_t>(difference * difference);
        }
        distances[i] = distance;
        values[i] = static_cast<double>(distance);
    }
    const double distance_ms = elapsed(start);
    if (distances[0] != 0 || distances[1] != 8323200 || distances[3] != 1 || distances[4] != 1)
        throw std::runtime_error("Plaintext boundary fixture is incorrect");

    // CLIENT KEY GENERATION: keep the upstream intermediate modulus of 27 bits.
    // Increasing it merely to fit a larger message would need a security review.
    start = Clock::now();
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(3);
    parameters.SetFirstModSize(60);
    parameters.SetScalingModSize(50);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetBatchSize(slots);
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(SCHEMESWITCH);
    auto keys = cc->KeyGen();
    SchSwchParams switch_parameters;
    switch_parameters.SetSecurityLevelCKKS(HEStd_128_classic);
    switch_parameters.SetSecurityLevelFHEW(STD128);
    switch_parameters.SetCtxtModSizeFHEWLargePrec(log_q);
    switch_parameters.SetNumSlotsCKKS(slots);
    auto lwe_key = cc->EvalCKKStoFHEWSetup(switch_parameters);
    auto bin = cc->GetBinCCForSchemeSwitch();
    cc->EvalCKKStoFHEWKeyGen(keys, lwe_key);
    const double setup_ms = elapsed(start);
    const uint64_t recommended_p = q / (2 * bin->GetBeta().ConvertToInt());
    std::cout << "count=" << count << " dimensions=" << dimensions
              << " CKKS_ring=" << cc->GetRingDimension() << " slots=" << slots
              << " LWE_dimension=" << lwe_key->GetElement().GetLength()
              << " q=" << q << " recommended_p=" << recommended_p
              << " requested_p=" << full_p << '\n';
    std::cout << "distance_ms=" << distance_ms << " setup_keygen_ms=" << setup_ms << '\n';

    // CLIENT ENCRYPTION: the original 500 full distances occupy one CKKS ciphertext.
    start = Clock::now();
    const auto encrypted = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(values));
    const double encrypt_ms = elapsed(start);

    std::ofstream csv(argv[2]);
    if (!csv) throw std::runtime_error("Cannot create timing CSV");
    csv << std::fixed << std::setprecision(6)
        << "case,count,dimensions,ring_dimension,lwe_dimension,q,p,distance_ms,setup_keygen_ms,encode_encrypt_ms,precompute_ms,switch_ms,verify_ms,exact_matches,modular_matches,max_circular_error\n";
    size_t full_matches = 0;
    for (const auto& item : std::vector<std::pair<std::string, uint64_t>>{
             {"full23", full_p}, {"modular_control", recommended_p}}) {
        const auto& name = item.first;
        const uint64_t p = item.second;
        // SERVER PRECOMPUTATION AND EVALUATION: no secret key or decryption here.
        // full23 deliberately probes beyond the recommended noise margin.
        // The control uses the SAME CKKS input and checks its residue modulo p.
        start = Clock::now();
        cc->EvalCKKStoFHEWPrecompute(1.0 / static_cast<double>(p));
        const double precompute_ms = elapsed(start);
        start = Clock::now();
        const auto converted = cc->EvalCKKStoFHEW(encrypted, count);
        const double switch_ms = elapsed(start);
        if (converted.size() != count) throw std::runtime_error("Unexpected output count");

        // CLIENT DIAGNOSTIC DECRYPTION: never feeds plaintext back to encryption.
        start = Clock::now();
        std::vector<uint64_t> decoded(count);
        size_t exact_matches = 0, modular_matches = 0;
        uint64_t max_error = 0;
        for (size_t i = 0; i < count; ++i) {
            LWEPlaintext result;
            bin->Decrypt(lwe_key, converted[i], &result, p);
            decoded[i] = static_cast<uint64_t>(result);
            exact_matches += decoded[i] == distances[i];
            modular_matches += decoded[i] == distances[i] % p;
            const auto difference = (decoded[i] + p - distances[i] % p) % p;
            max_error = std::max(max_error, std::min(difference, p - difference));
        }
        const double verify_ms = elapsed(start);
        if (name == "full23") full_matches = exact_matches;
        std::cout << name << ": precompute_ms=" << precompute_ms << " switch_ms=" << switch_ms
                  << " verify_ms=" << verify_ms << " exact_matches=" << exact_matches << "/" << count
                  << " modular_matches=" << modular_matches << "/" << count
                  << " max_circular_error=" << max_error << '\n';
        csv << name << ',' << count << ',' << dimensions << ',' << cc->GetRingDimension() << ','
            << lwe_key->GetElement().GetLength() << ',' << q << ',' << p << ',' << distance_ms << ','
            << setup_ms << ',' << encrypt_ms << ',' << precompute_ms << ',' << switch_ms << ','
            << verify_ms << ',' << exact_matches << ',' << modular_matches << ',' << max_error << '\n';

        // LOCAL TEST FIXTURE ONLY: raw coefficients, matching secret and reference
        // results let TFHE-rs independently verify the imported LWE phase.
        // The secret is NOT a server input and is never written to a report/log.
        std::ofstream out(std::string(argv[1]) + "_" + name + ".bin", std::ios::binary);
        out.write("OFHETFH1", 8);
        const auto& secret = lwe_key->GetElement();
        write_u64(out, count); write_u64(out, secret.GetLength());
        write_u64(out, q); write_u64(out, p);
        for (size_t j = 0; j < secret.GetLength(); ++j) {
            const auto coefficient = secret[j].ConvertToInt();
            const auto modulus = secret.GetModulus().ConvertToInt();
            if (coefficient != 0 && coefficient != 1 && coefficient != modulus - 1)
                throw std::runtime_error("Unexpected non-ternary key");
            write_u64(out, coefficient == modulus - 1 ? UINT64_MAX : coefficient);
        }
        for (size_t i = 0; i < count; ++i) {
            write_u64(out, distances[i]); write_u64(out, decoded[i]);
            if (converted[i]->GetModulus().ConvertToInt() != q)
                throw std::runtime_error("Unexpected ciphertext modulus");
            for (size_t j = 0; j < secret.GetLength(); ++j)
                write_u64(out, converted[i]->GetA()[j].ConvertToInt());
            write_u64(out, converted[i]->GetB().ConvertToInt());
        }
        if (!out) throw std::runtime_error("Cannot write private diagnostic fixture");
        if (name == "modular_control" && modular_matches != count)
            throw std::runtime_error("Supported-range modular control failed");
    }

    // API LIMIT PROBE: enough headroom for 23 message bits would require logQ=31
    // under the example's q/(2*beta) rule. Test the actual upstream exception.
    try {
        BinFHEContext limit_probe;
        limit_probe.GenerateBinFHEContext(STD128, false, 31);
        throw std::runtime_error("Expected OpenFHE logQ=31 rejection was absent");
    } catch (const OpenFHEException& error) {
        std::cout << "LARGE_MODULUS_PROBE: " << error.what() << '\n';
    }
    std::cout << "openfhe_wall_ms=" << elapsed(wall_start) << '\n';
    std::cout << (full_matches == count ? "FULL_DISTANCE_SAMPLE_PASS" : "FULL_DISTANCE_PRECISION_FAIL") << '\n';
    return full_matches == count ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
}
