#include "openfhe.h"
#include "binfhecontext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using namespace lbcrypto;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("Usage: ckks_topk_cpu OUTPUT_CSV");
    std::cout << std::unitbuf << std::fixed << std::setprecision(6);
    std::ofstream csv(argv[1]);
    if (!csv) throw std::runtime_error("Cannot open CSV");
    csv << std::fixed << std::setprecision(6) << "stage,iteration,milliseconds\n";
    const auto wall_start = Clock::now();
    constexpr uint32_t count = 100, k = 10, dimensions = 128, slots = 128;
    constexpr double divisor = 65536.0, penalty = 256.0;
    // The native argmin returns near level 20; leave room for mask comparison.
    // Refreshing masks between rounds avoids a depth proportional to k.
    constexpr uint32_t depth = 23;

    // PLAINTEXT INPUT: preserve full uint8 squared distances using an exact
    // power-of-two scale. No integer truncation or quantization is performed.
    auto start = Clock::now();
    std::mt19937 rng(20261012);
    std::vector<uint64_t> distances(count, 0);
    std::vector<double> values(slots, penalty), max_values(slots, -penalty);
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t j = 0; j < dimensions; ++j) {
            const int64_t query = (j % 2) ? 255 : 0;
            int64_t record = rng() & 255;
            if (i == 0) record = query;
            if (i == 1) record = 255 - query;
            const int64_t difference = query - record;
            distances[i] += difference * difference;
        }
        values[i] = max_values[i] = distances[i] / divisor;
    }
    std::vector<uint32_t> expected(count);
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(expected.begin(), expected.end(), [&](auto a, auto b) {
        return distances[a] < distances[b];
    });
    uint64_t min_gap = UINT64_MAX;
    for (uint32_t i = 1; i < count; ++i)
        min_gap = std::min(min_gap, distances[expected[i]] - distances[expected[i - 1]]);
    csv << "plaintext_distances_and_reference,0," << elapsed(start) << '\n';
    std::cout << "INPUT: count=" << count << " k=" << k << " dimensions=" << dimensions
              << " coordinate_bits=8 padded_slots=" << slots << " min_distance_gap=" << min_gap
              << " top10_boundary_gap=" << distances[expected[k]] - distances[expected[k - 1]] << '\n';

    // CLIENT KEY GENERATION: use the native secure parameter sets, not TOY.
    start = Clock::now();
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(depth);
    params.SetScalingModSize(50);
    params.SetFirstModSize(60);
    params.SetScalingTechnique(FLEXIBLEAUTOEXT);
    params.SetSecurityLevel(HEStd_128_classic);
    params.SetNumLargeDigits(4);
    params.SetBatchSize(slots);
    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);
    auto keys = cc->KeyGen();
    SchSwchParams switching;
    switching.SetSecurityLevelCKKS(HEStd_128_classic);
    switching.SetSecurityLevelFHEW(STD128);
    switching.SetCtxtModSizeFHEWLargePrec(25);
    switching.SetNumSlotsCKKS(slots);
    switching.SetNumValues(slots);
    switching.SetComputeArgmin(true);
    switching.SetOneHotEncoding(true);
    auto lwe_key = cc->EvalSchemeSwitchingSetup(switching);
    std::cout << "SETUP: CKKS_ring=" << cc->GetRingDimension() << " depth=" << depth
              << " FHEW_dimension=" << lwe_key->GetElement().GetLength()
              << "; generating switching and bootstrapping keys...\n";
    cc->EvalSchemeSwitchingKeyGen(keys, lwe_key);
    const double keygen_ms = elapsed(start);
    csv << "setup_keygen,0," << keygen_ms << '\n';
    std::cout << "setup_keygen_ms=" << keygen_ms << '\n';

    // SERVER PRECOMPUTATION: scaled score differences remain inside (-512,512),
    // including the public padding and encrypted exclusion penalties.
    start = Clock::now();
    const uint32_t p_lwe = (uint32_t{1} << 25) / (2 * cc->GetBinCCForSchemeSwitch()->GetBeta().ConvertToInt());
    cc->EvalCompareSwitchPrecompute(p_lwe, 128.0);
    const double precompute_ms = elapsed(start);
    csv << "comparison_precompute,0," << precompute_ms << '\n';

    // CLIENT ENCRYPTION: the evaluator receives only ciphertexts and public keys.
    start = Clock::now();
    const auto original = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(values));
    const auto maximum_input = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(max_values));
    const auto half = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.5)));
    csv << "client_encoding_encryption,0," << elapsed(start) << '\n';

    // SERVER MAX: benchmark the existing public max/argmax interface separately.
    start = Clock::now();
    const auto maximum = cc->EvalMaxSchemeSwitching(maximum_input, keys.publicKey, slots, slots);
    const double max_ms = elapsed(start);
    csv << "native_max,0," << max_ms << '\n';
    std::cout << "native_max_ms=" << max_ms << '\n';

    // SERVER TOP-K: ten native min/argmin calls with encrypted exclusion masks.
    // OpenFHE supplies the comparisons; this experiment only schedules calls.
    // Refresh each one-hot mask through a native comparison with 0.5. Its return
    // to CKKS resets the mask's level, avoiding ten rounds of accumulated depth.
    auto current = original;
    Ciphertext<DCRTPoly> excluded;
    std::vector<std::vector<Ciphertext<DCRTPoly>>> results;
    results.reserve(k);
    double topk_ms = 0.0, argmin_total_ms = 0.0, mask_total_ms = 0.0;
    for (uint32_t round = 0; round < k; ++round) {
        start = Clock::now();
        auto result = cc->EvalMinSchemeSwitching(current, keys.publicKey, slots, slots);
        const double argmin_ms = elapsed(start);
        double mask_ms = 0.0;
        if (round + 1 < k) {
            start = Clock::now();
            auto refreshed = cc->EvalCompareSchemeSwitching(half, result[1], slots, slots);
            excluded = excluded ? cc->EvalAdd(excluded, refreshed) : refreshed;
            current = cc->EvalAdd(original, cc->EvalMult(excluded, penalty));
            mask_ms = elapsed(start);
        }
        argmin_total_ms += argmin_ms;
        mask_total_ms += mask_ms;
        topk_ms += argmin_ms + mask_ms;
        csv << "native_argmin," << round << ',' << argmin_ms << '\n'
            << "encrypted_mask_refresh_update," << round << ',' << mask_ms << '\n';
        csv.flush();
        std::cout << "round=" << round + 1 << " argmin_ms=" << argmin_ms << " mask_ms=" << mask_ms
                  << " result_level=" << result[1]->GetLevel() << '\n';
        results.push_back(std::move(result));
    }
    csv << "topk_evaluation_total,0," << topk_ms << '\n';

    // CLIENT DECRYPTION AND VERIFICATION: only after every server round finishes.
    start = Clock::now();
    auto verify_indicator = [&](const Ciphertext<DCRTPoly>& encrypted, uint32_t winner) {
        Plaintext plaintext;
        cc->Decrypt(keys.secretKey, encrypted, &plaintext);
        plaintext->SetLength(slots);
        const auto decoded = plaintext->GetRealPackedValue();
        for (uint32_t i = 0; i < slots; ++i) {
            const double wanted = i == winner ? 1.0 : 0.0;
            if (!std::isfinite(decoded[i]) || std::abs(decoded[i] - wanted) > 0.01)
                throw std::runtime_error("Incorrect encrypted winner indicator at slot " + std::to_string(i)
                    + "; expected winner=" + std::to_string(winner) + " decoded=" + std::to_string(decoded[i]));
        }
    };
    verify_indicator(maximum[1], expected.back());
    double max_value_error = 0.0;
    for (uint32_t round = 0; round < k; ++round) {
        verify_indicator(results[round][1], expected[round]);
        Plaintext plaintext;
        cc->Decrypt(keys.secretKey, results[round][0], &plaintext);
        plaintext->SetLength(1);
        const double decoded_distance = plaintext->GetRealPackedValue()[0] * divisor;
        if (!std::isfinite(decoded_distance)) throw std::runtime_error("Non-finite selected value");
        max_value_error = std::max(max_value_error, std::abs(decoded_distance - distances[expected[round]]));
        std::cout << "verified_rank=" << round + 1 << " id=" << expected[round]
                  << " distance=" << distances[expected[round]] << " decoded_distance=" << decoded_distance << '\n';
    }
    Plaintext unchanged;
    cc->Decrypt(keys.secretKey, original, &unchanged);
    unchanged->SetLength(slots);
    for (uint32_t i = 0; i < slots; ++i)
        if (std::abs(unchanged->GetRealPackedValue()[i] - values[i]) > 1e-6)
            throw std::runtime_error("Original input was modified");
    csv << "client_decryption_verification,0," << elapsed(start) << '\n';
    csv << "wall_total,0," << elapsed(wall_start) << '\n';
    std::cout << "PASS: argmax index and all 10 ordered Top-k indices, padding excluded, input unchanged\n"
              << "Selected distance values are approximate; exact integer values are not asserted.\n"
              << "argmin_total_ms=" << argmin_total_ms << " mask_total_ms=" << mask_total_ms
              << " topk_evaluation_ms=" << topk_ms << " max_selected_value_error=" << max_value_error
              << " wall_ms=" << elapsed(wall_start) << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
