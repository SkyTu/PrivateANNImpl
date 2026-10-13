#include "utils/conversion/bfv_lwe.hpp"
#include <gmpxx.h>

// Adapted from PrivateANN/single-server/he-comparison-exploration/
// bfv-batch-pipeline/pipeline.cpp, commit 79389cd5f349bc67a9c4e69945927c00a27f4fc1.
// This file implements only slot isolation and torus LWE extraction.

namespace privateann::utils {
namespace {
BfvContext require_extraction_keys(const EvaluationKeys& keys) {
    if (!keys.bfv_context)
        throw std::invalid_argument("BFV LWE extraction requires a context");
    keys.bfv_parameters.validate();
    if (keys.bfv_context->get_poly_modulus_degree() != static_cast<int>(keys.bfv_parameters.slots) ||
        keys.bfv_context->get_plain_modulus().value != keys.bfv_parameters.plaintext_modulus)
        throw std::invalid_argument("BFV extraction parameters do not match the context");
    return keys.bfv_context;
}
} // namespace

BfvLweExtractor::BfvLweExtractor(const EvaluationKeys& keys)
    : context_(require_extraction_keys(keys)), rotations_(keys.rotation_key),
      encoder_(context_), evaluator_(context_, encoder_) {}

void BfvLweExtractor::validate(BfvCiphertext& input) const {
    if (input.size() != 2 || input.in_ntt_domain() ||
        input.ring_size() != context_->get_poly_modulus_degree() ||
        input.coeff_modulus_count() != context_->get_ciphertext_modulus_count())
        throw std::invalid_argument("Extraction needs a two-component coefficient-form BFV ciphertext");
}

BfvCiphertext BfvLweExtractor::isolate_slot(BfvCiphertext& input, std::size_t slot) {
    validate(input);
    if (!rotations_) throw std::invalid_argument("BFV slot isolation requires rotation keys");
    const auto n = static_cast<std::size_t>(context_->get_poly_modulus_degree());
    if (slot >= n) throw std::invalid_argument("BFV slot index out of range");
    std::vector<std::uint64_t> mask(n, 0);
    mask[slot] = 1;
    heongpu::Plaintext<bfv_scheme> plain(context_);
    encoder_.encode(plain, mask);
    BfvCiphertext out(context_), rotated(context_);
    evaluator_.multiply_plain(input, plain, out);
    for (std::size_t shift = 1; shift < n / 2; shift *= 2) {
        evaluator_.rotate_rows(out, rotated, *rotations_, static_cast<int>(shift));
        evaluator_.add_inplace(out, rotated);
    }
    evaluator_.rotate_columns(out, rotated, *rotations_);
    evaluator_.add_inplace(out, rotated);
    return out;
}

BfvLweSample BfvLweExtractor::extract_coefficient(BfvCiphertext& input,
                                                 std::size_t coefficient) {
    validate(input);
    const auto n = static_cast<std::size_t>(input.ring_size());
    if (coefficient >= n) throw std::invalid_argument("BFV coefficient index out of range");
    const auto count = static_cast<std::size_t>(input.coeff_modulus_count());
    auto primes = context_->get_key_modulus();
    primes.resize(count); // Ciphertext Q only: exclude the key-switching prime P.
    std::vector<::Data64> words;
    input.get_data(words);
    if (words.size() != 2 * count * n)
        throw std::invalid_argument("Unexpected HEonGPU RNS ciphertext layout");

    // SERVER, CPU arithmetic: exact CRT reconstruction of each public ciphertext
    // coefficient. HEonGPU stores [component][Q prime][coefficient]. These are
    // ciphertext words; no plaintext or secret key is read here.
    mpz_class Q = 1;
    for (const auto& prime : primes) Q *= prime.value;
    std::vector<mpz_class> weights;
    for (const auto& prime : primes) {
        mpz_class part = Q / prime.value, modulus = prime.value, inverse;
        if (!mpz_invert(inverse.get_mpz_t(), part.get_mpz_t(), modulus.get_mpz_t()))
            throw std::runtime_error("BFV RNS moduli are not coprime");
        weights.push_back(part * inverse);
    }
    const mpz_class torus_mask = (mpz_class(1) << 64) - 1;
    const auto torus = [&](std::size_t component, std::size_t index) -> std::uint64_t {
        mpz_class value = 0;
        for (std::size_t j = 0; j < count; ++j)
            value += weights[j] * words[(component * count + j) * n + index];
        value %= Q;
        // Round Q-residues to the native 64-bit torus, exactly as in the CPU
        // reference. Rounding can add error; it cannot repair earlier BFV noise.
        mpz_class rounded = (((value << 64) + Q / 2) / Q) & torus_mask;
        return rounded.get_ui();
    };
    BfvLweSample out{std::vector<std::uint64_t>(n), torus(0, coefficient),
                    static_cast<std::uint32_t>(context_->get_plain_modulus().value)};
    // BFV phase is c0+c1*s, while LWE phase is b-<a,s>. Wraparound terms in
    // X^N+1 have the opposite sign. Unsigned subtraction reduces modulo 2^64.
    for (std::size_t j = 0; j < n; ++j)
        out.mask[j] = j <= coefficient ? std::uint64_t{0} - torus(1, coefficient - j)
                                      : torus(1, n + coefficient - j);
    return out;
}

BfvLweSample BfvLweExtractor::extract_slot(BfvCiphertext& input, std::size_t slot) {
    auto isolated = isolate_slot(input, slot);
    return extract_coefficient(isolated);
}

} // namespace privateann::utils
