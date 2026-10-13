#pragma once

#include "utils/types.hpp"

namespace privateann::utils {

// Torus LWE under the ORIGINAL BFV coefficient secret, with phase b - <a,s>.
// A message m has phase approximately m / plaintext_modulus on R/Z.
// This is an intermediate ciphertext, not a native TFHE-rs radix integer.
struct BfvLweSample {
    std::vector<std::uint64_t> mask;
    std::uint64_t body;
    std::uint32_t plaintext_modulus;
};

// SERVER: first stage of the CPU SEAL/MOSFHET conversion, adapted to HEonGPU.
// Only the context and optional Galois evaluation key are retained; no secret
// key. Coefficient extraction needs no rotation key or slot isolation.
class BfvLweExtractor {
public:
    explicit BfvLweExtractor(const EvaluationKeys& keys);
    // Replicate one public batching slot into all slots. The resulting plaintext
    // polynomial is constant, so coefficient 0 carries that slot's message.
    BfvCiphertext isolate_slot(BfvCiphertext& input, std::size_t slot);
    // Low-level coefficient extraction; a coefficient index is NOT a slot ID.
    BfvLweSample extract_coefficient(BfvCiphertext& input, std::size_t coefficient = 0);
    BfvLweSample extract_slot(BfvCiphertext& input, std::size_t slot);

private:
    BfvContext context_;
    std::shared_ptr<heongpu::Galoiskey<bfv_scheme>> rotations_;
    heongpu::HEEncoder<bfv_scheme> encoder_;
    heongpu::HEArithmeticOperator<bfv_scheme> evaluator_;
    void validate(BfvCiphertext& input) const;
};

} // namespace privateann::utils
