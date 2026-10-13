#pragma once

#include "utils/types.hpp"

namespace privateann::utils {

// SERVER CONVERSION: receives only public parameters and evaluation keys.
// Construct after Client::enable_bfv_tfhe_conversion(). Secret keys and crypto
// utilities are neither accepted nor needed. Reuse for multiple batches.
class BfvToTfheConverter {
public:
    explicit BfvToTfheConverter(EvaluationKeys keys);
    // Public coefficient indices and IDs. drop_bits=0 preserves the full
    // distance; 7 explicitly requests the CPU reference's distance >> 7.
    std::vector<TfheRecord> convert(BfvCiphertext& distances,
        const std::vector<std::size_t>& coefficients, const std::vector<std::uint64_t>& ids,
        unsigned id_bits, unsigned drop_bits = 0) const;
private:
    EvaluationKeys keys_;
};
} // namespace privateann::utils
