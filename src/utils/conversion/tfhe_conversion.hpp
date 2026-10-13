#pragma once

#include "he/entity/tfhe_keys.hpp"
#include <vector>

namespace privateann::utils {

// Internal adapter for the Rust conversion bridge. Client setup and server
// conversion have separate entry points and accept different key material.
class TfheConversion {
public:
    // CLIENT KEY GENERATION: borrowed secret words are not retained.
    static std::shared_ptr<he::tfhe_rs::ConversionKey> generate_key(
        const entity::TfheKeys& keys, const std::vector<std::uint64_t>& bfv_secret,
        std::uint32_t modulus);
    // SERVER CONVERSION: [mask..., body] samples, evaluation keys only.
    static std::vector<he::tfhe_rs::Record> convert_lwe(
        const he::tfhe_rs::EvaluationKey& target, const he::tfhe_rs::ConversionKey& key,
        const std::vector<std::uint64_t>& samples, const std::vector<std::uint64_t>& ids,
        unsigned id_bits, unsigned drop_bits);
};
} // namespace privateann::utils
