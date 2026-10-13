#pragma once

#include "utils/parameters.hpp"
#include "he/detail/tfhe_types.hpp"
#include <heongpu/heongpu.hpp>
#include <memory>
#include <vector>

namespace privateann {

inline constexpr auto bfv_scheme = heongpu::Scheme::BFV;
using BfvContext = heongpu::HEContext<bfv_scheme>;
using BfvCiphertext = heongpu::Ciphertext<bfv_scheme>;
using TfheRecord = he::tfhe_rs::Record;

// CLIENT: each ciphertext encrypts a constant polynomial. Coordinate d holds
// q[d] at coefficient 0; squared_norm holds sum_d q[d]^2 at coefficient 0.
// SERVER: reuse this query across database batches without changing it.
struct BfvQuery {
    std::vector<BfvCiphertext> coordinates;
    BfvCiphertext squared_norm;
};

// Public parameters and evaluation keys only. No secret keys or decryptors
// can be passed to Server through this type. Shared ownership avoids copying
// large keys inside this single-process test harness.
struct EvaluationKeys {
    BfvParameters bfv_parameters;
    BfvContext bfv_context;
    // Optional: used only by the historical ciphertext-square diagnostic.
    std::shared_ptr<heongpu::Relinkey<bfv_scheme>> relin_key;
    std::shared_ptr<he::tfhe_rs::EvaluationKey> tfhe_key;
    std::shared_ptr<heongpu::Galoiskey<bfv_scheme>> rotation_key;
    std::shared_ptr<he::tfhe_rs::ConversionKey> conversion_key;
};

} // namespace privateann
