#pragma once

#include "utils/types.hpp"
#include <algorithm>
#include <stdexcept>

namespace privateann::utils {

// Encode raw polynomial coefficients, without the CRT batching transform.
// HEonGPU has no public coefficient-vector constructor: encode zero to create
// a valid plaintext buffer, then write the coefficients. High entries stay zero.
inline heongpu::Plaintext<bfv_scheme> encode_bfv_coefficients(
    BfvContext context, heongpu::HEEncoder<bfv_scheme>& encoder,
    const std::vector<std::uint64_t>& coefficients) {
    const auto n = static_cast<std::size_t>(context->get_poly_modulus_degree());
    const auto t = context->get_plain_modulus().value;
    if (coefficients.empty() || coefficients.size() > n ||
        std::any_of(coefficients.begin(), coefficients.end(),
                    [t](auto value) { return value >= t; }))
        throw std::invalid_argument("BFV polynomial coefficients must be residues below t");
    heongpu::Plaintext<bfv_scheme> plain(context);
    encoder.encode(plain, std::vector<std::uint64_t>(n, 0));
    const auto error = cudaMemcpy(plain.data(), coefficients.data(),
                                  coefficients.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("BFV coefficient encoding: ") + cudaGetErrorString(error));
    return plain;
}

} // namespace privateann::utils
