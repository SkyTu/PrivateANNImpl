#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace privateann {

struct BfvParameters {
    // 128 unsigned byte coordinates need at most 128 * 255^2 = 8,323,200
    // (23 bits). This 23-bit prime exceeds that bound and is 1 modulo 2N,
    // as required by HEonGPU's plaintext NTT tables even with coefficient
    // packing. N=8192 has no suitable 23-bit prime. 'slots' is the ring degree
    // and the number of records per batch; distance outputs use coefficients.
    std::size_t slots = 4096;
    std::uint32_t plaintext_modulus = 8380417;
    std::uint32_t max_coordinate = 255;

    void validate() const {
        if (slots < 4096 || slots > 65536 || (slots & (slots - 1)) != 0)
            throw std::invalid_argument("BFV ring degree must be a power of two in [4096, 65536]");
        if (plaintext_modulus > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            plaintext_modulus < 2 || plaintext_modulus % (2 * slots) != 1 ||
            max_coordinate >= plaintext_modulus)
            throw std::invalid_argument("Invalid BFV batching modulus or coordinate bound");
        // Batching needs a prime modulus, not just t = 1 (mod 2N).
        for (std::uint32_t divisor = 3; divisor <= plaintext_modulus / divisor; divisor += 2)
            if (plaintext_modulus % divisor == 0)
                throw std::invalid_argument("BFV batching modulus must be prime");
    }

    void validate_distance_dimension(std::size_t dimension) const {
        validate();
        if (dimension == 0)
            throw std::invalid_argument("Squared L2 needs at least one coordinate");
        const std::uint64_t max_squared = std::uint64_t(max_coordinate) * max_coordinate;
        if (max_squared && dimension > (plaintext_modulus - 1) / max_squared)
            throw std::invalid_argument("Squared L2 can wrap modulo the BFV plaintext modulus");
    }
};

inline void validate_word(std::uint64_t value, unsigned bits) {
    if (bits == 0 || bits > 32 || value >= (std::uint64_t{1} << bits))
        throw std::invalid_argument("TFHE word must fit in 1..32 unsigned bits");
}

} // namespace privateann
