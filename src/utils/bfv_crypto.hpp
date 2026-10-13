#pragma once

#include "utils/types.hpp"

namespace privateann::entity { class Client; }
namespace privateann::utils {
struct BfvLweSample;

// CLIENT ENCRYPTION/DECRYPTION: reusable encoder, encryptor and decryptor.
// Shares Client's private key lifetime. Never pass this object to Server.
class BfvCrypto {
public:
    explicit BfvCrypto(const entity::Client& client);
    ~BfvCrypto();
    BfvCrypto(const BfvCrypto&) = delete;
    BfvCrypto& operator=(const BfvCrypto&) = delete;
    // Constant encrypted coordinates and squared norm for linear distance.
    BfvQuery encrypt_bfv_query(const std::vector<std::uint64_t>& coordinates);
    // Raw polynomial coefficients, used by distance and conversion.
    BfvCiphertext encrypt_bfv_coefficients(const std::vector<std::uint64_t>& coefficients);
    std::vector<std::uint64_t> decrypt_bfv_coefficients(BfvCiphertext& ciphertext);
    // SIMD batching is retained for the historical CRT diagnostic.
    BfvCiphertext encrypt_bfv_slots(const std::vector<std::uint64_t>& values);
    std::vector<std::uint64_t> decrypt_bfv_slots(BfvCiphertext& ciphertext);
    // CLIENT DIAGNOSTICS: no decoded value is fed back into server evaluation.
    int bfv_noise_budget(BfvCiphertext& ciphertext);
    std::uint64_t decrypt_bfv_lwe(const BfvLweSample& sample);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace privateann::utils
