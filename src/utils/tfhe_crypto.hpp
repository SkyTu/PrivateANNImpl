#pragma once

#include "he/entity/tfhe_keys.hpp"

namespace privateann::utils {

// CLIENT ENCRYPTION/DECRYPTION: shares the private key lifetime with TfheKeys.
// Construct once and reuse; server computation never receives this object.
class TfheCrypto {
public:
    explicit TfheCrypto(const entity::TfheKeys& keys);
    he::tfhe_rs::Record encrypt(std::uint64_t score, std::uint64_t id,
                              unsigned score_bits, unsigned id_bits) const;
    he::PlainRecord decrypt(const he::tfhe_rs::Record& record) const;
private:
    std::shared_ptr<entity::TfheKeys::Impl> keys_;
};
} // namespace privateann::utils
