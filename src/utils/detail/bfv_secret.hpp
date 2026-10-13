#pragma once

#include "utils/types.hpp"
#include <openssl/crypto.h>

namespace privateann::utils::detail {

// CLIENT ONLY: temporary coefficient secret, wiped when the scope exits.
struct SecretWords {
    std::vector<std::uint64_t> values;
    explicit SecretWords(std::vector<std::uint64_t> words) : values(std::move(words)) {}
    SecretWords(const SecretWords&) = delete;
    SecretWords& operator=(const SecretWords&) = delete;
    ~SecretWords() { OPENSSL_cleanse(values.data(), values.size() * sizeof(std::uint64_t)); }
};
SecretWords coefficient_secret(const BfvContext& context,
                              heongpu::Secretkey<bfv_scheme>& secret_key);
} // namespace privateann::utils::detail
