#include "utils/bfv_crypto.hpp"
#include "utils/bfv_encoding.hpp"
#include "utils/conversion/bfv_lwe.hpp"
#include "utils/detail/bfv_secret.hpp"
#include "he/entity/detail/client_state.hpp"
#include <algorithm>
#include <stdexcept>

namespace privateann::utils {
namespace {
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("BFV client diagnostic: ") + cudaGetErrorString(error));
}

} // namespace

struct BfvCrypto::Impl {
    std::shared_ptr<entity::Client::Impl> keys;
    heongpu::HEEncoder<bfv_scheme> encoder;
    heongpu::HEEncryptor<bfv_scheme> encryptor;
    heongpu::HEDecryptor<bfv_scheme> decryptor;
    explicit Impl(const entity::Client& client)
        : keys(client.impl_), encoder(keys->context),
          encryptor(keys->context, keys->public_key),
          decryptor(keys->context, keys->secret_key) {}
};
BfvCrypto::BfvCrypto(const entity::Client& client) : impl_(std::make_unique<Impl>(client)) {}
BfvCrypto::~BfvCrypto() = default;

BfvQuery BfvCrypto::encrypt_bfv_query(const std::vector<std::uint64_t>& coordinates) {
    impl_->keys->parameters.validate_distance_dimension(coordinates.size());
    if (std::any_of(coordinates.begin(), coordinates.end(), [&](auto value) {
            return value > impl_->keys->parameters.max_coordinate;
        }))
        throw std::invalid_argument("BFV query coordinates exceed the configured bound");
    // CLIENT ENCRYPTION: send one constant ciphertext per coordinate and one
    // for ||q||^2. The server never receives the plaintext query or its norm.
    std::vector<BfvCiphertext> encrypted;
    encrypted.reserve(coordinates.size());
    std::uint64_t norm = 0;
    for (auto value : coordinates) {
        encrypted.push_back(encrypt_bfv_coefficients({value}));
        norm += value * value;
    }
    return {std::move(encrypted), encrypt_bfv_coefficients({norm})};
}

int BfvCrypto::bfv_noise_budget(BfvCiphertext& ciphertext) {
    return impl_->decryptor.remainder_noise_budget(ciphertext);
}

std::uint64_t BfvCrypto::decrypt_bfv_lwe(const BfvLweSample& sample) {
    const auto n = impl_->keys->parameters.slots;
    if (sample.mask.size() != n || sample.plaintext_modulus != impl_->keys->parameters.plaintext_modulus)
        throw std::invalid_argument("LWE sample does not match this BFV client's parameters");
    auto secret = detail::coefficient_secret(impl_->keys->context, impl_->keys->secret_key);
    std::uint64_t phase = sample.body;
    for (std::size_t j = 0; j < n; ++j) phase -= sample.mask[j] * secret.values[j];
    // Round the torus phase using the source BFV modulus, not TFHE-rs encoding.
    const auto scaled = static_cast<unsigned __int128>(phase) * sample.plaintext_modulus;
    return static_cast<std::uint64_t>((scaled + (static_cast<unsigned __int128>(1) << 63)) >> 64)
           % sample.plaintext_modulus;
}

BfvCiphertext BfvCrypto::encrypt_bfv_slots(const std::vector<std::uint64_t>& values) {
    if (values.size() != impl_->keys->parameters.slots ||
        std::any_of(values.begin(), values.end(), [&](auto value) {
            return value > impl_->keys->parameters.max_coordinate;
        }))
        throw std::invalid_argument("BFV query needs one bounded coordinate per slot");
    // CLIENT: encode and encrypt a query column.
    heongpu::Plaintext<bfv_scheme> plain(impl_->keys->context);
    impl_->encoder.encode(plain, values);
    BfvCiphertext ciphertext(impl_->keys->context);
    impl_->encryptor.encrypt(ciphertext, plain);
    return ciphertext;
}

std::vector<std::uint64_t> BfvCrypto::decrypt_bfv_slots(BfvCiphertext& ciphertext) {
    // CLIENT: diagnostic decryption for distance component tests.
    heongpu::Plaintext<bfv_scheme> plain(impl_->keys->context);
    impl_->decryptor.decrypt(plain, ciphertext);
    std::vector<std::uint64_t> values;
    impl_->encoder.decode(values, plain);
    return values;
}

BfvCiphertext BfvCrypto::encrypt_bfv_coefficients(const std::vector<std::uint64_t>& coefficients) {
    // CLIENT: encode and encrypt a polynomial under this client's public key.
    auto plain = encode_bfv_coefficients(impl_->keys->context, impl_->encoder, coefficients);
    BfvCiphertext ciphertext(impl_->keys->context);
    impl_->encryptor.encrypt(ciphertext, plain);
    return ciphertext;
}

std::vector<std::uint64_t> BfvCrypto::decrypt_bfv_coefficients(BfvCiphertext& ciphertext) {
    // CLIENT DIAGNOSTIC: read decrypted polynomial coefficients without the
    // batching decoder. Calling decrypt_bfv_slots() would change their meaning.
    heongpu::Plaintext<bfv_scheme> plain(impl_->keys->context);
    impl_->decryptor.decrypt(plain, ciphertext);
    std::vector<std::uint64_t> coefficients(impl_->keys->parameters.slots);
    cuda_check(cudaMemcpy(coefficients.data(), plain.data(), coefficients.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    return coefficients;
}

} // namespace privateann::utils
