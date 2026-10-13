#include "utils/tfhe_crypto.hpp"
#include "he/detail/tfhe_state.hpp"

namespace privateann::utils {
using namespace he;
using namespace he::tfhe_rs;

TfheCrypto::TfheCrypto(const entity::TfheKeys& keys) : keys_(keys.impl_) {}

Record TfheCrypto::encrypt(std::uint64_t score, std::uint64_t id,
                        unsigned score_bits, unsigned id_bits) const {
    const RecordLayout layout(score_bits, id_bits);
    layout.validate(score, id);
    auto& api = Api::instance();
    auto encrypted_id = create(api.fhe_uint32_destroy, "encrypt TFHE ID",
        api.fhe_uint32_try_encrypt_with_client_key_u32, static_cast<std::uint32_t>(id), keys_->key.get());
    if (layout.distance_ciphertext_bits() == 32) {
        auto distance = create(api.fhe_uint32_destroy, "encrypt TFHE distance32",
            api.fhe_uint32_try_encrypt_with_client_key_u32, static_cast<std::uint32_t>(score), keys_->key.get());
        return Record(std::make_shared<Record::Impl>(
            std::move(distance), std::move(encrypted_id), layout, keys_->evaluation->impl_->tag));
    }
    auto distance = create(api.fhe_uint64_destroy, "encrypt TFHE distance64",
        api.fhe_uint64_try_encrypt_with_client_key_u64, score, keys_->key.get());
    return Record(std::make_shared<Record::Impl>(
        std::move(distance), std::move(encrypted_id), layout, keys_->evaluation->impl_->tag));
}

PlainRecord TfheCrypto::decrypt(const Record& record) const {
    if (!record.impl_ || record.impl_->tag != keys_->evaluation->impl_->tag)
        throw std::invalid_argument("TFHE ciphertext is empty or belongs to another client");
    // CLIENT: diagnostic decryption for component tests only.
    auto& api = Api::instance();
    std::uint32_t id = 0;
    api.check(api.fhe_uint32_decrypt(record.impl_->id.get(), keys_->key.get(), &id), "decrypt TFHE ID");
    std::uint64_t score = 0;
    if (record.impl_->distance32) {
        std::uint32_t value = 0;
        api.check(api.fhe_uint32_decrypt(record.impl_->distance32.get(), keys_->key.get(), &value),
                  "decrypt TFHE distance32");
        score = value;
    } else {
        api.check(api.fhe_uint64_decrypt(record.impl_->distance64.get(), keys_->key.get(), &score),
                  "decrypt TFHE distance64");
    }
    return record.impl_->layout.decode(score, id);
}

} // namespace privateann::utils
