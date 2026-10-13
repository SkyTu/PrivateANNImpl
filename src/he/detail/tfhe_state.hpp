#pragma once

// Internal C API ownership and shared state used by keys, crypto and selection.
// Public callers use the opaque types; no raw key or ciphertext handles escape.
#include "he/entity/tfhe_keys.hpp"
#include "utils/detail/tfhe_rs_api.hpp"

namespace privateann::he::tfhe_rs {
// All objects are owned by the library and destroyed using its C API.
template <class T, class Function, class... Args>
std::shared_ptr<T> create(int (*destroy)(T*), const char* name,
                          Function function, Args... args) {
    T* raw = nullptr;
    const int status = function(args..., &raw);
    std::shared_ptr<T> handle(raw, [destroy](T* pointer) { destroy(pointer); });
    Api::instance().check(status, name);
    if (!raw) throw std::runtime_error(std::string(name) + " returned a null handle");
    return handle;
}

using KeyTag = std::shared_ptr<const int>;

struct Record::Impl {
    std::shared_ptr<FheUint32> distance32;
    std::shared_ptr<FheUint64> distance64;
    std::shared_ptr<FheUint32> id;
    RecordLayout layout;
    KeyTag tag;
    Impl(std::shared_ptr<FheUint32> distance, std::shared_ptr<FheUint32> record_id,
         RecordLayout format, KeyTag key)
        : distance32(std::move(distance)), id(std::move(record_id)), layout(format), tag(std::move(key)) {}
    Impl(std::shared_ptr<FheUint64> distance, std::shared_ptr<FheUint32> record_id,
         RecordLayout format, KeyTag key)
        : distance64(std::move(distance)), id(std::move(record_id)), layout(format), tag(std::move(key)) {}
};
struct EvaluationKey::Impl {
    std::shared_ptr<CompressedServerKey> key;
    KeyTag tag = std::make_shared<const int>(0);
};
struct ConversionKey::Impl {
    std::shared_ptr<void> key;
    KeyTag tag;
};
} // namespace privateann::he::tfhe_rs

namespace privateann::entity {
struct TfheKeys::Impl {
    std::shared_ptr<::ClientKey> key;
    std::shared_ptr<he::tfhe_rs::EvaluationKey> evaluation;
};
} // namespace privateann::entity
