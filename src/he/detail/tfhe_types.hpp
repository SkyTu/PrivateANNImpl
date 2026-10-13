#pragma once

#include "he/detail/record_layout.hpp"
#include <memory>
#include <utility>

namespace privateann::entity { class TfheKeys; }
namespace privateann::utils { class TfheCrypto; class TfheConversion; }

namespace privateann::he::tfhe_rs {
class Evaluator;

// SERVER: opaque cross-scheme evaluation material, with no client secret.
class ConversionKey {
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit ConversionKey(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    friend class privateann::utils::TfheConversion;
};

// Immutable shared ownership of separate encrypted distance and ID fields.
class Record {
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    explicit Record(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
    friend class Evaluator;
    friend class privateann::utils::TfheCrypto;
    friend class privateann::utils::TfheConversion;
public:
    Record() = default;
    unsigned distance_ciphertext_bits() const;
    unsigned id_ciphertext_bits() const;
};

// SERVER: compressed evaluation key only. No secret material.
class EvaluationKey {
    struct Impl;
    std::shared_ptr<Impl> impl_;
    explicit EvaluationKey(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
    friend class Evaluator;
    friend class privateann::entity::TfheKeys;
    friend class privateann::utils::TfheCrypto;
    friend class privateann::utils::TfheConversion;
};
} // namespace privateann::he::tfhe_rs
