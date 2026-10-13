#pragma once

#include "he/detail/tfhe_types.hpp"

namespace privateann::entity {

// CLIENT KEY GENERATION: reusable TFHE-only key owner, also used by Client.
// Only evaluation_key() is sent to server evaluation code.
class TfheKeys {
public:
    TfheKeys();
    ~TfheKeys();
    TfheKeys(const TfheKeys&) = delete;
    TfheKeys& operator=(const TfheKeys&) = delete;
    std::shared_ptr<he::tfhe_rs::EvaluationKey> evaluation_key() const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class privateann::utils::TfheCrypto;
    friend class privateann::utils::TfheConversion;
};
} // namespace privateann::entity
