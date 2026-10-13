#pragma once

// Internal client key ownership, shared only with client-side crypto utilities.
#include "he/entity/client.hpp"

namespace privateann::entity {
struct Client::Impl {
    BfvParameters parameters;
    BfvContext context;
    heongpu::Secretkey<bfv_scheme> secret_key;
    heongpu::Publickey<bfv_scheme> public_key;
    std::shared_ptr<heongpu::Relinkey<bfv_scheme>> relin_key;
    std::shared_ptr<heongpu::Galoiskey<bfv_scheme>> rotation_key;
    std::unique_ptr<TfheKeys> tfhe;
    std::shared_ptr<he::tfhe_rs::ConversionKey> conversion_key;
    Impl(BfvParameters config, bool enable_tfhe);
};
} // namespace privateann::entity
