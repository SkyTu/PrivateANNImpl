#pragma once

#include "utils/types.hpp"
#include "he/entity/tfhe_keys.hpp"

namespace privateann::utils { class BfvCrypto; }
namespace privateann::entity {

// CLIENT KEY OWNER: initializes parameters and generates keys once. Send only
// evaluation_keys() to Server and conversion utilities. Client-side crypto
// utilities share the private key lifetime without exposing raw secret keys.
class Client {
public:
    explicit Client(BfvParameters parameters = {}, bool enable_tfhe = false);
    ~Client();
    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;
    Client(Client&&) = delete;
    Client& operator=(Client&&) = delete;

    EvaluationKeys evaluation_keys() const;
    // CLIENT ONLY: for constructing utils::TfheCrypto, never sent to Server.
    const TfheKeys& tfhe_keys() const;
    // Generate cross-scheme evaluation material before sharing evaluation_keys().
    void enable_bfv_tfhe_conversion();
    // Optional keys used by the historical CRT diagnostics, not linear distance.
    void enable_bfv_relinearization();
    void enable_bfv_lwe_extraction();
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class privateann::utils::BfvCrypto;
};
} // namespace privateann::entity
