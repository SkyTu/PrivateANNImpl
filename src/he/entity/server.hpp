#pragma once

#include "utils/types.hpp"

namespace privateann::he::tfhe_rs { struct GatherTrace; }

namespace privateann::entity {

// SERVER: receives evaluation material only and never stores a secret key.
class Server {
public:
    explicit Server(EvaluationKeys keys);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    Server(Server&&) = delete;
    Server& operator=(Server&&) = delete;

    BfvCiphertext squared_l2(
        BfvQuery& query,
        const std::vector<std::vector<std::uint64_t>>& record_columns);
    std::vector<TfheRecord> topk(std::vector<TfheRecord> records, std::size_t k);
    // SERVER: obliviously gather records using the encrypted IDs from Top-u.
    std::vector<TfheRecord> gather_clusters(const std::vector<TfheRecord>& selected_centroids,
        const std::vector<std::vector<TfheRecord>>& clusters,
        he::tfhe_rs::GatherTrace* trace = nullptr);
    // SERVER: public seeded permutation of positions; never inspect secret IDs.
    std::vector<std::vector<TfheRecord>> random_buckets(const std::vector<TfheRecord>& records,
        std::size_t bucket_count, std::uint64_t seed);
    // SERVER: fixed-capacity buckets; invalid slots hide secret occupancy.
    std::vector<TfheRecord> bucket_minima(const std::vector<std::vector<TfheRecord>>& buckets);
    // Diagnostics for the last layered TFHE operation, without secret data.
    unsigned tfhe_last_workers_used() const;
    unsigned tfhe_last_peak_inflight() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace privateann::entity
