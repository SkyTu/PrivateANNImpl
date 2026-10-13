#include "he/entity/server.hpp"
#include "he/bfv_distance.hpp"
#include "he/tfhe_topk.hpp"

namespace privateann::entity {

struct Server::Impl {
    EvaluationKeys keys;
    heongpu::HEEncoder<bfv_scheme> encoder;
    heongpu::HEArithmeticOperator<bfv_scheme> evaluator;
    std::unique_ptr<he::tfhe_rs::Evaluator> tfhe;

    static BfvContext require_context(const EvaluationKeys& bundle) {
        if (!bundle.bfv_context)
            throw std::invalid_argument("Server requires a BFV context");
        bundle.bfv_parameters.validate();
        if (bundle.bfv_context->get_poly_modulus_degree() != static_cast<int>(bundle.bfv_parameters.slots) ||
            bundle.bfv_context->get_plain_modulus().value != bundle.bfv_parameters.plaintext_modulus)
            throw std::invalid_argument("BFV context does not match the declared parameters");
        return bundle.bfv_context;
    }

    explicit Impl(EvaluationKeys bundle)
        : keys(std::move(bundle)), encoder(require_context(keys)),
          evaluator(keys.bfv_context, encoder) {
        if (keys.tfhe_key) tfhe = std::make_unique<he::tfhe_rs::Evaluator>(keys.tfhe_key);
    }
};

Server::Server(EvaluationKeys keys) : impl_(std::make_unique<Impl>(std::move(keys))) {}
Server::~Server() = default;

BfvCiphertext Server::squared_l2(
    BfvQuery& query,
    const std::vector<std::vector<std::uint64_t>>& record_columns) {
    return he::squared_l2(impl_->keys, impl_->encoder, impl_->evaluator,
                          query, record_columns);
}

std::vector<TfheRecord> Server::topk(std::vector<TfheRecord> records, std::size_t k) {
    if (!impl_->tfhe) throw std::logic_error("TFHE-rs was not enabled on the client");
    return he::topk(*impl_->tfhe, std::move(records), k);
}

std::vector<TfheRecord> Server::gather_clusters(
    const std::vector<TfheRecord>& selected_centroids,
    const std::vector<std::vector<TfheRecord>>& clusters, he::tfhe_rs::GatherTrace* trace) {
    if (!impl_->tfhe) throw std::logic_error("TFHE-rs was not enabled on the client");
    return impl_->tfhe->gather_clusters(selected_centroids, clusters, trace);
}

std::vector<std::vector<TfheRecord>> Server::random_buckets(
    const std::vector<TfheRecord>& records, std::size_t bucket_count, std::uint64_t seed) {
    if (!impl_->tfhe) throw std::logic_error("TFHE-rs was not enabled on the client");
    return impl_->tfhe->random_buckets(records, bucket_count, seed);
}

std::vector<TfheRecord> Server::bucket_minima(
    const std::vector<std::vector<TfheRecord>>& buckets) {
    if (!impl_->tfhe) throw std::logic_error("TFHE-rs was not enabled on the client");
    return impl_->tfhe->bucket_minima(buckets);
}

unsigned Server::tfhe_last_workers_used() const {
    if (!impl_->tfhe) throw std::logic_error("TFHE-rs was not enabled on the client");
    return impl_->tfhe->last_workers_used();
}

unsigned Server::tfhe_last_peak_inflight() const {
    if (!impl_->tfhe) throw std::logic_error("TFHE-rs was not enabled on the client");
    return impl_->tfhe->last_peak_inflight();
}

} // namespace privateann::entity
