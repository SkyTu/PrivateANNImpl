#include "he/tfhe_select.hpp"
#include "he/detail/tfhe_state.hpp"
#include <atomic>
#include <exception>
#include <optional>
#include <limits>
#include <numeric>
#include <random>
#include <omp.h>

namespace privateann::he::tfhe_rs {
namespace {
// Each worker installs a key once for the whole network. The native C API clones
// the key's stream set, giving each worker independent CUDA streams. Nested gate
// scopes reuse that activation; standalone gates still activate their own key.
thread_local const CudaServerKey* active_cuda_key = nullptr;
struct CudaKeyScope {
    Api& api = Api::instance();
    const CudaServerKey* previous = active_cuda_key;
    bool changed;
    explicit CudaKeyScope(const CudaServerKey* key) : changed(key != previous) {
        if (changed) {
            api.check(api.set_cuda_server_key(key), "set_cuda_server_key");
            active_cuda_key = key;
        }
    }
    ~CudaKeyScope() {
        if (changed) {
            if (previous) api.set_cuda_server_key(previous);
            else api.unset_server_key();
            active_cuda_key = previous;
        }
    }
};

} // namespace

unsigned Record::distance_ciphertext_bits() const {
    if (!impl_) throw std::invalid_argument("Empty TFHE record");
    return impl_->layout.distance_ciphertext_bits();
}
unsigned Record::id_ciphertext_bits() const {
    if (!impl_) throw std::invalid_argument("Empty TFHE record");
    return impl_->layout.id_ciphertext_bits();
}
struct Evaluator::Impl {
    std::shared_ptr<CudaServerKey> key;
    KeyTag tag;
    unsigned workers = 4;
    std::atomic<unsigned> last_workers{0}, last_peak{0};
};

Evaluator::Evaluator(std::shared_ptr<EvaluationKey> key, unsigned workers)
    : impl_(std::make_unique<Impl>()) {
    if (!key || !key->impl_) throw std::invalid_argument("TFHE evaluation key is required");
    if (workers < 1 || workers > 32) throw std::invalid_argument("TFHE workers must be in [1, 32]");
    impl_->workers = workers;
    auto& api = Api::instance();
    // SERVER: decompress evaluation material to GPU once. No secret key.
    impl_->key = create(api.cuda_server_key_destroy, "compressed_server_key_decompress_to_gpu",
                          api.compressed_server_key_decompress_to_gpu, key->impl_->key.get());
    impl_->tag = key->impl_->tag;
}
Evaluator::~Evaluator() = default;

void Evaluator::validate(const std::vector<Record>& records) const {
    for (const auto& record : records) {
        if (!record.impl_ || record.impl_->tag != impl_->tag)
            throw std::invalid_argument("TFHE ciphertext is empty or uses the wrong evaluation key");
        if (record.impl_->layout != records.front().impl_->layout)
            throw std::invalid_argument("TFHE batch must use identical score/ID bit widths");
    }
}

Record Evaluator::padding(const Record& prototype) const {
    if (!prototype.impl_ || prototype.impl_->tag != impl_->tag)
        throw std::invalid_argument("Padding requires a record under this evaluation key");
    const auto layout = prototype.impl_->layout;
    CudaKeyScope scope(impl_->key.get());
    auto& api = Api::instance();
    // Public sentinel exceeds all declared distances, including UINT32_MAX.
    auto id = create(api.fhe_uint32_destroy, "padding ID", api.fhe_uint32_try_encrypt_trivial_u32, 0u);
    if (layout.distance_ciphertext_bits() == 32) {
        auto distance = create(api.fhe_uint32_destroy, "padding distance32",
            api.fhe_uint32_try_encrypt_trivial_u32, static_cast<std::uint32_t>(layout.padding()));
        return Record(std::make_shared<Record::Impl>(std::move(distance), std::move(id), layout, impl_->tag));
    }
    auto distance = create(api.fhe_uint64_destroy, "padding distance64",
        api.fhe_uint64_try_encrypt_trivial_u64, layout.padding());
    return Record(std::make_shared<Record::Impl>(std::move(distance), std::move(id), layout, impl_->tag));
}

Record Evaluator::select(const FheBool* condition, const Record& if_true, const Record& if_false) const {
    // Callers validate inputs and install the CUDA key. Select both fields with
    // one encrypted condition so distance, ID and validity never get separated.
    auto& api = Api::instance();
    auto id = create(api.fhe_uint32_destroy, "select ID", api.fhe_uint32_if_then_else,
        condition, if_true.impl_->id.get(), if_false.impl_->id.get());
    if (if_true.impl_->distance32) {
        auto distance = create(api.fhe_uint32_destroy, "select distance32", api.fhe_uint32_if_then_else,
            condition, if_true.impl_->distance32.get(), if_false.impl_->distance32.get());
        return Record(std::make_shared<Record::Impl>(std::move(distance), std::move(id), if_true.impl_->layout, impl_->tag));
    }
    auto distance = create(api.fhe_uint64_destroy, "select distance64", api.fhe_uint64_if_then_else,
        condition, if_true.impl_->distance64.get(), if_false.impl_->distance64.get());
    return Record(std::make_shared<Record::Impl>(std::move(distance), std::move(id), if_true.impl_->layout, impl_->tag));
}

Record Evaluator::minimum(const Record& a, const Record& b) const {
    validate({a, b});
    CudaKeyScope scope(impl_->key.get());
    auto& api = Api::instance();
    // SERVER COMPARISON: IDs do not enter gt. A tie retains the left record.
    const auto swap = a.impl_->distance32
        ? create(api.fhe_bool_destroy, "distance32 gt", api.fhe_uint32_gt,
                 a.impl_->distance32.get(), b.impl_->distance32.get())
        : create(api.fhe_bool_destroy, "distance64 gt", api.fhe_uint64_gt,
                 a.impl_->distance64.get(), b.impl_->distance64.get());
    return select(swap.get(), b, a);
}

Record Evaluator::maximum(const Record& a, const Record& b) const {
    validate({a, b});
    CudaKeyScope scope(impl_->key.get());
    auto& api = Api::instance();
    // SERVER COMPARISON: only distance determines the winner. The encrypted
    // condition selects both fields; a tie retains the right record.
    const auto swap = a.impl_->distance32
        ? create(api.fhe_bool_destroy, "distance32 gt", api.fhe_uint32_gt,
                 a.impl_->distance32.get(), b.impl_->distance32.get())
        : create(api.fhe_bool_destroy, "distance64 gt", api.fhe_uint64_gt,
                 a.impl_->distance64.get(), b.impl_->distance64.get());
    return select(swap.get(), a, b);
}

std::pair<Record, Record> Evaluator::minmax(const Record& a, const Record& b) const {
    validate({a, b});
    CudaKeyScope scope(impl_->key.get());
    auto& api = Api::instance();
    // SERVER: one distance comparison drives all four field CMUX operations.
    // Equal inputs keep their local order; the whole network is not stable.
    const auto swap = a.impl_->distance32
        ? create(api.fhe_bool_destroy, "distance32 gt", api.fhe_uint32_gt,
                 a.impl_->distance32.get(), b.impl_->distance32.get())
        : create(api.fhe_bool_destroy, "distance64 gt", api.fhe_uint64_gt,
                 a.impl_->distance64.get(), b.impl_->distance64.get());
    auto minimum = select(swap.get(), b, a);
    auto maximum = select(swap.get(), a, b);
    return {std::move(minimum), std::move(maximum)};
}

std::vector<Record> Evaluator::gather_clusters(
    const std::vector<Record>& selected_centroids,
    const std::vector<std::vector<Record>>& clusters, GatherTrace* trace) const {
    // Public metadata validation finishes before any encrypted selection.
    if (trace) trace->accesses.clear();
    const auto u = selected_centroids.size(), count = clusters.size();
    if (!u || !count || u > count || clusters.front().empty())
        throw std::invalid_argument("Gather requires 1 <= u <= C and nonempty equal-size clusters");
    validate(selected_centroids);
    const auto centroid_layout = selected_centroids.front().impl_->layout;
    if (count - 1 > centroid_layout.max_id())
        throw std::invalid_argument("Centroid ID width cannot address every cluster");
    const auto rows = clusters.front().size();
    for (const auto& cluster : clusters) {
        if (cluster.size() != rows)
            throw std::invalid_argument("Gather requires equal-size clusters");
        validate(cluster);
        if (cluster.front().impl_->layout != clusters.front().front().impl_->layout)
            throw std::invalid_argument("All gathered records must share one score/ID layout");
    }
    const auto checked_product = [](std::size_t a, std::size_t b) {
        if (b > std::numeric_limits<std::size_t>::max() / a)
            throw std::invalid_argument("Gather dimensions overflow the address space");
        return a * b;
    };
    const auto output_count = checked_product(u, rows);
    if (trace) trace->accesses.resize(checked_product(output_count, count));
    std::vector<Record> output(output_count, padding(clusters.front().front()));

    const unsigned threads = std::min<std::size_t>(impl_->workers, u);
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    const auto capture_error = [&] {
        const auto current = std::current_exception();
#pragma omp critical(privateimpl_tfhe_gather_error)
        { if (!error) error = current; }
        failed.store(true);
    };

    // Public job j owns output[j*L .. (j+1)*L). Each worker keeps independent
    // CUDA streams active for its jobs. Native calls finish before returning;
    // shared inputs are immutable and output regions never overlap.
#pragma omp parallel num_threads(threads)
    {
        std::optional<CudaKeyScope> scope;
        try { scope.emplace(impl_->key.get()); }
        catch (...) { capture_error(); }
#pragma omp for schedule(static)
        for (std::size_t j = 0; j < u; ++j) {
            if (failed.load()) continue;
            try {
                auto& api = Api::instance();
                const auto& centroid = selected_centroids[j];
                // Invalid centroid padding must not select cluster zero even
                // when its ID is zero. Validity stays encrypted throughout.
                const auto valid = centroid.impl_->distance32
                    ? create(api.fhe_bool_destroy, "centroid validity32", api.fhe_uint32_scalar_lt,
                        centroid.impl_->distance32.get(), static_cast<std::uint32_t>(centroid_layout.padding()))
                    : create(api.fhe_bool_destroy, "centroid validity64", api.fhe_uint64_scalar_lt,
                        centroid.impl_->distance64.get(), centroid_layout.padding());
                // SERVER: scan all clusters for every selected output. Only
                // public counters choose addresses; the encrypted flag never
                // becomes a host boolean, an index, or a scheduling decision.
                for (std::size_t c = 0; c < count; ++c) {
                    auto equal = create(api.fhe_bool_destroy, "centroid ID == cluster",
                        api.fhe_uint32_scalar_eq, centroid.impl_->id.get(), static_cast<std::uint32_t>(c));
                    auto flag = create(api.fhe_bool_destroy, "valid centroid match",
                        api.fhe_bool_bitand, valid.get(), equal.get());
                    // One equality is reused for every row in this cluster.
                    for (std::size_t r = 0; r < rows; ++r) {
                        const auto destination = j * rows + r;
                        if (trace) trace->accesses[(j * count + c) * rows + r] = {c, r, destination};
                        const auto& source = clusters[c][r];
                        const auto& previous = output[destination];
                        // The same encrypted flag selects distance and ID.
                        output[destination] = select(flag.get(), source, previous);
                    }
                }
            } catch (...) { capture_error(); }
        }
    }
    if (error) std::rethrow_exception(error);
    return output;
}

std::vector<std::vector<Record>> Evaluator::random_buckets(
    const std::vector<Record>& records, std::size_t bucket_count, std::uint64_t seed) const {
    if (records.empty() || bucket_count == 0 || bucket_count > 65536)
        throw std::invalid_argument("Random buckets requires records and 1 <= B <= 65536");
    validate(records);
    const auto capacity = records.size() / bucket_count + (records.size() % bucket_count != 0);
    validate_topk_size(capacity, 1);
    if (bucket_count > std::numeric_limits<std::size_t>::max() / capacity)
        throw std::invalid_argument("Random bucket dimensions overflow the address space");
    std::vector<std::size_t> permutation(records.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937_64 random(seed);
    std::shuffle(permutation.begin(), permutation.end(), random);
    std::vector<std::vector<Record>> buckets(
        bucket_count, std::vector<Record>(capacity, padding(records.front())));
    // Public gather-output positions are shuffled on the host. These handles
    // still hold encrypted original IDs; no secret determines an address.
    // Unlike hashing secret IDs, this balanced mapping needs no encrypted
    // routing comparisons and has no overflow bucket or dropped candidate.
    for (std::size_t i = 0; i < permutation.size(); ++i)
        buckets[i % bucket_count][i / bucket_count] = records[permutation[i]];
    return buckets;
}

std::vector<Record> Evaluator::bucket_minima(
    const std::vector<std::vector<Record>>& buckets) const {
    // Only public shape/key/layout metadata is checked on the host. Secret
    // occupancy must never determine a vector length or a skipped bucket.
    if (buckets.empty() || buckets.front().empty())
        throw std::invalid_argument("Bucket minima requires positive bucket count and capacity");
    const auto capacity = buckets.front().size();
    const auto local = topk_schedule(capacity, 1);
    if (buckets.size() > std::numeric_limits<std::size_t>::max() / capacity)
        throw std::invalid_argument("Bucket dimensions overflow the address space");
    for (const auto& bucket : buckets) {
        if (bucket.size() != capacity)
            throw std::invalid_argument("Bucket minima requires equal public capacities");
        validate(bucket);
        if (bucket.front().impl_->layout != buckets.front().front().impl_->layout)
            throw std::invalid_argument("All buckets must share one score/ID layout");
    }

    // Reuse Top-k with k=1: capacity-1 distance compare/select gates, no max gates and no
    // extra power-of-two padding. Each bucket's winner finishes at offset 0.
    // Offset the existing gates instead of introducing another sorting circuit.
    TopkSchedule network{buckets.size() * capacity, {}};
    network.layers.resize(local.layers.size());
    std::vector<Record> values;
    values.reserve(network.size);
    for (std::size_t b = 0; b < buckets.size(); ++b) {
        const auto offset = b * capacity;
        values.insert(values.end(), buckets[b].begin(), buckets[b].end());
        for (std::size_t layer = 0; layer < local.layers.size(); ++layer)
            for (const auto& gate : local.layers[layer])
                network.layers[layer].push_back({offset + gate.a, offset + gate.b, true});
    }

    // SERVER GPU EVALUATION: the existing executor installs independent CUDA
    // streams per worker and waits at every layer boundary. Slots in a layer
    // are disjoint. Independent streams permit concurrency; they do not prove
    // kernel overlap, particularly with synchronous cudaMalloc/cudaFree on a
    // device without memory pools. No secret selects an address or host branch.
    execute_layers(network, [&](const TopkGate& gate) {
        values[gate.a] = minimum(values[gate.a], values[gate.b]);
    });
    std::vector<Record> winners;
    winners.reserve(buckets.size());
    for (std::size_t b = 0; b < buckets.size(); ++b)
        winners.push_back(std::move(values[b * capacity]));
    return winners;
}

unsigned Evaluator::workers() const { return impl_->workers; }
unsigned Evaluator::last_workers_used() const { return impl_->last_workers.load(); }
unsigned Evaluator::last_peak_inflight() const { return impl_->last_peak.load(); }

void Evaluator::execute_layers(const TopkSchedule& schedule,
                              const std::function<void(const TopkGate&)>& operation) const {
    std::size_t max_layer = 0;
    for (const auto& layer : schedule.layers) max_layer = std::max(max_layer, layer.size());
    impl_->last_workers = 0;
    impl_->last_peak = 0;
    if (!max_layer) return;
    const unsigned threads = std::min<std::size_t>(impl_->workers, max_layer);
    std::atomic<bool> failed{false};
    std::atomic<unsigned> inflight{0}, peak{0};
    std::exception_ptr error;
    // Errors must not escape an OpenMP worker or bypass a layer's barrier.
    const auto capture_error = [&] {
        const auto current = std::current_exception();
#pragma omp critical(privateimpl_tfhe_error)
        { if (!error) error = current; }
        failed.store(true);
    };
#pragma omp parallel num_threads(threads)
    {
        std::optional<CudaKeyScope> scope;
        try { scope.emplace(impl_->key.get()); }
        catch (...) { capture_error(); }
#pragma omp single
        { impl_->last_workers = static_cast<unsigned>(omp_get_num_threads()); }
        for (const auto& layer : schedule.layers) {
            // Public schedule guarantees that each slot occurs once per layer.
            // Every native operation completes before the implicit layer barrier.
#pragma omp for schedule(static)
            for (std::size_t i = 0; i < layer.size(); ++i) {
                if (failed.load()) continue;
                const auto active = inflight.fetch_add(1) + 1;
                auto previous = peak.load();
                while (previous < active && !peak.compare_exchange_weak(previous, active)) {}
                try { operation(layer[i]); }
                catch (...) { capture_error(); }
                inflight.fetch_sub(1);
            }
        }
    }
    impl_->last_peak = peak.load();
    if (error) std::rethrow_exception(error);
}

} // namespace privateann::he::tfhe_rs
