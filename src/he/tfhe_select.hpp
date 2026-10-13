#pragma once

#include "he/detail/tfhe_types.hpp"
#include "he/detail/topk_schedule.hpp"
#include <functional>
#include <vector>

struct FheBool; // Opaque native condition, used only by the private selector.

namespace privateann::he::tfhe_rs {

// Optional test diagnostics: actual logical reads/writes in public job order.
// Thread interleavings are not recorded. No ciphertext contents are inspected.
struct GatherAccess {
    std::size_t cluster, row, output;
    bool operator==(const GatherAccess& other) const {
        return cluster == other.cluster && row == other.row && output == other.output;
    }
};
struct GatherTrace {
    std::vector<GatherAccess> accesses;
};

// SERVER SELECTION: native GPU distance comparison and paired field selection.
// IDs are never compared for ordering; equal distances may return either ID.
// The evaluator
// owns a reusable CUDA evaluation key; it has no encryption or conversion API.
class Evaluator {
public:
    // Each host worker gets independent CUDA streams. This allows overlapping
    // API calls, but does not guarantee GPU kernel overlap or a speedup. The
    // device and backend allocation/synchronization paths determine concurrency.
    // One worker serializes host gate submission.
    explicit Evaluator(std::shared_ptr<EvaluationKey> key, unsigned workers = 4);
    ~Evaluator();
    Evaluator(const Evaluator&) = delete;
    Evaluator& operator=(const Evaluator&) = delete;
    void validate(const std::vector<Record>& records) const;
    Record padding(const Record& prototype) const;
    Record minimum(const Record& a, const Record& b) const;
    // Maximum uses distance order, including the existing high sentinel.
    // Callers selecting valid maxima must supply valid records (no padding).
    Record maximum(const Record& a, const Record& b) const;
    std::pair<Record, Record> minmax(const Record& a, const Record& b) const;
    // SERVER: selected_centroids normally comes from topk(centroids, u).
    // Its encrypted ID is the public index of a source cluster. Every cluster
    // has L > 0 records in the same layout; centroid and record layouts may
    // differ, but all ciphertexts must use this evaluator's key.
    // Returns u*L encrypted records in (centroid rank, source row) order.
    // Invalid/unmatched selectors yield invalid records; repeats repeat a
    // cluster. Valid, distinct centroid IDs are a prerequisite of Top-u.
    // Every output scans every cluster. No secret index or host branch is used.
    std::vector<Record> gather_clusters(const std::vector<Record>& selected_centroids,
        const std::vector<std::vector<Record>>& clusters, GatherTrace* trace = nullptr) const;
    // SERVER PUBLIC ROUTING: shuffle positions with a public seed, then assign
    // them round-robin to B buckets. The seed must be independent of secrets.
    // No ciphertext is inspected; distance/ID/validity stay in one record.
    // Equal capacity ceil(n/B), invalid padding, and no candidate is dropped.
    std::vector<std::vector<Record>> random_buckets(const std::vector<Record>& records,
        std::size_t bucket_count, std::uint64_t seed) const;
    // SERVER: one encrypted (distance, ID) minimum per public bucket.
    // Use B > 0 equal-capacity buckets, with 1 <= capacity <= 65536. Empty
    // slots must contain invalid padding, not absent records. All records
    // share one key/layout. An all-invalid bucket returns an invalid record.
    // The public Top-1 schedule runs for every bucket, regardless of validity;
    // independent gates across AND within buckets share the GPU worker pool.
    std::vector<Record> bucket_minima(const std::vector<std::vector<Record>>& buckets) const;
    void execute_layers(const TopkSchedule& schedule,
                        const std::function<void(const TopkGate&)>& operation) const;
    unsigned workers() const;
    // Host scheduling diagnostics only; neither value measures GPU concurrency.
    unsigned last_workers_used() const;
    unsigned last_peak_inflight() const;
private:
    Record select(const FheBool* condition, const Record& if_true, const Record& if_false) const;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace privateann::he::tfhe_rs
