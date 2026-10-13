#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace privateann::he {

inline void validate_topk_size(std::size_t n, std::size_t k) {
    if (n == 0 || k == 0 || k > n || n > 65536)
        throw std::invalid_argument("Top-k needs 1 <= k <= n <= 65536");
}

struct TopkGate {
    // A full gate writes min to a and max to b. A min-only gate writes min
    // to a and discards its larger output; b is outside the surviving blocks.
    std::size_t a, b;
    bool min_only;
};
struct TopkSchedule {
    std::size_t size;
    std::vector<std::vector<TopkGate>> layers;
};

namespace detail {
inline std::size_t power_below(std::size_t n) {
    std::size_t p = 1;
    while (p <= (n - 1) / 2) p *= 2;
    return p;
}

inline void merge_topk_block(std::vector<TopkGate>& gates, std::size_t offset,
                             std::size_t n) {
    // Batcher bitonic merge: the input must be bitonic, and the output is
    // ascending. Compare opposite halves, then merge each half recursively.
    // In this Top-k construction n is always a power of two, so m = n/2.
    if (n < 2) return;
    const auto m = power_below(n);
    for (std::size_t i = 0; i < n - m; ++i)
        gates.push_back({offset + i, offset + i + m, false});
    merge_topk_block(gates, offset, m);
    merge_topk_block(gates, offset + m, n - m);
}
} // namespace detail


// Algorithm: grouped sorting followed by pruned bitonic merges.
//   1. Let width = next_power_of_two(k). Pad and sort blocks of that width.
//   2. Merge pairs of ascending blocks, keeping only their smallest width
//      values. Since width >= k, each discarded value has at least k values
//      no greater than itself, and cannot improve the final smallest-k result.
//   3. Repeat until one sorted block remains; the executor returns its first k.
// Initial block sorting uses Batcher's odd-even merge schedule below, whereas
// the subsequent merges are bitonic. This is not a full bitonic sort of n.
// The executor compares distance only and carries the corresponding ID through
// the same selection. Equal distances have no ID tie-breaking requirement.
// Example n=100, k=10: width=16, padded size=112, 25 layers of independent gates,
// 633 full gates and 96 min-only gates (729 gates total).
inline TopkSchedule topk_schedule(std::size_t n, std::size_t k) {
    validate_topk_size(n, k);
    std::size_t width = 1;
    while (width < k) width *= 2;
    TopkSchedule network{((n + width - 1) / width) * width, {}};
    std::vector<TopkGate> gates;

    // Stage 1: iterative Batcher odd-even merge sorting within each block.
    // p selects the merge scale; q/r/d generate its comparator distances and
    // parity groups. All gates write the smaller-distance record to the lower index.
    for (std::size_t offset = 0; offset < network.size; offset += width) {
        if (width < 2) continue;
        const auto t = detail::power_below(width);
        for (auto p = t; p > 0; p /= 2) {
            auto q = t;
            std::size_t r = 0, d = p;
            while (d > 0) {
                for (std::size_t i = 0; i < width - d; ++i)
                    if ((i & p) == r) gates.push_back({offset + i, offset + i + d, false});
                d = q - p;
                q /= 2;
                r = p;
            }
        }
    }

    // Stage 2: pair a left block with the rightmost remaining block in reverse
    // order. Ascending left + reversed right is a bitonic sequence of 2*width
    // values. The first bitonic split needs only min outputs: they contain the
    // smallest width values and form a bitonic block for merge_topk_block().
    // Survivors occupy the leftmost ceil(blocks/2) blocks. If blocks is odd,
    // the unpaired middle block stays sorted and survives without any gates.
    for (auto blocks = network.size / width; blocks > 1; blocks = (blocks + 1) / 2) {
        for (std::size_t i = 0; i < blocks / 2; ++i) {
            for (std::size_t j = 0; j < width; ++j)
                gates.push_back({i * width + j, blocks * width - i * width - j - 1, true});
            detail::merge_topk_block(gates, i * width, width);
        }
    }

    // Stage 3: assign gates to layers while preserving dependencies on BOTH
    // input positions. Even min-only gates read b, so mark b as used too.
    // No two gates in a layer touch the same slot. The evaluator may run them
    // concurrently, with a barrier before advancing to the next layer.
    std::vector<std::size_t> last(network.size, 0);
    for (const auto& gate : gates) {
        const auto layer = std::max(last[gate.a], last[gate.b]);
        if (network.layers.size() <= layer) network.layers.resize(layer + 1);
        network.layers[layer].push_back(gate);
        last[gate.a] = last[gate.b] = layer + 1;
    }
    return network;
}

} // namespace privateann::he
