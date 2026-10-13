#pragma once

#include "he/detail/topk_schedule.hpp"
#include <cstddef>
#include <utility>
#include <vector>

namespace privateann::he {

// This code only schedules native comparison and selection. Encrypted comparison
// is implemented entirely by TFHE-rs. All control flow uses public metadata.
template <class Operations, class Value>
std::vector<Value> topk_network(Operations& operations, std::vector<Value> values,
                                std::size_t k) {
    const auto network = topk_schedule(values.size(), k);
    // padding() uses a distance sentinel larger than every valid distance.
    // Public padding therefore cannot displace a real result from the first k.
    if (network.size != values.size())
        values.resize(network.size, operations.padding(values.front()));
    // Each layer touches disjoint slots. The executor must complete the layer
    // before starting the next one. Ciphertext values never determine wiring.
    operations.execute_layers(network, [&](const auto& comparator) {
        auto& a = values[comparator.a];
        auto& b = values[comparator.b];
        if (comparator.min_only) {
            // The larger output belongs to a discarded block. Do not evaluate
            // max or overwrite b; later layers never use that discarded block.
            a = operations.minimum(a, b);
        } else {
            auto [minimum, maximum] = operations.minmax(a, b);
            a = std::move(minimum);
            b = std::move(maximum);
        }
    });
    values.erase(values.begin() + static_cast<std::ptrdiff_t>(k), values.end());
    return values;
}

} // namespace privateann::he
