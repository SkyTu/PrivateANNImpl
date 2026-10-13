#include "he/tfhe_topk.hpp"
#include "he/detail/selection_network.hpp"

namespace privateann::he {

std::vector<tfhe_rs::Record> topk(tfhe_rs::Evaluator& evaluator,
                                 std::vector<tfhe_rs::Record> records, std::size_t k) {
    validate_topk_size(records.size(), k);
    evaluator.validate(records);
    return topk_network(evaluator, std::move(records), k);
}

} // namespace privateann::he
