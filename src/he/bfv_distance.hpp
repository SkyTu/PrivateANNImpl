#pragma once

#include "utils/types.hpp"

namespace privateann::he {

// SERVER: compute Enc(||q||^2) - 2 * sum_d Enc(q[d]) * x[d] + ||x||^2.
// Each input query ciphertext encrypts a constant polynomial. Database column d
// packs x[d][i] at polynomial coefficient i; output coefficient i is distance i.
// All columns must contain exactly N records (pad the last batch publicly).
// Only ciphertext-plaintext multiplication and addition are used; no relin key.
// HEonGPU takes mutable references, but the query's message is left unchanged.
BfvCiphertext squared_l2(const EvaluationKeys& keys,
                         heongpu::HEEncoder<bfv_scheme>& encoder,
                         heongpu::HEArithmeticOperator<bfv_scheme>& evaluator,
                         BfvQuery& query,
                         const std::vector<std::vector<std::uint64_t>>& record_columns);

} // namespace privateann::he
