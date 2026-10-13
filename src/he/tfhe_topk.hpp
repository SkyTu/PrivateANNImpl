#pragma once

#include "he/tfhe_select.hpp"

namespace privateann::he {

// Returns the k smallest encrypted records, ordered by distance; ties are arbitrary.
// Top-u over centroids uses this same function with k=u.
// Distances use FheUint32 (FheUint64 for a full uint32 value plus sentinel); IDs use FheUint32.
// Every record in a batch must have the same score/ID layout. Gates within a
// layer run in parallel; consecutive layers are synchronized.
// Gates compare distances with native gt and conditionally select both fields.
// Input count and k are public.
// Returned records are internal server state, not a final ANN client response.
std::vector<tfhe_rs::Record> topk(tfhe_rs::Evaluator& evaluator,
                                 std::vector<tfhe_rs::Record> records, std::size_t k);

} // namespace privateann::he
