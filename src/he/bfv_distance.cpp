#include "he/bfv_distance.hpp"
#include "utils/bfv_encoding.hpp"

#include <algorithm>

namespace privateann::he {

BfvCiphertext squared_l2(const EvaluationKeys& keys,
                         heongpu::HEEncoder<bfv_scheme>& encoder,
                         heongpu::HEArithmeticOperator<bfv_scheme>& evaluator,
                         BfvQuery& query,
                         const std::vector<std::vector<std::uint64_t>>& record_columns) {
    keys.bfv_parameters.validate_distance_dimension(query.coordinates.size());
    if (query.coordinates.size() != record_columns.size())
        throw std::invalid_argument("Query and record dimensions differ");
    const auto n = keys.bfv_parameters.slots;
    const std::uint64_t t = keys.bfv_parameters.plaintext_modulus;
    auto validate_ciphertext = [&](BfvCiphertext& ciphertext) {
        if (ciphertext.ring_size() != static_cast<int>(n) || ciphertext.size() != 2 ||
            ciphertext.in_ntt_domain() ||
            ciphertext.coeff_modulus_count() != keys.bfv_context->get_ciphertext_modulus_count())
            throw std::invalid_argument("Expected a two-component coefficient-form BFV query");
    };
    validate_ciphertext(query.squared_norm);
    for (std::size_t dim = 0; dim < record_columns.size(); ++dim) {
        const auto& column = record_columns[dim];
        if (column.size() != keys.bfv_parameters.slots ||
            std::any_of(column.begin(), column.end(), [&](auto value) {
                return value > keys.bfv_parameters.max_coordinate;
            }))
            throw std::invalid_argument("Each record column needs N bounded coordinates");
        validate_ciphertext(query.coordinates[dim]);
    }

    BfvCiphertext total(keys.bfv_context);
    // SERVER: replicate the encrypted constant ||q||^2 into all coefficients
    // by multiplying by 1 + X + ... + X^(N-1). No slot mask or rotation is used.
    auto ones = utils::encode_bfv_coefficients(keys.bfv_context, encoder, std::vector<std::uint64_t>(n, 1));
    evaluator.multiply_plain(query.squared_norm, ones, total);
    std::vector<std::uint64_t> column(n), record_norms(n, 0);
    for (std::size_t dim = 0; dim < query.coordinates.size(); ++dim) {
        // SERVER: constant q[d] times a plaintext polynomial scales each of its
        // coefficients independently. Encode -2*x modulo t and accumulate ||x||^2.
        for (std::size_t i = 0; i < n; ++i) {
            const auto value = record_columns[dim][i];
            column[i] = (t - (2 * value) % t) % t;
            record_norms[i] += value * value;
        }
        auto plain = utils::encode_bfv_coefficients(keys.bfv_context, encoder, column);
        BfvCiphertext term(keys.bfv_context);
        evaluator.multiply_plain(query.coordinates[dim], plain, term);
        evaluator.add_inplace(total, term);
    }
    // SERVER: add the public record norms. Intermediate arithmetic is modulo t;
    // the configured distance bound ensures every final distance is exact.
    auto norms = utils::encode_bfv_coefficients(keys.bfv_context, encoder, record_norms);
    evaluator.add_plain_inplace(total, norms);
    return total;
}

} // namespace privateann::he
