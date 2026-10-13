#include "utils/convert.hpp"
#include "utils/conversion/bfv_lwe.hpp"
#include "utils/conversion/tfhe_conversion.hpp"
#include <stdexcept>

namespace privateann::utils {
BfvToTfheConverter::BfvToTfheConverter(EvaluationKeys keys) : keys_(std::move(keys)) {
    if (!keys_.bfv_context || !keys_.tfhe_key || !keys_.conversion_key)
        throw std::invalid_argument("Conversion requires BFV context and TFHE/conversion evaluation keys");
    keys_.bfv_parameters.validate();
    if (keys_.bfv_parameters.slots != 4096 || keys_.bfv_parameters.plaintext_modulus != 8380417 ||
        keys_.bfv_context->get_poly_modulus_degree() != 4096 ||
        keys_.bfv_context->get_plain_modulus().value != 8380417)
        throw std::invalid_argument("Conversion requires a matching N=4096, t=8380417 BFV context");
}

std::vector<TfheRecord> BfvToTfheConverter::convert(BfvCiphertext& distances,
    const std::vector<std::size_t>& coefficients, const std::vector<std::uint64_t>& ids,
    unsigned id_bits, unsigned drop_bits) const {
    if (coefficients.empty() || coefficients.size() != ids.size() ||
        coefficients.size() > keys_.bfv_parameters.slots ||
        (drop_bits != 0 && drop_bits != 7))
        throw std::invalid_argument("Invalid conversion indices, IDs or truncation setting");
    const he::RecordLayout layout(23 - drop_bits, id_bits);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        layout.validate(0, ids[i]);
        if (coefficients[i] >= keys_.bfv_parameters.slots)
            throw std::invalid_argument("Conversion coefficient index exceeds the ring degree");
    }
    // SERVER: sample extraction and cross-key/radix conversion use evaluation
    // material only. Source distances remain encrypted throughout both stages.
    BfvLweExtractor extractor(keys_);
    std::vector<std::uint64_t> samples;
    samples.reserve(coefficients.size() * (keys_.bfv_parameters.slots + 1));
    for (auto coefficient : coefficients) {
        const auto sample = extractor.extract_coefficient(distances, coefficient);
        samples.insert(samples.end(), sample.mask.begin(), sample.mask.end());
        samples.push_back(sample.body);
    }
    return TfheConversion::convert_lwe(
        *keys_.tfhe_key, *keys_.conversion_key, samples, ids, id_bits, drop_bits);
}

} // namespace privateann::utils
