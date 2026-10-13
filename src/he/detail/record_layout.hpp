#pragma once

#include "utils/parameters.hpp"

namespace privateann::he {

struct PlainRecord {
    std::uint32_t score;
    std::uint32_t id;
    bool valid;
};

// Distance and ID are separate ciphertexts. Only distance controls ordering;
// equal distances may return either record. A distance sentinel encodes invalid
// slots, so no extra encrypted validity field is needed during selection.
class RecordLayout {
public:
    RecordLayout(unsigned score_bits = 32, unsigned id_bits = 32)
        : score_bits_(score_bits), id_bits_(id_bits) {
        validate_word(0, score_bits);
        validate_word(0, id_bits);
    }
    unsigned distance_ciphertext_bits() const { return score_bits_ < 32 ? 32 : 64; }
    unsigned id_ciphertext_bits() const { return 32; }
    std::uint64_t padding() const { return std::uint64_t{1} << score_bits_; }
    std::uint64_t max_id() const { return (std::uint64_t{1} << id_bits_) - 1; }
    void validate(std::uint64_t score, std::uint64_t id) const {
        validate_word(score, score_bits_);
        validate_word(id, id_bits_);
    }
    PlainRecord decode(std::uint64_t score, std::uint64_t id) const {
        if (score > padding()) throw std::invalid_argument("Malformed TFHE distance");
        if (score == padding()) return {0, 0, false};
        validate(score, id);
        return {static_cast<std::uint32_t>(score), static_cast<std::uint32_t>(id), true};
    }
    bool operator==(const RecordLayout& other) const {
        return score_bits_ == other.score_bits_ && id_bits_ == other.id_bits_;
    }
    bool operator!=(const RecordLayout& other) const { return !(*this == other); }
private:
    unsigned score_bits_, id_bits_;
};

} // namespace privateann::he
