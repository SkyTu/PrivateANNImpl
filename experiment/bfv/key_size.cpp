#include "he/entity/client.hpp"

#include <iostream>
#include <ostream>
#include <stdexcept>
#include <streambuf>

namespace {
// Count the library's serialized bytes without retaining or printing key data.
class CountingBuffer : public std::streambuf {
public:
    std::size_t bytes = 0;
protected:
    std::streamsize xsputn(const char*, std::streamsize count) override {
        bytes += static_cast<std::size_t>(count);
        return count;
    }
    int_type overflow(int_type value) override {
        if (!traits_type::eq_int_type(value, traits_type::eof())) ++bytes;
        return traits_type::not_eof(value);
    }
};

template <class Key>
std::size_t serialized_size(const Key& key) {
    CountingBuffer buffer;
    std::ostream sink(&buffer);
    key.save(sink);
    if (!sink) throw std::runtime_error("Key size measurement failed");
    return buffer.bytes;
}
} // namespace

int main() {
    using namespace privateann;
    try {
        // Reuse the exact default context of the 1M/128D distance benchmark.
        // Generate a fresh probe pair with those parameters; size does not
        // depend on the sampled key values or on the number of database rows.
        entity::Client client;
        const auto keys = client.evaluation_keys();
        const auto context = keys.bfv_context;
        if (keys.relin_key || keys.rotation_key || keys.tfhe_key || keys.conversion_key)
            throw std::runtime_error("Expected BFV linear-distance keys only");
        heongpu::HEKeyGenerator<bfv_scheme> generator(context);
        heongpu::Secretkey<bfv_scheme> secret(context);
        heongpu::Publickey<bfv_scheme> public_key(context);
        generator.generate_secret_key(secret);
        generator.generate_public_key(public_key, secret);
        const auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        if (!secret.is_on_device() || !public_key.is_on_device())
            throw std::runtime_error("Probe keys must use device storage");
        const auto secret_bytes = std::size_t(secret.ring_size()) *
            secret.coeff_modulus_count() * sizeof(::Data64);
        const auto public_bytes = 2 * std::size_t(public_key.ring_size()) *
            public_key.coeff_modulus_count() * sizeof(::Data64);
        const auto secret_serialized = serialized_size(secret);
        const auto public_serialized = serialized_size(public_key);
        if (secret_serialized < secret_bytes || public_serialized < public_bytes)
            throw std::runtime_error("Serialized key is shorter than its raw payload");

        std::cout << "{\n  \"status\": \"PASS\",\n"
                  << "  \"ring_degree\": " << context->get_poly_modulus_degree() << ",\n"
                  << "  \"plaintext_modulus\": " << keys.bfv_parameters.plaintext_modulus << ",\n"
                  << "  \"key_rns_moduli\": " << context->get_key_modulus_count() << ",\n"
                  << "  \"ciphertext_rns_moduli\": " << context->get_ciphertext_modulus_count() << ",\n"
                  << "  \"coefficient_word_bytes\": " << sizeof(::Data64) << ",\n"
                  << "  \"secret_key_payload_bytes\": " << secret_bytes << ",\n"
                  << "  \"public_key_payload_bytes\": " << public_bytes << ",\n"
                  << "  \"secret_key_serialized_bytes\": " << secret_serialized << ",\n"
                  << "  \"public_key_serialized_bytes\": " << public_serialized << ",\n"
                  << "  \"total_pair_payload_bytes\": " << secret_bytes + public_bytes << ",\n"
                  << "  \"relinearization_key_bytes\": 0,\n"
                  << "  \"rotation_key_bytes\": 0,\n"
                  << "  \"server_distance_evaluation_key_bytes\": 0,\n"
                  << "  \"scope\": \"BFV linear distance only; excludes context, queries, ciphertexts, allocator overhead and TFHE conversion keys\"\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
