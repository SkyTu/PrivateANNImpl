#include "utils/detail/bfv_secret.hpp"
#include <stdexcept>

namespace privateann::utils::detail {
namespace {
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("BFV client diagnostic: ") + cudaGetErrorString(error));
}

} // namespace

SecretWords coefficient_secret(const BfvContext& context,
                               heongpu::Secretkey<bfv_scheme>& secret_key) {
    const auto n = context->get_poly_modulus_degree();
    // CLIENT ONLY: recover the original ternary coefficient secret from the
    // library's NTT representation. Use its own inverse NTT and public tables.
    // This private helper is used only by client setup and client verification.
    std::vector<::Modulus64> primes{context->get_key_modulus().front()};
    const int power = context->get_log_poly_modulus_degree();
    auto roots = heongpu::generate_primitive_root_of_unity(n, primes);
    heongpu::DeviceVector<::Root64> tables(heongpu::generate_intt_table(roots, primes, power));
    heongpu::DeviceVector<::Modulus64> moduli(primes);
    heongpu::DeviceVector<::Ninverse64> inverse(heongpu::generate_n_inverse(n, primes));
    heongpu::DeviceVector<::Data64> secret(n);
    cuda_check(cudaMemcpy(secret.data(), secret_key.data(),
                                 n * sizeof(::Data64), cudaMemcpyDeviceToDevice));
    gpuntt::ntt_rns_configuration<::Data64> config{};
    config.n_power = power;
    config.ntt_type = gpuntt::INVERSE;
    config.ntt_layout = gpuntt::PerPolynomial;
    config.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
    config.zero_padding = false;
    config.mod_inverse = inverse.data();
    config.stream = cudaStreamDefault;
    gpuntt::GPU_INTT_Inplace(secret.data(), tables.data(), moduli.data(), config, 1, 1);
    std::vector<::Data64> coefficients(n);
    cuda_check(cudaMemcpy(coefficients.data(), secret.data(),
                                 n * sizeof(::Data64), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemset(secret.data(), 0, n * sizeof(::Data64)));
    bool valid = true;
    for (auto& value : coefficients) {
        if (value == primes.front().value - 1) value = std::uint64_t(-1);
        else if (value != 0 && value != 1) valid = false;
    }
    if (!valid) {
        OPENSSL_cleanse(coefficients.data(), coefficients.size() * sizeof(::Data64));
        throw std::runtime_error("Inverse NTT did not recover a ternary BFV secret");
    }
    return SecretWords(std::move(coefficients));
}

} // namespace privateann::utils::detail
