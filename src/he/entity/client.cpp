#include "he/entity/client.hpp"
#include "he/entity/detail/client_state.hpp"
#include "utils/detail/bfv_secret.hpp"
#include "utils/conversion/tfhe_conversion.hpp"
#include <stdexcept>

namespace privateann::entity {
namespace {
BfvContext make_bfv_context(const BfvParameters& parameters) {
    parameters.validate();
    auto context = heongpu::GenHEContext<bfv_scheme>();
    context->set_poly_modulus_degree(parameters.slots);
    context->set_coeff_modulus_default_values(1);
    context->set_plain_modulus(static_cast<int>(parameters.plaintext_modulus));
    // HEonGPU's default reserves 90% of free VRAM. Both backends share this
    // process, so leave room for TFHE-rs keys and integer PBS workspaces.
    auto pool = heongpu::MemoryPoolConfig::Defaults();
    pool.initial_device_fraction = 0.05f;
    pool.max_device_fraction = 0.25f;
    context->generate(pool);
    return context;
}

} // namespace

Client::Impl::Impl(BfvParameters config, bool enable_tfhe)
    : parameters(config), context(make_bfv_context(config)),
      secret_key(context), public_key(context) {
    // CLIENT KEY GENERATION: secret keys stay in this key owner.
    heongpu::HEKeyGenerator<bfv_scheme> keygen(context);
    keygen.generate_secret_key(secret_key);
    keygen.generate_public_key(public_key, secret_key);
    if (enable_tfhe) tfhe = std::make_unique<TfheKeys>();
}

Client::Client(BfvParameters parameters, bool enable_tfhe)
    : impl_(std::make_shared<Impl>(parameters, enable_tfhe)) {}
Client::~Client() = default;

EvaluationKeys Client::evaluation_keys() const {
    return {impl_->parameters, impl_->context, impl_->relin_key,
            impl_->tfhe ? impl_->tfhe->evaluation_key() : nullptr, impl_->rotation_key, impl_->conversion_key};
}

void Client::enable_bfv_lwe_extraction() {
    if (impl_->rotation_key) return;
    std::vector<int> shifts;
    for (std::size_t shift = 1; shift < impl_->parameters.slots / 2; shift *= 2)
        shifts.push_back(static_cast<int>(shift));
    auto key = std::make_shared<heongpu::Galoiskey<bfv_scheme>>(impl_->context, shifts);
    heongpu::HEKeyGenerator<bfv_scheme> keygen(impl_->context);
    keygen.generate_galois_key(*key, impl_->secret_key);
    impl_->rotation_key = std::move(key);
}

void Client::enable_bfv_relinearization() {
    if (impl_->relin_key) return;
    auto key = std::make_shared<heongpu::Relinkey<bfv_scheme>>(impl_->context);
    heongpu::HEKeyGenerator<bfv_scheme> keygen(impl_->context);
    keygen.generate_relin_key(*key, impl_->secret_key);
    impl_->relin_key = std::move(key);
}

const TfheKeys& Client::tfhe_keys() const {
    if (!impl_->tfhe) throw std::logic_error("Construct Client with enable_tfhe=true");
    return *impl_->tfhe;
}

void Client::enable_bfv_tfhe_conversion() {
    if (impl_->conversion_key) return;
    if (!impl_->tfhe) throw std::logic_error("Conversion requires Client(enable_tfhe=true)");
    if (impl_->parameters.slots != 4096 || impl_->parameters.plaintext_modulus != 8380417)
        throw std::invalid_argument("Conversion currently supports N=4096, t=8380417");
    // CLIENT KEY GENERATION: cross-key material is created while both secrets
    // remain in the client setup scope. The temporary BFV key is wiped on exit.
    auto secret = utils::detail::coefficient_secret(impl_->context, impl_->secret_key);
    impl_->conversion_key = utils::TfheConversion::generate_key(
        *impl_->tfhe, secret.values, impl_->parameters.plaintext_modulus);
}

} // namespace privateann::entity
