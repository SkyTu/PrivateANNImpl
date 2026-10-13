#include "he/entity/tfhe_keys.hpp"
#include "he/detail/tfhe_state.hpp"

namespace privateann::entity {
using namespace he::tfhe_rs;

TfheKeys::TfheKeys() : impl_(std::make_shared<Impl>()) {
    auto& api = Api::instance();
    // CLIENT: generate native TFHE-rs keys using the upstream GPU C example's
    // default configuration. ConfigBuilder and Config are consumed by the API.
    ConfigBuilder* builder = nullptr;
    api.check(api.config_builder_default(&builder), "config_builder_default");
    Config* config = nullptr;
    api.check(api.config_builder_build(builder, &config), "config_builder_build");
    impl_->key = create(api.client_key_destroy, "client_key_generate",
                         api.client_key_generate, config);
    auto evaluation = std::make_shared<EvaluationKey::Impl>();
    evaluation->key = create(api.compressed_server_key_destroy, "compressed_server_key_new",
                              api.compressed_server_key_new, impl_->key.get());
    impl_->evaluation = std::shared_ptr<EvaluationKey>(new EvaluationKey(std::move(evaluation)));
}
TfheKeys::~TfheKeys() = default;

std::shared_ptr<EvaluationKey> TfheKeys::evaluation_key() const { return impl_->evaluation; }

} // namespace privateann::entity
