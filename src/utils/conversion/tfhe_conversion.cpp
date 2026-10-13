#include "utils/conversion/tfhe_conversion.hpp"
#include "he/detail/tfhe_state.hpp"
#include <cstdlib>
#include <dlfcn.h>

namespace privateann::utils {
using namespace he;
using namespace he::tfhe_rs;
namespace {
struct BridgeBuffer { std::uint8_t* pointer = nullptr; std::size_t length = 0; };
struct BridgeApi {
    const char* (*last_error)();
    int (*key_new)(const std::uint64_t*, std::size_t, std::uint64_t,
                   const std::uint8_t*, std::size_t, void**);
    void (*key_destroy)(void*);
    int (*convert)(const void*, const std::uint64_t*, std::size_t,
                   const std::uint64_t*, unsigned, unsigned, BridgeBuffer*);
    void (*buffer_destroy)(BridgeBuffer);
    BridgeApi() {
        const char* path = std::getenv("PRIVATEIMPL_BFV_TFHE_BRIDGE");
        if (!path || !*path) path = PRIVATEIMPL_BFV_TFHE_BRIDGE_DEFAULT;
        void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (!library) throw LibraryUnavailable(std::string("Cannot load BFV bridge: ") +
            dlerror() + ". Run bash third_party/build_bfv_tfhe_bridge.sh");
        try {
#define LOAD_BRIDGE(member, symbol) \
            member = reinterpret_cast<decltype(member)>(dlsym(library, symbol)); \
            if (!member) throw LibraryUnavailable("BFV bridge symbol missing: " symbol);
            LOAD_BRIDGE(last_error, "bfv_tfhe_last_error")
            LOAD_BRIDGE(key_new, "bfv_tfhe_key_new")
            LOAD_BRIDGE(key_destroy, "bfv_tfhe_key_destroy")
            LOAD_BRIDGE(convert, "bfv_tfhe_convert_split")
            LOAD_BRIDGE(buffer_destroy, "bfv_tfhe_buffer_destroy")
#undef LOAD_BRIDGE
        } catch (...) { dlclose(library); throw; }
        dlclose(library);
    }
    static BridgeApi& instance() { static BridgeApi api; return api; }
    void check(int status) const {
        if (status) throw std::runtime_error(std::string("BFV-to-TFHE conversion: ") + last_error());
    }
};

// CLIENT ONLY: the temporary native key serialization never reaches Server.
// Volatile writes prevent the compiler from dropping the wipe before release.
struct ClientKeyBuffer {
    DynamicBuffer buffer{};
    ~ClientKeyBuffer() {
        volatile std::uint8_t* bytes = buffer.pointer;
        for (std::size_t i = 0; i < buffer.length; ++i) bytes[i] = 0;
        Api::instance().destroy_dynamic_buffer(&buffer);
    }
};
} // namespace

std::shared_ptr<ConversionKey> TfheConversion::generate_key(
    const entity::TfheKeys& keys,
    const std::vector<std::uint64_t>& bfv_secret, std::uint32_t modulus) {
    auto& api = Api::instance();
    auto& bridge = BridgeApi::instance();
    ClientKeyBuffer encoded;
    api.check(api.client_key_serialize(keys.impl_->key.get(), &encoded.buffer), "client_key_serialize");
    void* raw = nullptr;
    const int status = bridge.key_new(bfv_secret.data(), bfv_secret.size(), modulus,
        encoded.buffer.pointer, encoded.buffer.length, &raw);
    auto out = std::make_shared<ConversionKey::Impl>();
    out->key = std::shared_ptr<void>(raw, bridge.key_destroy);
    bridge.check(status);
    if (!raw) throw std::runtime_error("Conversion key generation returned a null handle");
    out->tag = keys.impl_->evaluation->impl_->tag;
    return std::shared_ptr<ConversionKey>(new ConversionKey(std::move(out)));
}

std::vector<Record> TfheConversion::convert_lwe(
    const EvaluationKey& target, const ConversionKey& key,
    const std::vector<std::uint64_t>& samples, const std::vector<std::uint64_t>& ids,
    unsigned id_bits, unsigned drop_bits) {
    if (!target.impl_ || !key.impl_ || key.impl_->tag != target.impl_->tag)
        throw std::invalid_argument("BFV conversion key belongs to another TFHE client");
    if (ids.empty() || ids.size() > 4096 || samples.size() != ids.size() * 4097 ||
        (drop_bits != 0 && drop_bits != 7))
        throw std::invalid_argument("Invalid BFV LWE buffer or truncation setting");
    const RecordLayout layout(23 - drop_bits, id_bits);
    auto& bridge = BridgeApi::instance();
    struct Packet {
        BridgeBuffer buffer;
        ~Packet() { BridgeApi::instance().buffer_destroy(buffer); }
    } packet;
    bridge.check(bridge.convert(key.impl_->key.get(), samples.data(), ids.size(), ids.data(),
                                 drop_bits, id_bits, &packet.buffer));
    auto& api = Api::instance();
    std::vector<Record> out;
    out.reserve(ids.size());
    std::size_t offset = 0;
    const auto read_field = [&]() {
        if (offset > packet.buffer.length || packet.buffer.length - offset < 8)
            throw std::runtime_error("Truncated split conversion output");
        std::uint64_t length = 0;
        for (unsigned j = 0; j < 8; ++j)
            length |= std::uint64_t(packet.buffer.pointer[offset++]) << (8 * j);
        if (length > packet.buffer.length - offset)
            throw std::runtime_error("Invalid split conversion output length");
        DynamicBufferView view{packet.buffer.pointer + offset, static_cast<std::size_t>(length)};
        auto field = create(api.fhe_uint32_destroy, "deserialize converted field",
                            api.fhe_uint32_deserialize, view);
        offset += length;
        return field;
    };
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto distance = read_field();
        auto id = read_field();
        out.push_back(Record(std::make_shared<Record::Impl>(
            std::move(distance), std::move(id), layout, target.impl_->tag)));
    }
    if (offset != packet.buffer.length) throw std::runtime_error("Extra bytes in conversion output");
    return out;
}

} // namespace privateann::utils
