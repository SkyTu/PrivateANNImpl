#include "utils/detail/tfhe_rs_api.hpp"
#include <cstdlib>
#include <dlfcn.h>
#include <string>

namespace privateann::he::tfhe_rs {

Api::Api() {
    // Resolve the installed Rust GPU library once. An explicit environment
    // override is useful when testing another build of the same C ABI.
    const char* path = std::getenv("PRIVATEIMPL_TFHE_RS_LIBRARY");
    if (!path || !*path) path = PRIVATEIMPL_TFHE_RS_LIBRARY_DEFAULT;
    // Rust uses thread-local state. Keep its code mapped until process exit,
    // including after C++ objects release their references.
    void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (!library)
        throw LibraryUnavailable(std::string("Cannot load TFHE-rs GPU C API: ") +
            dlerror() + ". Run bash third_party/install_tfhe_rs.sh");
    try {
#define PRIVATEIMPL_LOAD_API(result, name, args) \
        name = reinterpret_cast<decltype(name)>(dlsym(library, #name)); \
        if (!name) throw LibraryUnavailable("TFHE-rs GPU C API symbol missing: " #name);
        PRIVATEIMPL_TFHE_RS_API(PRIVATEIMPL_LOAD_API)
#undef PRIVATEIMPL_LOAD_API
    } catch (...) {
        dlclose(library);
        throw;
    }
    // NODELETE keeps code and TLS alive; release the dlopen reference itself.
    dlclose(library);
}

Api& Api::instance() {
    static Api api;
    return api;
}

void Api::check(int status, const char* operation) const {
    // The C boundary returns status codes; expose failures as C++ exceptions
    // with the original TFHE-rs error text for the calling he/entity/test.
    if (status != 0) {
        const char* error = tfhe_error_get_last();
        throw std::runtime_error(std::string(operation) + ": " +
                                 (error ? error : "TFHE-rs C API failed"));
    }
}

} // namespace privateann::he::tfhe_rs
