#include <cuda_profiler_api.h>
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_version.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// CUPTI activity timestamps measure actual GPU kernel intervals. Use the
// concurrent-kernel activity kind so collection does not serialize kernels.
// Only the explicit profiler range is collected; timings include profiling
// overhead and must not replace the uninstrumented benchmark.
namespace {
struct Kernel {
    uint64_t start, end;
    uint32_t device, stream, grid, block, shared;
    std::string name;
};
struct Runtime { uint64_t start, end; uint32_t thread, cbid; };
std::mutex mutex;
std::vector<Kernel> kernels;
std::vector<Runtime> calls;
size_t dropped = 0;
bool initialized = false;
uint64_t range_start = 0, range_end = 0;
void check(CUptiResult status, const char* operation = "activity callback") {
    if (status != CUPTI_SUCCESS) {
        const char* message = nullptr;
        cuptiGetResultString(status, &message);
        std::fprintf(stderr, "CUPTI failure in %s: %s\n", operation, message ? message : "unknown");
        const char* detail = nullptr;
        cuptiGetErrorMessage(status, &detail);
        std::fprintf(stderr, "CUPTI detail: %s\n", detail ? detail : "unavailable");
        std::abort();
    }
}
void CUPTIAPI request(uint8_t** buffer, size_t* size, size_t* records) {
    *size = 4 * 1024 * 1024;
    *buffer = static_cast<uint8_t*>(std::malloc(*size));
    if (!*buffer) std::abort();
    *records = 0;
}
void CUPTIAPI complete(CUcontext context, uint32_t stream, uint8_t* buffer,
                       size_t, size_t valid) {
    std::lock_guard<std::mutex> lock(mutex);
    CUpti_Activity* record = nullptr;
    for (;;) {
        const auto status = cuptiActivityGetNextRecord(buffer, valid, &record);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        check(status);
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            const auto* k = reinterpret_cast<const CUpti_ActivityKernel10*>(record);
            kernels.push_back({k->start, k->end, k->deviceId, k->streamId,
                uint32_t(k->gridX * k->gridY * k->gridZ),
                uint32_t(k->blockX * k->blockY * k->blockZ),
                uint32_t(k->staticSharedMemory + k->dynamicSharedMemory), k->name});
        } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            const auto* api = reinterpret_cast<const CUpti_ActivityAPI*>(record);
            calls.push_back({api->start, api->end, api->threadId, api->cbid});
        }
    }
    size_t count = 0;
    check(cuptiActivityGetNumDroppedRecords(context, stream, &count));
    dropped += count;
    std::free(buffer);
}
template<class T> T next(const char* name) {
    const auto pointer = dlsym(RTLD_NEXT, name);
    if (!pointer) std::abort();
    return reinterpret_cast<T>(pointer);
}
struct EarlyInitialization {
    EarlyInitialization() {
        uint32_t version = 0;
        check(cuptiGetVersion(&version), "get version before CUDA context");
        check(cuptiActivityRegisterCallbacks(request, complete), "early register callbacks");
        check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "early enable kernels");
        check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME), "early enable runtime");
        initialized = true;
    }
} early_initialization;
}
extern "C" cudaError_t CUDARTAPI cudaProfilerStart() {
    if (!initialized) {
        check(cuptiActivityRegisterCallbacks(request, complete), "register callbacks");
        initialized = true;
    }
    check(cuptiGetTimestamp(&range_start), "range start");
    static const auto real = next<decltype(&cudaProfilerStart)>("cudaProfilerStart");
    return real();
}
extern "C" cudaError_t CUDARTAPI cudaProfilerStop() {
    static const auto real = next<decltype(&cudaProfilerStop)>("cudaProfilerStop");
    const auto result = real();
    check(cuptiGetTimestamp(&range_end), "range end");
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
    check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    const auto* prefix = std::getenv("PRIVATEIMPL_CUPTI_REPORT");
    const std::string path = prefix ? prefix : "logs/cupti_overlap";
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream gpu(path + "_kernels.csv"), cpu(path + "_runtime.csv"), raw(path + "_all_kernels.csv");
    gpu << "start_ns,end_ns,device,stream,blocks,threads_per_block,shared_bytes,name\n";
    raw << "start_ns,end_ns,device,stream,blocks,threads_per_block,shared_bytes,name\n";
    size_t captured_kernels = 0, captured_calls = 0;
    for (const auto& k : kernels) {
        raw << k.start << ',' << k.end << ',' << k.device << ',' << k.stream << ','
            << k.grid << ',' << k.block << ',' << k.shared << ',' << k.name << '\n';
        if (k.start < range_start || k.end > range_end || k.end <= k.start) continue;
        ++captured_kernels;
        gpu << k.start << ',' << k.end << ',' << k.device << ',' << k.stream << ','
            << k.grid << ',' << k.block << ',' << k.shared << ',' << k.name << '\n';
    }
    cpu << "start_ns,end_ns,thread,api\n";
    for (const auto& c : calls) {
        if (c.start < range_start || c.end > range_end || c.end <= c.start) continue;
        ++captured_calls;
        const char* name = nullptr;
        check(cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, c.cbid, &name));
        cpu << c.start << ',' << c.end << ',' << c.thread << ',' << name << '\n';
    }
    std::fprintf(stderr, "CUPTI records: kernels=%zu runtime=%zu dropped=%zu total_kernels=%zu total_runtime=%zu\n",
                 captured_kernels, captured_calls, dropped, kernels.size(), calls.size());
    std::fprintf(stderr, "CUPTI range: %llu .. %llu ns\n",
        (unsigned long long)range_start, (unsigned long long)range_end);
    if (!gpu || !cpu || !raw) std::abort();
    return result;
}
