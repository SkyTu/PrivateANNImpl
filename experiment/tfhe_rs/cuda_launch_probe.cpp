#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

// Diagnostic LD_PRELOAD shim. Forward every CUDA launch unchanged and record
// only public launch metadata within cudaProfilerStart/Stop. It does not inspect
// ciphertexts, synchronize streams, change allocation, or time GPU kernels.
namespace {
std::atomic<bool> capturing{false};
std::atomic<std::uint64_t> malloc_calls{0}, free_calls{0}, allocated_bytes{0};
std::mutex mutex;
using Shape = std::tuple<bool, std::uintptr_t, unsigned, unsigned, unsigned,
                         unsigned, unsigned, unsigned, std::size_t>;
struct Count { std::uint64_t calls = 0, null_streams = 0; std::set<std::uintptr_t> streams; };
std::map<Shape, Count> shapes;

template<class Function> Function next(const char* name) {
    const auto pointer = dlsym(RTLD_NEXT, name);
    if (!pointer) { std::fprintf(stderr, "Cannot resolve CUDA symbol %s\n", name); std::abort(); }
    return reinterpret_cast<Function>(pointer);
}
void record(bool cooperative, const void* function, dim3 grid, dim3 block,
            std::size_t shared, cudaStream_t stream) {
    if (!capturing.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(mutex);
    auto& value = shapes[{cooperative, reinterpret_cast<std::uintptr_t>(function),
                         grid.x, grid.y, grid.z, block.x, block.y, block.z, shared}];
    ++value.calls;
    value.null_streams += stream == nullptr;
    value.streams.insert(reinterpret_cast<std::uintptr_t>(stream));
}
void write_report() {
    const auto* path = std::getenv("PRIVATEIMPL_LAUNCH_REPORT");
    if (!path) path = "logs/tfhe_launch_shapes.csv";
    std::ofstream out(path);
    out << "kind,function_address,grid_x,grid_y,grid_z,block_x,block_y,block_z,"
           "total_blocks,threads_per_block,dynamic_shared_bytes,calls,streams,null_stream_calls\n";
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& [shape, count] : shapes) {
        const auto& [cooperative, function, gx, gy, gz, bx, by, bz, shared] = shape;
        out << (cooperative ? "cooperative" : "ordinary") << ',' << function << ','
            << gx << ',' << gy << ',' << gz << ',' << bx << ',' << by << ',' << bz << ','
            << std::uint64_t(gx) * gy * gz << ',' << std::uint64_t(bx) * by * bz << ','
            << shared << ',' << count.calls << ',' << count.streams.size() << ',' << count.null_streams << '\n';
    }
    out.close();
    std::fprintf(stderr, "%s launch metadata: %s\n", out ? "Saved" : "Failed to save", path);
    std::ofstream allocations(std::string(path) + ".allocations.csv");
    allocations << "cuda_malloc_calls,cuda_free_calls,cuda_malloc_bytes\n"
                << malloc_calls.load() << ',' << free_calls.load() << ',' << allocated_bytes.load() << '\n';
}
} // namespace

extern "C" cudaError_t CUDARTAPI cudaProfilerStart() {
    static const auto real = next<decltype(&cudaProfilerStart)>("cudaProfilerStart");
    const auto status = real();
    if (status == cudaSuccess) {
        std::lock_guard<std::mutex> lock(mutex);
        shapes.clear();
        malloc_calls = 0;
        free_calls = 0;
        allocated_bytes = 0;
        capturing = true;
    }
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaProfilerStop() {
    capturing = false;
    write_report();
    static const auto real = next<decltype(&cudaProfilerStop)>("cudaProfilerStop");
    return real();
}
extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                                  void** arguments, std::size_t shared, cudaStream_t stream) {
    static const auto real = next<decltype(&cudaLaunchKernel)>("cudaLaunchKernel");
    const auto status = real(function, grid, block, arguments, shared, stream);
    if (status == cudaSuccess) record(false, function, grid, block, shared, stream);
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void* function, dim3 grid, dim3 block,
                                                             void** arguments, std::size_t shared, cudaStream_t stream) {
    static const auto real = next<decltype(&cudaLaunchCooperativeKernel)>("cudaLaunchCooperativeKernel");
    const auto status = real(function, grid, block, arguments, shared, stream);
    if (status == cudaSuccess) record(true, function, grid, block, shared, stream);
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaMalloc(void** ptr, std::size_t bytes) {
    static const auto real = next<decltype(&cudaMalloc)>("cudaMalloc");
    const auto status = real(ptr, bytes);
    if (capturing.load() && status == cudaSuccess) {
        ++malloc_calls;
        allocated_bytes += bytes;
    }
    return status;
}
extern "C" cudaError_t CUDARTAPI cudaFree(void* ptr) {
    static const auto real = next<decltype(&cudaFree)>("cudaFree");
    const auto status = real(ptr);
    if (capturing.load() && status == cudaSuccess) ++free_calls;
    return status;
}
