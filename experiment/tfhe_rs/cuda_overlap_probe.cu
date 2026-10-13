#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
struct Stamp { unsigned long long start, end; unsigned sm; };
__device__ unsigned long long global_time() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}
__global__ void mark_interval(Stamp* output, unsigned long long duration) {
    const auto start = global_time();
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    while (global_time() - start < duration) {}
    if (threadIdx.x == 0) output[blockIdx.x] = {start, global_time(), sm};
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace

// This deliberately tiny diagnostic proves only whether independent kernels
// CAN overlap on this GPU. It does not measure TFHE occupancy or its speedup.
// Device global timestamps avoid depending on unavailable Nsight activity data.
int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::runtime_error("Usage: cuda_overlap_probe [OUTPUT_CSV]");
        const std::filesystem::path path = argc == 2 ? argv[1] : "logs/cuda_overlap_probe.csv";
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        std::ofstream csv(path);
        if (!csv) throw std::runtime_error("Cannot open probe report");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU=" << properties.name << " SMs=" << properties.multiProcessorCount
                  << " concurrent_kernels=" << properties.concurrentKernels
                  << " cooperative_launch=" << properties.cooperativeLaunch << '\n';
        constexpr unsigned blocks = 8;
        unsigned long long duration = 5000000; // Five milliseconds per block.
        Stamp *device = nullptr;
        cudaStream_t streams[2];
        check(cudaMalloc(&device, sizeof(Stamp) * blocks * 2));
        for (auto& stream : streams) check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        // Load the kernel before recording; all allocation precedes both grids.
        mark_interval<<<1, 32, 0, streams[0]>>>(device, 0);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        csv << "kind,sm_count,concurrent_kernels,blocks_per_grid,threads_per_block,"
               "grid0_span_ms,grid1_span_ms,max_cross_grid_block_overlap_ms,"
               "distinct_sms_grid0,distinct_sms_grid1,distinct_sms_combined\n";
        for (bool cooperative : {false, true}) {
            if (cooperative && !properties.cooperativeLaunch) continue;
            for (unsigned grid = 0; grid < 2; ++grid) {
                auto* destination = device + grid * blocks;
                if (cooperative) {
                    void* arguments[] = {&destination, &duration};
                    check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(mark_interval),
                        dim3(blocks), dim3(32), arguments, 0, streams[grid]));
                } else {
                    mark_interval<<<blocks, 32, 0, streams[grid]>>>(destination, duration);
                    check(cudaGetLastError());
                }
            }
            check(cudaDeviceSynchronize());
            Stamp stamps[2 * blocks];
            check(cudaMemcpy(stamps, device, sizeof(stamps), cudaMemcpyDeviceToHost));
            unsigned long long begin[2] = {~0ull, ~0ull}, end[2] = {0, 0}, overlap = 0;
            std::set<unsigned> sms[2], combined;
            for (unsigned grid = 0; grid < 2; ++grid)
                for (unsigned i = 0; i < blocks; ++i) {
                    const auto& s = stamps[grid * blocks + i];
                    if (s.end < s.start || s.end - s.start < duration)
                        throw std::runtime_error("Invalid device timestamp interval");
                    begin[grid] = std::min(begin[grid], s.start);
                    end[grid] = std::max(end[grid], s.end);
                    sms[grid].insert(s.sm);
                    combined.insert(s.sm);
                }
            for (unsigned a = 0; a < blocks; ++a)
                for (unsigned b = blocks; b < 2 * blocks; ++b) {
                    const auto lo = std::max(stamps[a].start, stamps[b].start);
                    const auto hi = std::min(stamps[a].end, stamps[b].end);
                    if (hi > lo) overlap = std::max(overlap, hi - lo);
                }
            const auto name = cooperative ? "cooperative" : "ordinary";
            csv << name << ',' << properties.multiProcessorCount << ',' << properties.concurrentKernels
                << ',' << blocks << ",32," << (end[0] - begin[0]) / 1e6 << ',' << (end[1] - begin[1]) / 1e6
                << ',' << overlap / 1e6 << ',' << sms[0].size() << ',' << sms[1].size() << ',' << combined.size() << '\n';
            std::cout << name << " max_cross_grid_block_overlap_ms=" << overlap / 1e6
                      << " distinct_sms=" << combined.size() << '\n';
        }
        for (const auto stream : streams) check(cudaStreamDestroy(stream));
        check(cudaFree(device));
        csv.close();
        if (!csv) throw std::runtime_error("Failed to save probe report");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
