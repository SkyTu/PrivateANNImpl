#pragma once

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <vector>
#include "he/detail/record_layout.hpp"

#if PRIVATEIMPL_HAS_GPU
#include <cuda_runtime_api.h>
#endif

namespace privateann::checks {

// Independent plaintext oracle: compare only valid distances. Any member of
// the cutoff tie is accepted, but every returned (distance, ID) must come from
// the input with its original multiplicity. This detects detached/duplicate IDs.
inline void require_distance_topk(const std::vector<he::PlainRecord>& actual,
                                 const std::vector<he::PlainRecord>& input, std::size_t k) {
    if (!k || k > input.size() || actual.size() != k)
        throw std::runtime_error("Wrong distance Top-k output count");
    const auto order = [](const he::PlainRecord& value) {
        return std::make_pair(!value.valid, value.valid ? value.score : 0u);
    };
    auto sorted = input, remaining = input;
    std::sort(sorted.begin(), sorted.end(), [&](const auto& a, const auto& b) { return order(a) < order(b); });
    for (std::size_t i = 0; i < k; ++i) {
        if (order(actual[i]) != order(sorted[i]))
            throw std::runtime_error("Top-k distances differ from plaintext order");
        const auto match = std::find_if(remaining.begin(), remaining.end(), [&](const auto& value) {
            return value.valid == actual[i].valid && (!value.valid ||
                (value.score == actual[i].score && value.id == actual[i].id));
        });
        if (match == remaining.end())
            throw std::runtime_error("Top-k detached or duplicated a distance/ID pair");
        remaining.erase(match);
    }
}
inline void require_distance_topk(const std::vector<he::PlainRecord>& actual,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& input, std::size_t k) {
    std::vector<he::PlainRecord> records;
    for (const auto& [score, id] : input) records.push_back({static_cast<std::uint32_t>(score),
                                                         static_cast<std::uint32_t>(id), true});
    require_distance_topk(actual, records, k);
}

using Clock = std::chrono::steady_clock;
inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

enum class Status { passed, failed, unavailable };
enum class Phase { setup, encryption, evaluation, decryption };
struct Result {
    std::string name;
    Status status = Status::passed;
    std::string reason;
    double setup_ms = 0, encryption_ms = 0, evaluation_ms = 0, decryption_ms = 0;
    double wall_ms = 0;
    std::size_t evaluations = 0;
    // The large BFV workload is reported separately from small regression cases.
    std::size_t distance_records = 0, distance_dimensions = 0, distance_batches = 0;
    double distance_evaluation_ms = 0;
    // Conversion fixture: a BFV batch, selected distances, then Top-3 of 11.
    std::size_t conversion_records = 0;
    unsigned conversion_drop_bits = 0;
    double conversion_source_ms = 0, conversion_evaluation_ms = 0, conversion_topk_ms = 0;
};

struct Unavailable : std::runtime_error { using std::runtime_error::runtime_error; };

inline const char* status_name(Status status) {
    switch (status) {
    case Status::passed: return "PASS";
    case Status::failed: return "FAIL";
    case Status::unavailable: return "INCOMPLETE";
    }
    return "FAIL";
}
inline int exit_code(const Result& result) {
    return result.status == Status::passed ? 0 :
           result.status == Status::unavailable ? 77 : 1;
}

#if PRIVATEIMPL_HAS_GPU
inline void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#endif

inline void require_gpu() {
#if PRIVATEIMPL_HAS_GPU
    int count = 0;
    const auto error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess || count == 0)
        throw Unavailable(std::string("No accessible CUDA device: ") + cudaGetErrorString(error));
    for (int device = 0; device < count; ++device) {
        int pools = 0;
        cuda_check(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device),
                   "cudaDevAttrMemoryPoolsSupported");
        std::cout << "CUDA device " << device << ": memory pools supported="
                  << (pools ? "yes" : "no") << '\n';
    }
#else
    throw Unavailable("Built with PRIVATEIMPL_BUILD_GPU=OFF");
#endif
}

inline void synchronize_gpu() {
#if PRIVATEIMPL_HAS_GPU
    // TFHE-rs may use multiple devices. Wait for all visible devices, then
    // restore the caller's device. This measures completed work, not launches.
    int original = 0, count = 0;
    cuda_check(cudaGetDevice(&original), "cudaGetDevice");
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    try {
        for (int device = 0; device < count; ++device) {
            cuda_check(cudaSetDevice(device), "cudaSetDevice");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
    } catch (...) {
        cudaSetDevice(original);
        throw;
    }
    cuda_check(cudaSetDevice(original), "cudaSetDevice restore");
#endif
}

template <class Function>
auto measure(Result& result, Phase phase, const char* label, Function&& function,
             bool print_timing = true)
    -> std::invoke_result_t<Function> {
    synchronize_gpu();
    const auto start = Clock::now();
    auto finish = [&] {
        synchronize_gpu();
        const double ms = elapsed_ms(start);
        switch (phase) {
        case Phase::setup: result.setup_ms += ms; break;
        case Phase::encryption: result.encryption_ms += ms; break;
        case Phase::evaluation: result.evaluation_ms += ms; ++result.evaluations; break;
        case Phase::decryption: result.decryption_ms += ms; break;
        }
        if (print_timing)
            std::cout << "TIME " << label << " = " << std::fixed << std::setprecision(3)
                      << ms << " ms" << std::endl;
    };
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
        std::forward<Function>(function)();
        finish();
    } else {
        auto value = std::forward<Function>(function)();
        finish();
        return value;
    }
}

template <class Function>
Result run(const char* name, Function&& function) {
    Result result;
    result.name = name;
    std::cout << "Running " << name << std::endl;
    const auto start = Clock::now();
    try {
        function(result);
    } catch (const Unavailable& error) {
        result.status = Status::unavailable;
        result.reason = error.what();
    } catch (const std::exception& error) {
        result.status = Status::failed;
        result.reason = error.what();
    }
    result.wall_ms = elapsed_ms(start);
    std::cout << status_name(result.status) << ": " << name;
    if (!result.reason.empty()) std::cout << " - " << result.reason;
    std::cout << '\n';
    if (result.evaluations)
        std::cout << "Completed server evaluations: " << result.evaluations
                  << "; evaluation_ms=" << result.evaluation_ms
                  << "; setup_ms=" << result.setup_ms
                  << "; encryption_ms=" << result.encryption_ms
                  << "; decryption_ms=" << result.decryption_ms << '\n';
    if (result.distance_batches) {
        std::cout << "BFV distance workload: records_verified=" << result.distance_records
                  << "; dimensions=" << result.distance_dimensions
                  << "; batches=" << result.distance_batches
                  << "; distance_evaluation_ms=" << result.distance_evaluation_ms << '\n';
        if (result.distance_evaluation_ms > 0)
            std::cout << "BFV distance evaluation throughput: "
                      << result.distance_records * 1000.0 / result.distance_evaluation_ms
                      << " records/s\n";
    }
    if (result.conversion_records) {
        std::cout << "BFV-to-TFHE workload: records=" << result.conversion_records
                  << "; drop_bits=" << result.conversion_drop_bits
                  << "; source_distance_ms=" << result.conversion_source_ms
                  << "; conversion_ms=" << result.conversion_evaluation_ms
                  << "; converted_top3_of_11_ms=" << result.conversion_topk_ms
                  << "; fixture_chain_ms=" << result.conversion_source_ms +
                     result.conversion_evaluation_ms + result.conversion_topk_ms << '\n';
    }
    std::cout << "Component wall time: " << result.wall_ms << " ms" << std::endl;
    return result;
}

Result bfv_distance();
Result bfc_tfhe_conversion(std::size_t count = 500, unsigned drop_bits = 7);
Result tfhe_topk();

} // namespace privateann::checks
