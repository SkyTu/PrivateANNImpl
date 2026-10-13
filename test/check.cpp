// Compile this section alone when installing the generated TFHE-rs C API.
#include "utils/detail/tfhe_rs_api.hpp"
#ifdef PRIVATEIMPL_VERIFY_TFHE_RS_ABI
#include <cstddef>
#include <type_traits>
using privateann::he::tfhe_rs::Api;
#define PRIVATEIMPL_CHECK_API(result, name, args) \
    static_assert(std::is_same_v<decltype(&::name), decltype(Api::name)>, #name " ABI mismatch");
PRIVATEIMPL_TFHE_RS_API(PRIVATEIMPL_CHECK_API)
#undef PRIVATEIMPL_CHECK_API
#endif

#ifndef PRIVATEIMPL_ABI_ONLY
#include "utils/check_support.hpp"
#include <filesystem>
#include <fstream>
#include <vector>
#include "he/detail/record_layout.hpp"
#include "he/detail/selection_network.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

namespace {
using privateann::he::PlainRecord;
using privateann::he::RecordLayout;
const RecordLayout wide_layout;
// A scalar CPU model executes the public topology, carrying IDs without ever
// comparing them. The separate oracle checks only distance order and membership.
struct Operations {
    std::size_t calls = 0;
    PlainRecord padding(PlainRecord) const { return {0, 0, false}; }
    template <class Function>
    void execute_layers(const privateann::he::TopkSchedule& schedule, Function operation) {
        for (const auto& layer : schedule.layers)
            for (const auto& gate : layer) operation(gate);
    }
    bool greater(const PlainRecord& a, const PlainRecord& b) {
        return std::make_pair(!a.valid, a.valid ? a.score : 0u) >
               std::make_pair(!b.valid, b.valid ? b.score : 0u);
    }
    PlainRecord minimum(PlainRecord a, PlainRecord b) { ++calls; return greater(a, b) ? b : a; }
    std::pair<PlainRecord, PlainRecord> minmax(PlainRecord a, PlainRecord b) {
        ++calls;
        return greater(a, b) ? std::make_pair(b, a) : std::make_pair(a, b);
    }
};
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <class Function> void rejects(Function function) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Invalid argument was accepted");
}
std::size_t verify(const std::vector<PlainRecord>& input, std::size_t k) {
    Operations operations;
    const auto result = privateann::he::topk_network(operations, input, k);
    privateann::checks::require_distance_topk(result, input, k);
    return operations.calls;
}
} // namespace


privateann::checks::Result network_and_parameters() {
    return privateann::checks::run("network_and_parameters", [](privateann::checks::Result&) {
        for (unsigned n = 1; n <= 7; ++n) for (unsigned k = 1; k <= n; ++k)
            for (unsigned mask = 0; mask < (1U << n); ++mask) {
                std::vector<PlainRecord> input;
                for (unsigned i = 0; i < n; ++i) input.push_back({(mask >> i) & 1, n - i, true});
                verify(input, k);
            }
        std::mt19937 random(42);
        for (unsigned n : {1, 3, 17, 31, 65, 100, 499, 500})
            for (unsigned k : {1u, 3u, 10u, 16u, n}) if (k <= n) {
                std::vector<PlainRecord> input;
                for (unsigned i = 0; i < n; ++i)
                    input.push_back({static_cast<std::uint32_t>(random() % 19), i, true});
                const auto calls = verify(input, k);
                std::reverse(input.begin(), input.end());
                require(verify(input, k) == calls, "Network depends on secret values");
            }
        // Check all k, 32-bit boundaries, distance ordering and carried IDs, and invalid padding.
        const std::vector<PlainRecord> boundaries{
            {0, UINT32_MAX, true}, {UINT32_MAX, UINT32_MAX, true},
            {UINT32_MAX, 0, true}, {1, 0, true}, {0, 0, true}};
        for (std::size_t k = 1; k <= boundaries.size(); ++k) verify(boundaries, k);
        require(wide_layout.padding() > UINT32_MAX, "Padding collides with uint32 distance");
        require(!wide_layout.decode(wide_layout.padding(), 0).valid, "Padding must be invalid");
        Operations operations;
        rejects([&] { privateann::he::topk_network(operations, std::vector<PlainRecord>{}, 1); });
        rejects([&] { privateann::he::topk_network(operations, std::vector<PlainRecord>{{0,0,true}}, 0); });
        rejects([&] { privateann::he::topk_network(operations, std::vector<PlainRecord>{{0,0,true}}, 2); });
        rejects([] { wide_layout.validate(UINT64_MAX, 0); });
        rejects([] { wide_layout.validate(0, UINT64_MAX); });
        rejects([] { RecordLayout(4, 4).validate(16, 0); });
        for (const RecordLayout layout : {RecordLayout(23, 7), RecordLayout(23, 20), RecordLayout(32, 32)}) {
            const auto maximum = layout.padding() - 1;
            const auto decoded = layout.decode(maximum, layout.max_id());
            require(decoded.valid && decoded.score == maximum && decoded.id == layout.max_id(),
                    "Separate fields lost distance/ID bits");
            require(!layout.decode(layout.padding(), 0).valid, "Padding is valid");
            rejects([&] { layout.decode(layout.padding() + 1, 0); });
        }
        require(RecordLayout(23, 20).distance_ciphertext_bits() == 32,
                "IDs must not widen distance comparison");
        require(RecordLayout(31, 32).padding() == (std::uint64_t{1} << 31), "Bit-31 padding is wrong");
        require(RecordLayout(32, 32).distance_ciphertext_bits() == 64, "Full uint32 sentinel lost");
        rejects([] { RecordLayout(23, 7).validate(1u << 23, 0); });
        rejects([] { RecordLayout(23, 7).validate(0, 128); });
        rejects([] { RecordLayout(0, 7); });
        rejects([] { RecordLayout(23, 33); });
        // Equal-distance cutoff accepts arbitrary original IDs, but rejects
        // duplicate IDs or a score detached from its original record.
        privateann::checks::require_distance_topk({{4,9,true}},
            std::vector<PlainRecord>{{4,1,true},{4,9,true}}, 1);
        const auto must_fail = [](auto&& fn) {
            try { fn(); } catch (const std::runtime_error&) { return; }
            throw std::runtime_error("Invalid Top-k output passed the oracle");
        };
        must_fail([] { privateann::checks::require_distance_topk({{4,9,true},{4,9,true}},
            std::vector<PlainRecord>{{4,1,true},{4,9,true}}, 2); });
        must_fail([] { privateann::checks::require_distance_topk({{4,9,true}},
            std::vector<PlainRecord>{{4,1,true},{5,9,true}}, 1); });
        privateann::BfvParameters parameters;
        parameters.validate_distance_dimension(1);
        rejects([&] { parameters.validate_distance_dimension(0); });
        parameters.validate_distance_dimension(128);
        // 129 worst-case coordinates exceed this 23-bit plaintext modulus.
        rejects([&] { parameters.validate_distance_dimension(129); });
        require(parameters.slots == 4096 && parameters.plaintext_modulus == 8380417,
                "Default BFV parameters must support 23-bit 128D distances");
        parameters.slots = 8192;
        rejects([&] { parameters.validate(); });
        parameters.slots = 4096;
        parameters.plaintext_modulus = 16385;
        rejects([&] { parameters.validate(); });
        // Each layer must be safe to evaluate from its inputs concurrently.
        const auto schedule = privateann::he::topk_schedule(100, 10);
        require(schedule.size == 112, "Grouped Top-k uses incorrect padding");
        std::size_t full = 0, min_only = 0;
        for (const auto& layer : schedule.layers) {
            std::vector<bool> used(schedule.size, false);
            for (const auto& gate : layer) {
                require(gate.a < schedule.size && gate.b < schedule.size && gate.a != gate.b,
                        "Gate index out of range");
                require(!used[gate.a] && !used[gate.b], "Dependent gates share a layer");
                used[gate.a] = used[gate.b] = true;
                if (gate.min_only) ++min_only;
                else ++full;
            }
        }
        std::cout << "100/10 grouped schedule: " << full << " compare/select, "
                  << min_only << " min-only gates\n";
        std::cout << "PASS: grouped network scheduling, 100/10, 500/16, distance ordering, "
                     "ties, padding, uint32 boundaries, and parameter bounds\n";

    });
}

namespace {
void write_report(const std::filesystem::path& path,
                  const std::vector<privateann::checks::Result>& results,
                  double wall_ms) {
    using namespace privateann::checks;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write timing report: " + path.string());
    out << std::fixed << std::setprecision(6);
    out << "component,status,evaluation_calls,evaluation_ms,setup_ms,encryption_ms,decryption_ms,wall_ms,"
           "distance_records,distance_dimensions,distance_batches,distance_evaluation_ms,distance_records_per_second,"
           "conversion_records,conversion_drop_bits,conversion_source_ms,conversion_evaluation_ms,converted_top3_of_11_ms\n";
    double evaluation_ms = 0, setup_ms = 0, encryption_ms = 0, decryption_ms = 0;
    std::size_t calls = 0;
    for (const auto& result : results) {
        out << result.name << ',' << status_name(result.status) << ',' << result.evaluations << ',';
        if (result.evaluations) out << result.evaluation_ms;
        out << ',' << result.setup_ms << ',' << result.encryption_ms << ','
            << result.decryption_ms << ',' << result.wall_ms;
        if (result.distance_batches) {
            out << ',' << result.distance_records << ',' << result.distance_dimensions
                << ',' << result.distance_batches << ',' << result.distance_evaluation_ms << ',';
            if (result.distance_evaluation_ms > 0)
                out << result.distance_records * 1000.0 / result.distance_evaluation_ms;
        } else out << ",,,,,";
        if (result.conversion_records)
            out << ',' << result.conversion_records << ',' << result.conversion_drop_bits
                << ',' << result.conversion_source_ms << ',' << result.conversion_evaluation_ms
                << ',' << result.conversion_topk_ms;
        else out << ",,,,,";
        out << '\n';
        evaluation_ms += result.evaluation_ms;
        setup_ms += result.setup_ms;
        encryption_ms += result.encryption_ms;
        decryption_ms += result.decryption_ms;
        calls += result.evaluations;
    }
    // This sum covers independent fixtures, not one encrypted ANN query.
    out << "TOTAL_COMPONENTS,PARTIAL_PIPELINE," << calls << ',';
    if (calls) out << evaluation_ms;
    out << ',' << setup_ms << ',' << encryption_ms << ',' << decryption_ms << ',' << wall_ms << ",,,,,,,,,,\n";
    out << "FULL_ANN_PIPELINE,UNAVAILABLE,,,,,,,,,,,,,,,,\n";
    out.close();
    if (!out) throw std::runtime_error("Could not finish timing report: " + path.string());
    std::cout << "\nTotal server evaluation time (completed component calls): ";
    if (calls) std::cout << evaluation_ms << " ms across " << calls << " calls\n";
    else std::cout << "N/A (no GPU evaluation completed)\n";
    std::cout << "Total setup/key generation: " << setup_ms << " ms\n"
              << "Total encryption: " << encryption_ms << " ms\n"
              << "Total decryption: " << decryption_ms << " ms\n"
              << "Total check wall time: " << wall_ms << " ms\n"
              << "Full ANN evaluation time: N/A (output sanitization and dataset integration missing)\n"
              << "CSV report: " << std::filesystem::absolute(path) << std::endl;
}
} // namespace

int main(int argc, char** argv) try {
    using namespace privateann::checks;
    bool cpu_only = false;
    std::filesystem::path report = "logs/evaluation_times.csv";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cpu") cpu_only = true;
        else if (arg == "--report" && i + 1 < argc) report = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: check [--cpu] [--report PATH]\n"
                         "Default: ABI, library, CPU logic, BFV, conversion status, TFHE Top-k.\n"
                         "Exit codes: 0=all requested checks pass, 1=failure, 2=incomplete.\n";
            return 0;
        } else throw std::invalid_argument("Unknown or incomplete argument: " + arg);
    }
    const auto start = Clock::now();
    std::vector<Result> results;
    results.push_back(network_and_parameters());
    if (!cpu_only) {
        results.push_back(run("tfhe_c_api_abi", [](Result&) {
#ifndef PRIVATEIMPL_VERIFY_TFHE_RS_ABI
            throw Unavailable("Generated tfhe.h was not available when compiling check");
#endif
        }));
        results.push_back(run("tfhe_gpu_library", [](Result&) {
            privateann::he::tfhe_rs::Api::instance();
        }));
        results.push_back(bfv_distance());
        results.push_back(bfc_tfhe_conversion());
        results.push_back(tfhe_topk());
    }
    write_report(report, results, elapsed_ms(start));
    bool failed = false, incomplete = false;
    for (const auto& result : results) {
        failed |= result.status == Status::failed;
        incomplete |= result.status == Status::unavailable;
    }
    if (failed) {
        std::cerr << "FAIL: one or more components failed; inspect the component output.\n";
        return 1;
    }
    if (incomplete) {
        std::cerr << "INCOMPLETE: not all components are available; this is not an end-to-end pass.\n";
        return 2;
    }
    std::cout << (cpu_only ? "PASS: CPU logic checks (GPU components were not requested).\n"
                          : "PASS: all requested component checks.\n");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
#endif // PRIVATEIMPL_ABI_ONLY
