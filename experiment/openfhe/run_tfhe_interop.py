#!/usr/bin/env python3
"""Run the real CKKS -> OpenFHE LWE -> TFHE-rs import experiment.

Exit 2 means the GPU integer bridge is blocked, not that the test passed.
Temporary diagnostic fixtures contain synthetic test keys and are deleted.
"""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    project = Path(__file__).resolve().parents[2]
    os.chdir(project)
    os.umask(0o077)
    build = project / "build/openfhe-experiment"
    logs = project / "logs"
    logs.mkdir(exist_ok=True)
    preferred_cmake = project.parent / "SecPromptTuning-service/.venv/bin/cmake"
    cmake = os.environ.get("CMAKE") or (
        str(preferred_cmake) if preferred_cmake.exists() else shutil.which("cmake")
    )
    if not cmake:
        raise RuntimeError("cmake is required")
    env = dict(os.environ)
    env.update({
        "CARGO_HOME": str(project / "third_party/rust/cargo"),
        "RUSTUP_HOME": str(project / "third_party/rust/rustup"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        "OMP_DYNAMIC": "FALSE",
    })
    rustc = project / "third_party/rust/cargo/bin/rustc"
    toolchain = (project / "third_party/tfhe-rs/nightly-toolchain.txt").read_text().strip()
    rlib = project / "build/tfhe-rs/release/libtfhe.rlib"
    if not rlib.exists():
        raise RuntimeError("Build third_party/install_tfhe_rs.sh first; its cached Rust library is required")
    cuda_lib = str(Path(os.environ.get("CUDA_ROOT", "/usr/local/cuda")) / "lib64")
    commands = [
        [cmake, "-S", "experiment/openfhe", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
         f"-DOpenFHE_DIR={project}/third_party/openfhe-install/lib/OpenFHE"],
        [cmake, "--build", str(build), "--target", "ckks_tfhe_interop", "-j8"],
        [str(rustc), f"+{toolchain}", "--edition=2021", "-C", "opt-level=2",
         "experiment/openfhe/tfhe_raw_probe.rs", "--extern", f"tfhe={rlib}",
         "-L", f"dependency={project}/build/tfhe-rs/release/deps",
         "-L", f"native={cuda_lib}", "-C", f"link-arg=-Wl,-rpath,{cuda_lib}",
         "-o", str(build / "tfhe_raw_probe")],
    ]
    with (logs / "openfhe_tfhe_interop_build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    start = time.perf_counter()
    # Only this private temporary directory contains the diagnostic secret key.
    with tempfile.TemporaryDirectory(prefix="ckks_tfhe_", dir=project / "build") as temporary:
        prefix = Path(temporary) / "probe"
        with (logs / "openfhe_ckks_tfhe_500.log").open("w") as log:
            conversion = subprocess.run(
                [str(build / "ckks_tfhe_interop"), str(prefix), str(logs / "openfhe_ckks_tfhe_500.csv")],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        print((logs / "openfhe_ckks_tfhe_500.log").read_text(), end="", flush=True)
        if conversion.returncode not in (0, 2):
            raise RuntimeError(f"OpenFHE experiment failed with {conversion.returncode}")

        probe_codes = {}
        with (logs / "tfhe_rs_openfhe_import_500.log").open("w") as log:
            for case in ("full23", "modular_control"):
                result = subprocess.run(
                    [str(build / "tfhe_raw_probe"), f"{prefix}_{case}.bin"],
                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                output = f"CASE: {case}\n{result.stdout}probe_exit_code={result.returncode}\n"
                log.write(output)
                print(output, end="", flush=True)
                probe_codes[case] = result.returncode
                if result.returncode != 2:
                    raise RuntimeError(f"Unexpected TFHE-rs probe result: {result.returncode}")

    with (logs / "openfhe_ckks_tfhe_500.csv").open() as csv_file:
        conversion_results = list(csv.DictReader(csv_file))
    summary = {
        "bridge_status": "BLOCKED",
        "elements_per_case": 500,
        "coordinate_bits": 8,
        "dimensions": 128,
        "omp_threads": env["OMP_NUM_THREADS"],
        "conversion": conversion_results,
        "raw_import_matches_openfhe_per_case": 500,
        "gpu_input_conformant_per_case": 0,
        "gpu_minmax_executed": False,
        "end_to_end_gpu_evaluation_ms": None,
        "probe_exit_codes": probe_codes,
        "experiment_wall_ms_excluding_build": (time.perf_counter() - start) * 1000,
        "notes": [
            "Each case switches the same 500 full distances; the control checks residues modulo p.",
            "Diagnostic decryption never feeds a re-encryption step.",
            "Raw LWE import is not a complete bridge to TFHE-rs GPU integers.",
            "The OpenFHE intermediate modulus remains at the upstream default of 27 bits.",
        ],
    }
    (logs / "openfhe_tfhe_interop_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved logs/openfhe_tfhe_interop_summary.json; GPU bridge BLOCKED (exit 2).")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
