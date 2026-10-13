#!/usr/bin/env bash
# Build the pinned upstream GPU C API, including its generated header.
# Rust tooling and Cargo's cache stay inside third_party; no sudo is needed.
# Set TFHE_CUDA_ARCH=89 to build for L40S without a visible build GPU.
set -euo pipefail
third_party_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$third_party_dir")"
source_dir="$third_party_dir/tfhe-rs"
prefix="$third_party_dir/tfhe-rs-install"
expected_commit=638f4c8bab899659374b98df766ea7eb8acfc55f
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$expected_commit" ]]; then
    echo "Expected TFHE-rs 1.5.4 commit $expected_commit; refusing an unverified C ABI." >&2
    exit 1
fi
python3 "$third_party_dir/apply_patches.py" --only tfhe-rs
if [[ ! -f "$source_dir/Cargo.lock" && -f "$third_party_dir/locks/tfhe-rs.Cargo.lock" ]]; then
    cp "$third_party_dir/locks/tfhe-rs.Cargo.lock" "$source_dir/Cargo.lock"
fi

cuda_root="${CUDA_ROOT:-/usr/local/cuda}"
export PATH="$cuda_root/bin:$PATH"
if [[ -x "$project_dir/../SecPromptTuning-service/.venv/bin/cmake" ]]; then
    export PATH="$project_dir/../SecPromptTuning-service/.venv/bin:$PATH"
fi
for program in curl c++ cmake nvcc; do
    if ! command -v "$program" >/dev/null; then
        echo "Missing build tool: $program" >&2
        exit 1
    fi
done
if [[ -z "${LIBCLANG_PATH:-}" && -f /usr/lib/llvm-14/lib/libclang.so ]]; then
    export LIBCLANG_PATH=/usr/lib/llvm-14/lib
fi

export CARGO_HOME="$third_party_dir/rust/cargo"
export RUSTUP_HOME="$third_party_dir/rust/rustup"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME" "$project_dir/logs"
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
    installer="$third_party_dir/rust/rustup-init.sh"
    curl --proto '=https' --tlsv1.2 --fail --location --retry 2 \
        --connect-timeout 10 --max-time 120 https://sh.rustup.rs -o "$installer"
    sh "$installer" -y --no-modify-path --profile minimal --default-toolchain none
fi
# C API macro expansion uses the upstream pinned nightly, not stable Rust.
toolchain="$(tr -d '\r\n' < "$source_dir/nightly-toolchain.txt")"
if ! "$CARGO_HOME/bin/rustup" run "$toolchain" rustc --version >/dev/null 2>&1; then
    "$CARGO_HOME/bin/rustup" toolchain install "$toolchain" --profile minimal
fi
export RUSTUP_TOOLCHAIN="$toolchain"
export CARGO_TARGET_DIR="$project_dir/build/tfhe-rs"
export CARGO_BUILD_JOBS="${TFHE_BUILD_JOBS:-4}"
export CUDACXX="$cuda_root/bin/nvcc"
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,$cuda_root/lib64"

cd -- "$source_dir"
{
    # The upstream repository ignores Cargo.lock. Resolve dependencies on the
    # first build, then keep that lock file for all subsequent builds.
    if [[ ! -f Cargo.lock ]]; then
        echo "Generating Cargo.lock for the first TFHE-rs build..."
        "$CARGO_HOME/bin/cargo" generate-lockfile
    fi
    "$CARGO_HOME/bin/cargo" build --locked --release -p tfhe \
        --features high-level-c-api,gpu
} 2>&1 | tee "$project_dir/logs/tfhe_rs_build.log"

# build.rs emits tfhe.h at CARGO_TARGET_DIR when that variable is explicitly set.
header="$CARGO_TARGET_DIR/tfhe.h"
library="$CARGO_TARGET_DIR/release/libtfhe.so"
test -s "$header"
test -s "$library"
c++ -std=c++17 -DWITH_FEATURE_GPU -DPRIVATEIMPL_VERIFY_TFHE_RS_ABI \
    -DPRIVATEIMPL_ABI_ONLY -I "$project_dir/src" -I "$CARGO_TARGET_DIR" \
    -fsyntax-only "$project_dir/test/check.cpp"

# Publish the local install only after the generated ABI header passed checks.
mkdir -p "$prefix/lib" "$prefix/include"
install -m 644 "$header" "$prefix/include/tfhe.h"
install -m 755 "$library" "$prefix/lib/libtfhe.so"
cat > "$prefix/BUILD_INFO.txt" <<EOF
TFHE-rs 1.5.4
commit=$expected_commit
toolchain=$toolchain
features=high-level-c-api,gpu
abi_check=passed
cuda_arch=${TFHE_CUDA_ARCH:-auto}
cuda_device_source_sha256=$(sha256sum "$source_dir/backends/tfhe-cuda-backend/cuda/src/device.cu" | cut -d ' ' -f 1)
cuda_fallback_arena_sha256=$(sha256sum "$source_dir/backends/tfhe-cuda-backend/cuda/src/utils/fallback_cache.cuh" | cut -d ' ' -f 1)
EOF
echo "Installed TFHE-rs GPU C API at $prefix"
echo "Next: ./build/privateimpl/check from the PrivateImpl directory."
