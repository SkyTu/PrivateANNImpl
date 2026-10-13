#!/usr/bin/env bash
# Measure bridge keys using the same cached TFHE-rs dependency build.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export CARGO_HOME="$project_dir/third_party/rust/cargo"
export RUSTUP_HOME="$project_dir/third_party/rust/rustup"
toolchain="$(tr -d '\r\n' < third_party/tfhe-rs/nightly-toolchain.txt)"
deps="$project_dir/build/tfhe-rs/release/deps"
bincode=("$deps"/libbincode-*.rlib)
rayon=("$deps"/librayon-*.rlib)
[[ ${#bincode[@]} == 1 && ${#rayon[@]} == 1 ]]
mkdir -p build/privateimpl logs
"$CARGO_HOME/bin/rustc" "+$toolchain" --edition=2021 -C opt-level=3 \
    experiment/bfv_conversion/key_size.rs \
    --extern "tfhe=$project_dir/build/tfhe-rs/release/libtfhe.rlib" \
    --extern "bincode=${bincode[0]}" --extern "rayon=${rayon[0]}" \
    -L "dependency=$deps" -L native=/usr/local/cuda/lib64 \
    -C link-arg=-Wl,-rpath,/usr/local/cuda/lib64 \
    -o build/privateimpl/bfv_tfhe_key_size
exec ./build/privateimpl/bfv_tfhe_key_size "$@"
