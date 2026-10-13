#!/usr/bin/env bash
# Uses the exact cached Rust dependencies of the installed TFHE-rs library.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_dir"
export CARGO_HOME="$project_dir/third_party/rust/cargo"
export RUSTUP_HOME="$project_dir/third_party/rust/rustup"
toolchain="$(tr -d '\r\n' < third_party/tfhe-rs/nightly-toolchain.txt)"
deps="$project_dir/build/tfhe-rs/release/deps"
bincode=("$deps"/libbincode-*.rlib)
[[ ${#bincode[@]} == 1 ]]
mkdir -p build/privateimpl logs
"$CARGO_HOME/bin/rustc" "+$toolchain" --edition=2021 -C opt-level=3 \
    experiment/tfhe_rs/gather_capacity.rs \
    --extern "tfhe=$project_dir/build/tfhe-rs/release/libtfhe.rlib" \
    --extern "bincode=${bincode[0]}" -L "dependency=$deps" \
    -L native=/usr/local/cuda/lib64 -C link-arg=-Wl,-rpath,/usr/local/cuda/lib64 \
    -o build/privateimpl/tfhe_gather_capacity
exec ./build/privateimpl/tfhe_gather_capacity "$@"
