#!/usr/bin/env bash
# Build the project adapter against the exact, already-built TFHE-rs rlib.
# No upstream source edits or extra downloads are needed.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_HOME="$project_dir/third_party/rust/cargo"
export RUSTUP_HOME="$project_dir/third_party/rust/rustup"
toolchain="$(tr -d '\r\n' < "$project_dir/third_party/tfhe-rs/nightly-toolchain.txt")"
deps="$project_dir/build/tfhe-rs/release/deps"
rlib="$project_dir/build/tfhe-rs/release/libtfhe.rlib"
if [[ ! -s "$rlib" ]]; then
    echo "Run third_party/install_tfhe_rs.sh first." >&2
    exit 1
fi
bincode=("$deps"/libbincode-*.rlib)
rayon=("$deps"/librayon-*.rlib)
if [[ ${#bincode[@]} != 1 || ${#rayon[@]} != 1 ]]; then
    echo "Expected exactly one cached bincode and rayon rlib; inspect the TFHE-rs build." >&2
    exit 1
fi
mkdir -p "$project_dir/third_party/tfhe-rs-install/lib"
"$CARGO_HOME/bin/rustc" "+$toolchain" --edition=2021 --crate-type=cdylib -C opt-level=3 \
    "$project_dir/src/utils/conversion/rust_bridge/bfv_tfhe.rs" \
    --extern "tfhe=$rlib" --extern "bincode=${bincode[0]}" --extern "rayon=${rayon[0]}" \
    -L "dependency=$deps" -L native=/usr/local/cuda/lib64 \
    -C link-arg=-Wl,-rpath,/usr/local/cuda/lib64 \
    -o "$project_dir/third_party/tfhe-rs-install/lib/libbfv_tfhe_bridge.so.new"
mv -f "$project_dir/third_party/tfhe-rs-install/lib/libbfv_tfhe_bridge.so.new" \
    "$project_dir/third_party/tfhe-rs-install/lib/libbfv_tfhe_bridge.so"
