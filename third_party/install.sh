#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pinned_commit() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["commit"])' \
    "$root_dir/versions.json" "$1"
}

clone_source() {
  local name="$1" url="$2" ref="$3"
  local dest="$root_dir/$name"
  if [[ -d "$dest/.git" ]]; then
    echo "$name already exists: $(git -C "$dest" rev-parse HEAD)"
    return
  fi
  if [[ -e "$dest" ]]; then
    echo "Refusing to replace existing $dest" >&2
    return 1
  fi
  git clone --filter=blob:none "$url" "$dest"
  if [[ -n "$ref" ]]; then
    git -C "$dest" checkout "$ref"
  fi
  echo "$name: $(git -C "$dest" rev-parse HEAD)"
}

# These are source checkouts. Building them requires their documented toolchains.
# Override refs after validating interoperability; record the resulting SHAs.
clone_source heongpu https://github.com/Alisah-Ozcan/HEonGPU.git "${HEONGPU_REF:-$(pinned_commit heongpu)}"
clone_source tfhe-rs https://github.com/zama-ai/tfhe-rs.git "${TFHE_RS_REF:-$(pinned_commit tfhe-rs)}"
clone_source tfhe-sanitization https://github.com/SNUCP/tfhe-sanitization.git "${TFHE_SANITIZATION_REF:-$(pinned_commit tfhe-sanitization)}"
git -C "$root_dir/heongpu" submodule update --init --recursive
python3 "$root_dir/apply_patches.py"
if [[ ! -f "$root_dir/tfhe-rs/Cargo.lock" ]]; then
  cp "$root_dir/locks/tfhe-rs.Cargo.lock" "$root_dir/tfhe-rs/Cargo.lock"
fi
