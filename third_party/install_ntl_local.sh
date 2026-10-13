#!/usr/bin/env bash
# Install Ubuntu's NTL headers and runtime under third_party without sudo.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
download_dir="$root_dir/ntl-download"
install_dir="$root_dir/ntl"
mkdir -p "$download_dir" "$install_dir"
cd "$download_dir"
apt-get download libntl-dev libntl44 libgf2x3
shopt -s nullglob
packages=(libntl-dev_*.deb libntl44_*.deb libgf2x3_*.deb)
if [[ "${#packages[@]}" -ne 3 ]]; then
  echo "Expected libntl-dev, libntl44 and libgf2x3 .deb packages" >&2
  exit 1
fi
for package in "${packages[@]}"; do
  dpkg-deb -x "$package" "$install_dir"
done
test -f "$install_dir/usr/include/NTL/RR.h"
echo "NTL installed under $install_dir"
