#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

[[ "$(uname -s)" == Darwin ]] || { echo "This installer is for macOS." >&2; exit 1; }
[[ "$(uname -m)" == arm64 ]] || { echo "The current PyTorch release requires Apple silicon." >&2; exit 1; }
release_base="${COMPUTEFIELD_RELEASE_BASE_URL:-https://github.com/ComputeField/machine/releases/latest/download}"
download_dir="$(mktemp -d)"
archive="$download_dir/computefield-machine_macos-source.tar.gz"
checksum="$archive.sha256"
trap 'rm -rf "$download_dir"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 \
  "$release_base/computefield-machine_macos-source.tar.gz" --output "$archive"
curl --fail --location --proto '=https' --tlsv1.2 \
  "$release_base/computefield-machine_macos-source.tar.gz.sha256" --output "$checksum"
(cd "$download_dir" && shasum -a 256 --check "$(basename "$checksum")")
tar -xzf "$archive" -C "$download_dir"
"$download_dir/computefield-machine/packaging/install-macos.sh"
