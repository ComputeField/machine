#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

[[ "$(id -u)" -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }
[[ "$(uname -m)" == "x86_64" ]] || { echo "The published CUDA package currently requires x86_64." >&2; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || {
  echo "Install the NVIDIA driver and reboot before installing ComputeField Machine." >&2
  exit 1
}

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl
download_dir="$(mktemp -d)"
package="$download_dir/computefield-machine_amd64.deb"
checksum="$package.sha256"
release_base="${COMPUTEFIELD_RELEASE_BASE_URL:-https://github.com/ComputeField/machine/releases/latest/download}"
trap 'rm -rf "$download_dir"' EXIT
curl --fail --location --proto '=https' --tlsv1.2 \
  "$release_base/computefield-machine_amd64.deb" \
  --output "$package"
(curl --fail --location --proto '=https' --tlsv1.2 \
  "$release_base/computefield-machine_amd64.deb.sha256" \
  --output "$checksum")
(cd "$download_dir" && sha256sum --check computefield-machine_amd64.deb.sha256)
apt-get install -y "$package"

echo "Installed. Open Machines in Compute Field and copy the generated pairing command."
