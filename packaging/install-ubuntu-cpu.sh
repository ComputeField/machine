#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

[[ "$(id -u)" -eq 0 ]] || { echo "Run with sudo" >&2; exit 1; }
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv ca-certificates apparmor bubblewrap util-linux
python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || {
  echo "Python 3.10+ is required." >&2
  exit 1
}
bwrap --help 2>&1 | grep -q -- '--disable-userns' || {
  echo "Bubblewrap with --disable-userns is required (Ubuntu 24.04 or newer)." >&2
  exit 1
}
id computefield-machine-cpu >/dev/null 2>&1 || useradd --system --home /var/lib/computefield-machine-cpu --shell /usr/sbin/nologin computefield-machine-cpu
install -d -o computefield-machine-cpu -g computefield-machine-cpu -m 0700 /var/lib/computefield-machine-cpu
install -d -m 0755 /opt/computefield-machine-cpu
"$SOURCE_DIR/packaging/install-linux-sandbox.sh" \
  /opt/computefield-machine-cpu computefield-machine-cpu-bwrap
python3 -m venv /opt/computefield-machine-cpu/venv
/opt/computefield-machine-cpu/venv/bin/pip install --disable-pip-version-check --upgrade pip
/opt/computefield-machine-cpu/venv/bin/pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
/opt/computefield-machine-cpu/venv/bin/pip install "$SOURCE_DIR"
install -m 0755 "$SOURCE_DIR/packaging/ubuntu/computefield-machine" /usr/bin/computefield-machine-cpu
install -m 0644 "$SOURCE_DIR/packaging/systemd/computefield-machine-cpu.service" /etc/systemd/system/computefield-machine-cpu.service
systemctl daemon-reload
systemctl enable computefield-machine-cpu.service
runuser -u computefield-machine-cpu -- env \
  MACHINE_IDENTITY_FILE=/var/lib/computefield-machine-cpu/identity.json \
  MACHINE_WORK_DIR=/var/lib/computefield-machine-cpu/work \
  MACHINE_ISOLATION_MODE=sandbox \
  MACHINE_COMPUTE_MODE=cpu \
  CUDA_VISIBLE_DEVICES= \
  COMPUTEFIELD_BWRAP=/opt/computefield-machine-cpu/bin/bwrap \
  /opt/computefield-machine-cpu/venv/bin/computefield-machine doctor || {
    echo "ComputeField Machine requires working unprivileged user namespaces; host policy was not modified." >&2
    exit 1
  }
echo "Installed. Create a code on the Machines page, then run:"
echo "computefield-machine-cpu pair CODE"
