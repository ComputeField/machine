#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

[[ "$(id -u)" -eq 0 ]] || { echo "Run with sudo" >&2; exit 1; }
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || { echo "Install a working NVIDIA driver first; CUDA itself is bundled by PyTorch." >&2; exit 1; }
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
id computefield-machine >/dev/null 2>&1 || useradd --system --home /var/lib/computefield-machine --shell /usr/sbin/nologin computefield-machine
install -d -o computefield-machine -g computefield-machine -m 0700 /var/lib/computefield-machine
install -d -m 0755 /opt/computefield-machine
"$SOURCE_DIR/packaging/install-linux-sandbox.sh" \
  /opt/computefield-machine computefield-machine-bwrap
python3 -m venv /opt/computefield-machine/venv
/opt/computefield-machine/venv/bin/pip install --disable-pip-version-check --upgrade pip
/opt/computefield-machine/venv/bin/pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
/opt/computefield-machine/venv/bin/pip install nvidia-ml-py==12.560.30
/opt/computefield-machine/venv/bin/pip install "$SOURCE_DIR"
install -m 0755 "$SOURCE_DIR/packaging/ubuntu/computefield-machine" /usr/bin/computefield-machine
install -m 0644 "$SOURCE_DIR/packaging/systemd/computefield-machine.service" /etc/systemd/system/computefield-machine.service
systemctl daemon-reload
systemctl enable computefield-machine.service
runuser -u computefield-machine -- env \
  MACHINE_IDENTITY_FILE=/var/lib/computefield-machine/identity.json \
  MACHINE_WORK_DIR=/var/lib/computefield-machine/work \
  MACHINE_ISOLATION_MODE=sandbox \
  COMPUTEFIELD_BWRAP=/opt/computefield-machine/bin/bwrap \
  /opt/computefield-machine/venv/bin/computefield-machine doctor || {
    echo "ComputeField Machine requires working unprivileged user namespaces; host policy was not modified." >&2
    exit 1
  }
echo "Installed. Create a code on the Machines page, then pair this service account with it."
echo "computefield-machine pair CODE"
