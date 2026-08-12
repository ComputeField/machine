#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

VERSION="${1:-0.1.0}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
ROOT="$BUILD_DIR/computefield-machine_${VERSION}_amd64"
SOURCE_ROOT="$ROOT/opt/computefield-machine/source"
install -d "$ROOT/DEBIAN" "$SOURCE_ROOT" "$ROOT/usr/bin" "$ROOT/etc/systemd/system"
runtime_sources=(
  LICENSE requirements.txt pyproject.toml
  capabilities.py cli.py config.py delta.py executor.py gpu_device.py main.py
  model_artifact.py monitor.py pairing.py sandbox_runtime.py script_guard.py
  speedtest.py workload_runner.py workspace.py
)
git -C "$SOURCE_DIR" archive HEAD -- "${runtime_sources[@]}" | tar -x -C "$SOURCE_ROOT"
chmod -R go-w "$SOURCE_ROOT"
install -m 0755 "$SOURCE_DIR/packaging/ubuntu/computefield-machine" "$ROOT/usr/bin/computefield-machine"
install -m 0644 "$SOURCE_DIR/packaging/systemd/computefield-machine.service" "$ROOT/etc/systemd/system/"
cat >"$ROOT/DEBIAN/control" <<EOF
Package: computefield-machine
Version: $VERSION
Architecture: amd64
Maintainer: Compute Field Lab, LLC <support@computefield.com>
Depends: python3 (>= 3.10), python3-venv, ca-certificates, bubblewrap, util-linux
Section: science
Priority: optional
Description: Standalone ComputeField GPU compute service
EOF
cat >"$ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "Install a working NVIDIA driver before ComputeField Machine." >&2
  exit 1
}
bwrap --help 2>&1 | grep -q -- '--disable-userns' || {
  echo "Bubblewrap with --disable-userns is required (Ubuntu 24.04 or newer)." >&2
  exit 1
}
id computefield-machine >/dev/null 2>&1 || useradd --system --home /var/lib/computefield-machine --shell /usr/sbin/nologin computefield-machine
install -d -o computefield-machine -g computefield-machine -m 0700 /var/lib/computefield-machine
current=/opt/computefield-machine/venv
candidate="/opt/computefield-machine/venv.candidate.$$"
link="/opt/computefield-machine/.venv-link.$$"
activated=0
legacy=
rollback() {
  rm -f "$link"
  if [ "$activated" -eq 0 ]; then
    rm -rf "$candidate"
    if [ -n "$legacy" ] && [ ! -e "$current" ]; then
      mv "$legacy" "$current"
    fi
  fi
}
trap rollback EXIT HUP INT TERM
rm -rf "$candidate"
python3 -m venv --copies "$candidate"
"$candidate/bin/pip" install --disable-pip-version-check --upgrade pip
"$candidate/bin/pip" install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
"$candidate/bin/pip" install nvidia-ml-py==12.560.30
"$candidate/bin/pip" install /opt/computefield-machine/source
runuser -u computefield-machine -- env \
  MACHINE_IDENTITY_FILE=/var/lib/computefield-machine/identity.json \
  MACHINE_WORK_DIR=/var/lib/computefield-machine/work \
  MACHINE_ISOLATION_MODE=sandbox \
  "$candidate/bin/computefield-machine" doctor || {
    echo "ComputeField Machine requires working unprivileged user namespaces; host policy was not modified." >&2
    exit 1
  }

previous=
if [ -L "$current" ]; then
  previous="$(readlink -f "$current" || true)"
elif [ -d "$current" ]; then
  previous="/opt/computefield-machine/venv.legacy.$$"
  mv "$current" "$previous"
  legacy="$previous"
fi
ln -s "$(basename "$candidate")" "$link"
mv -Tf "$link" "$current"
activated=1
trap - EXIT HUP INT TERM
systemctl daemon-reload || true
systemctl enable computefield-machine.service || true
systemctl try-restart computefield-machine.service || true
case "$previous" in
  /opt/computefield-machine/venv.candidate.*|/opt/computefield-machine/venv.legacy.*)
    [ "$previous" = "$candidate" ] || rm -rf "$previous"
    ;;
esac
EOF
chmod 0755 "$ROOT/DEBIAN/postinst"
versioned="$SOURCE_DIR/computefield-machine_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "$ROOT" "$versioned"
cp "$versioned" "$SOURCE_DIR/computefield-machine_amd64.deb"
