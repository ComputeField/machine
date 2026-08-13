#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

VERSION="${1:?usage: packaging/build-deb.sh VERSION [gpu|cpu]}"
PROFILE="${2:-gpu}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "$SOURCE_DIR/pyproject.toml" | head -n 1)"
[[ "$VERSION" == "$project_version" ]] || {
  echo "Package version $VERSION does not match pyproject.toml ($project_version)." >&2
  exit 2
}

case "$PROFILE" in
  gpu)
    package_name="computefield-machine"
    command_name="computefield-machine"
    service_user="computefield-machine"
    service_name="computefield-machine.service"
    app_root="/opt/computefield-machine"
    state_root="/var/lib/computefield-machine"
    service_source="packaging/systemd/computefield-machine.service"
    apparmor_profile="computefield-machine-bwrap"
    torch_specs="torch==2.13.0 torchvision==0.28.0"
    torch_index_option="--index-url"
    torch_index_url="https://download.pytorch.org/whl/cu130"
    description="Standalone ComputeField NVIDIA GPU compute service"
    ;;
  cpu)
    package_name="computefield-machine-cpu"
    command_name="computefield-machine-cpu"
    service_user="computefield-machine-cpu"
    service_name="computefield-machine-cpu.service"
    app_root="/opt/computefield-machine-cpu"
    state_root="/var/lib/computefield-machine-cpu"
    service_source="packaging/systemd/computefield-machine-cpu.service"
    apparmor_profile="computefield-machine-cpu-bwrap"
    torch_specs="torch==2.13.0+cpu torchvision==0.28.0+cpu"
    torch_index_option="--extra-index-url"
    torch_index_url="https://download.pytorch.org/whl/cpu"
    description="Standalone ComputeField CPU compute service"
    ;;
  *)
    echo "usage: packaging/build-deb.sh VERSION [gpu|cpu]" >&2
    exit 2
    ;;
esac

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
ROOT="$BUILD_DIR/${package_name}_${VERSION}_amd64"
SOURCE_ROOT="$ROOT$app_root/source"
install -d "$ROOT/DEBIAN" "$SOURCE_ROOT" "$ROOT/usr/bin" "$ROOT/etc/systemd/system" "$ROOT/etc/apparmor.d"
runtime_sources=(
  LICENSE requirements.txt pyproject.toml
  capabilities.py cli.py config.py delta.py executor.py gpu_device.py main.py
  model_artifact.py monitor.py pairing.py sandbox_runtime.py script_guard.py
  speedtest.py workload_runner.py workspace.py
)
tar -C "$SOURCE_DIR" -cf - "${runtime_sources[@]}" | tar -x -C "$SOURCE_ROOT"
chmod -R go-w "$SOURCE_ROOT"
install -m 0755 "$SOURCE_DIR/packaging/ubuntu/computefield-machine" "$ROOT/usr/bin/$command_name"
install -m 0644 "$SOURCE_DIR/$service_source" "$ROOT/etc/systemd/system/$service_name"
cat >"$ROOT/etc/apparmor.d/$apparmor_profile" <<EOF
abi <abi/4.0>,
include <tunables/global>

profile $apparmor_profile $app_root/bin/bwrap flags=(unconfined) {
  userns,
}
EOF
cat >"$ROOT/DEBIAN/control" <<EOF
Package: $package_name
Version: $VERSION
Architecture: amd64
Maintainer: Compute Field Lab, LLC <support@computefield.com>
Depends: python3 (>= 3.10), python3-venv, ca-certificates, apparmor, bubblewrap, util-linux
Section: science
Priority: optional
Description: $description
EOF
cat >"$ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
profile="@PROFILE@"
service_user="@SERVICE_USER@"
service_name="@SERVICE_NAME@"
app_root="@APP_ROOT@"
state_root="@STATE_ROOT@"
apparmor_profile="@APPARMOR_PROFILE@"
if [ "$profile" = gpu ]; then
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || {
    echo "Install a working NVIDIA driver before ComputeField Machine." >&2
    exit 1
  }
fi
install -d -m 0755 "$app_root/bin"
install -o root -g root -m 0755 /usr/bin/bwrap "$app_root/bin/bwrap"
apparmor_parser -r "/etc/apparmor.d/$apparmor_profile" >/dev/null 2>&1 || true
"$app_root/bin/bwrap" --help 2>&1 | grep -q -- '--disable-userns' || {
  echo "Bubblewrap with --disable-userns is required (Ubuntu 24.04 or newer)." >&2
  exit 1
}
id "$service_user" >/dev/null 2>&1 || useradd --system --home "$state_root" --shell /usr/sbin/nologin "$service_user"
install -d -o "$service_user" -g "$service_user" -m 0700 "$state_root"
current="$app_root/venv"
candidate="$app_root/venv.candidate.$$"
link="$app_root/.venv-link.$$"
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
"$candidate/bin/pip" install @TORCH_SPECS@ @TORCH_INDEX_OPTION@ @TORCH_INDEX_URL@
if [ "$profile" = gpu ]; then
  "$candidate/bin/pip" install nvidia-ml-py==12.560.30
fi
"$candidate/bin/pip" install "$app_root/source"
if [ "$profile" = cpu ]; then
  runuser -u "$service_user" -- env \
    MACHINE_IDENTITY_FILE="$state_root/identity.json" \
    MACHINE_WORK_DIR="$state_root/work" \
    MACHINE_ISOLATION_MODE=sandbox \
    MACHINE_COMPUTE_MODE=cpu \
    CUDA_VISIBLE_DEVICES= \
    COMPUTEFIELD_BWRAP="$app_root/bin/bwrap" \
    "$candidate/bin/computefield-machine" doctor || {
      echo "ComputeField Machine requires working unprivileged user namespaces; host policy was not modified." >&2
      exit 1
    }
else
  runuser -u "$service_user" -- env \
    MACHINE_IDENTITY_FILE="$state_root/identity.json" \
    MACHINE_WORK_DIR="$state_root/work" \
    MACHINE_ISOLATION_MODE=sandbox \
    MACHINE_COMPUTE_MODE=auto \
    COMPUTEFIELD_BWRAP="$app_root/bin/bwrap" \
    "$candidate/bin/computefield-machine" doctor || {
      echo "ComputeField Machine requires working unprivileged user namespaces; host policy was not modified." >&2
      exit 1
    }
fi

previous=
if [ -L "$current" ]; then
  previous="$(readlink -f "$current" || true)"
elif [ -d "$current" ]; then
  previous="$app_root/venv.legacy.$$"
  mv "$current" "$previous"
  legacy="$previous"
fi
ln -s "$(basename "$candidate")" "$link"
mv -Tf "$link" "$current"
activated=1
trap - EXIT HUP INT TERM
systemctl daemon-reload || true
systemctl enable "$service_name" || true
if [ -s "$state_root/identity.json" ]; then
  systemctl restart "$service_name"
else
  systemctl stop "$service_name" >/dev/null 2>&1 || true
fi
case "$previous" in
  "$app_root"/venv.candidate.*|"$app_root"/venv.legacy.*)
    [ "$previous" = "$candidate" ] || rm -rf "$previous"
    ;;
esac
EOF
sed -i \
  -e "s|@PROFILE@|$PROFILE|g" \
  -e "s|@SERVICE_USER@|$service_user|g" \
  -e "s|@SERVICE_NAME@|$service_name|g" \
  -e "s|@APP_ROOT@|$app_root|g" \
  -e "s|@STATE_ROOT@|$state_root|g" \
  -e "s|@APPARMOR_PROFILE@|$apparmor_profile|g" \
  -e "s|@TORCH_SPECS@|$torch_specs|g" \
  -e "s|@TORCH_INDEX_OPTION@|$torch_index_option|g" \
  -e "s|@TORCH_INDEX_URL@|$torch_index_url|g" \
  "$ROOT/DEBIAN/postinst"
chmod 0755 "$ROOT/DEBIAN/postinst"
cat >"$ROOT/DEBIAN/prerm" <<EOF
#!/bin/sh
set -e
if [ "\${1:-}" = remove ]; then
  systemctl disable --now "$service_name" >/dev/null 2>&1 || true
  apparmor_parser -R "/etc/apparmor.d/$apparmor_profile" >/dev/null 2>&1 || true
fi
EOF
chmod 0755 "$ROOT/DEBIAN/prerm"
cat >"$ROOT/DEBIAN/postrm" <<EOF
#!/bin/sh
set -e
case "\${1:-}" in
  remove|purge)
    rm -rf "$app_root"
    systemctl daemon-reload >/dev/null 2>&1 || true
    ;;
esac
if [ "\${1:-}" = purge ]; then
  rm -rf "$state_root"
  rm -rf "/etc/${service_name%.service}" "/etc/systemd/system/$service_name.d"
  userdel "$service_user" >/dev/null 2>&1 || true
  systemctl daemon-reload >/dev/null 2>&1 || true
fi
EOF
chmod 0755 "$ROOT/DEBIAN/postrm"

versioned="$SOURCE_DIR/${package_name}_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "$ROOT" "$versioned"
cp "$versioned" "$SOURCE_DIR/${package_name}_amd64.deb"
