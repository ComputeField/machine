#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

[[ "$(uname -s)" == Darwin ]] || { echo "This installer is for macOS" >&2; exit 1; }
[[ "$(uname -m)" == arm64 ]] || { echo "The current PyTorch release requires Apple silicon" >&2; exit 1; }
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$HOME/Library/Application Support/ComputeField Machine"
PYTHON="$(command -v python3 || true)"
if [[ -z "$PYTHON" ]] || ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  command -v brew >/dev/null 2>&1 || { echo "Install Homebrew first" >&2; exit 1; }
  brew install python@3.11
  PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
fi
mkdir -p "$PREFIX" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
"$PYTHON" -m venv "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --disable-pip-version-check --upgrade pip
"$PREFIX/venv/bin/pip" install torch==2.13.0 torchvision==0.28.0
"$PREFIX/venv/bin/pip" install "$SOURCE_DIR"
sed -e "s|__PREFIX__|$PREFIX|g" -e "s|__HOME__|$HOME|g" \
  "$SOURCE_DIR/packaging/launchd/com.computefield.machine.plist" \
  >"$HOME/Library/LaunchAgents/com.computefield.machine.plist"
if command -v brew >/dev/null 2>&1; then
  BIN_DIR="$(brew --prefix)/bin"
else
  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
fi
install -m 0755 "$SOURCE_DIR/packaging/macos/computefield-machine" "$BIN_DIR/computefield-machine"
"$BIN_DIR/computefield-machine" doctor
echo "Installed with MPS/CPU support. Pair with:"
echo "computefield-machine pair CODE"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  echo "Add $BIN_DIR to PATH before running the command."
fi
echo "After pairing, run: computefield-machine start"
