#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

VERSION="${1:?usage: packaging/build-release.sh VERSION}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SOURCE_DIR"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Release artifacts require a clean reviewed Git revision." >&2
  exit 1
fi
project_version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' pyproject.toml | head -n 1)"
[[ "$VERSION" == "$project_version" ]] || {
  echo "Release version $VERSION does not match pyproject.toml ($project_version)." >&2
  exit 1
}

artifacts=(
  computefield-machine_amd64.deb
  computefield-machine_amd64.deb.sha256
  computefield-machine_macos-source.tar.gz
  computefield-machine_macos-source.tar.gz.sha256
  bootstrap-ubuntu.sh
  bootstrap-ubuntu.sh.sha256
  bootstrap-macos.sh
  bootstrap-macos.sh.sha256
)
rm -f -- "${artifacts[@]}" "computefield-machine_${VERSION}_amd64.deb"

if [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] && command -v dpkg-deb >/dev/null; then
  packaging/build-deb.sh "$VERSION"
else
  command -v docker >/dev/null || {
    echo "Building the amd64 Debian package on this host requires Docker Engine." >&2
    exit 1
  }
  docker info >/dev/null || {
    echo "Docker Engine is not running." >&2
    exit 1
  }
  docker run --rm --platform linux/amd64 \
    --volume "$SOURCE_DIR:/source" \
    --workdir /source \
    debian:bookworm-slim \
    bash -lc 'apt-get update && apt-get install -y --no-install-recommends ca-certificates dpkg-dev git && packaging/build-deb.sh "$1"' \
    build-release "$VERSION"
fi
rm -f -- "computefield-machine_${VERSION}_amd64.deb"
git archive --format=tar.gz --prefix=computefield-machine/ \
  --output=computefield-machine_macos-source.tar.gz HEAD
cp packaging/bootstrap-ubuntu.sh bootstrap-ubuntu.sh
cp packaging/bootstrap-macos.sh bootstrap-macos.sh
checksum() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}
checksum computefield-machine_amd64.deb >computefield-machine_amd64.deb.sha256
checksum computefield-machine_macos-source.tar.gz >computefield-machine_macos-source.tar.gz.sha256
checksum bootstrap-ubuntu.sh >bootstrap-ubuntu.sh.sha256
checksum bootstrap-macos.sh >bootstrap-macos.sh.sha256
echo "Release artifacts built for $VERSION. Upload all eight files to the manual GitHub release."
