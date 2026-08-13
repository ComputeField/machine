#!/usr/bin/env bash
# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.
set -Eeuo pipefail

APP_ROOT="${1:?usage: install-linux-sandbox.sh APP_ROOT PROFILE_NAME}"
PROFILE_NAME="${2:?usage: install-linux-sandbox.sh APP_ROOT PROFILE_NAME}"
PROFILE_PATH="/etc/apparmor.d/$PROFILE_NAME"
temporary="$(mktemp)"
trap 'rm -f "$temporary"' EXIT

install -d -m 0755 "$APP_ROOT/bin"
install -o root -g root -m 0755 /usr/bin/bwrap "$APP_ROOT/bin/bwrap"
{
  echo 'abi <abi/4.0>,'
  echo 'include <tunables/global>'
  echo
  echo "profile $PROFILE_NAME $APP_ROOT/bin/bwrap flags=(unconfined) {"
  echo '  userns,'
  echo '}'
} >"$temporary"
install -o root -g root -m 0644 "$temporary" "$PROFILE_PATH"

# Loading can legitimately fail when AppArmor is disabled. The mandatory
# sandbox self-test that follows installation remains the final authority.
apparmor_parser -r "$PROFILE_PATH" >/dev/null 2>&1 || true
