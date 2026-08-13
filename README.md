<!-- Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved. -->
# ComputeField Machine

The open-source compute client for a Compute Field platform. It is installed on
compute computers, not on the platform server. It connects outbound, processes
one leased workload at a time, and clears task data before reuse. Docker
Desktop is not required for the native packages.

## Install

Ubuntu 24.04 or newer, x86-64, with a working NVIDIA driver:

```bash
curl -fsSL https://github.com/ComputeField/machine/releases/latest/download/bootstrap-ubuntu.sh \
  | sudo -E bash
```

CPU-only Ubuntu 24.04 or newer, x86-64:

```bash
curl -fsSL https://github.com/ComputeField/machine/releases/latest/download/bootstrap-ubuntu-cpu.sh \
  | sudo -E bash
```

macOS with Apple silicon:

```bash
curl -fsSL https://github.com/ComputeField/machine/releases/latest/download/bootstrap-macos.sh | bash
```

From a source checkout:

```bash
sudo ./packaging/install-ubuntu.sh       # Ubuntu
sudo ./packaging/install-ubuntu-cpu.sh   # Ubuntu CPU service
./packaging/install-macos.sh             # macOS MPS/CPU
```

Tagged releases also contain separate GPU and CPU Debian packages and their
SHA-256 checksums. Each bootstrap installer verifies its package before
installation.
The Ubuntu package installs a hardened systemd service; the macOS package
installs a launchd agent. Dependencies stay inside the application virtual
environment. The first install downloads the pinned PyTorch runtime and can
take several minutes; an interrupted upgrade does not replace the previous
working environment.

Ubuntu's restricted-user-namespace policy is supported without disabling it:
the package provisions a path-specific AppArmor permission for its private,
root-owned Bubblewrap executable. No global security sysctl is changed.

## Pair

Open **Machines → Connect machine** at `https://computefield.net/machines` and
copy the command shown there. The CLI defaults to the same public HTTPS origin:

```bash
computefield-machine pair ABCD-EF12-3456
```

Use `computefield-machine-cpu pair ABCD-EF12-3456` for the CPU package. GPU
and CPU packages can coexist on one server. They have separate service users,
state directories, credentials, and systemd units, and each needs its own code.

Compare the fingerprint in the terminal and browser, confirm it in the
browser, then accept or decline the cross-account workload prompt. `--share`
or `--private` records the same choice non-interactively. On Ubuntu the command
requests `sudo`, writes the service identity, and restarts the service
automatically. On macOS, start it after pairing:

```bash
computefield-machine start                    # macOS
computefield-machine status
```

One account may pair several Machines. Private use is the default. Cross-account
work is available in every native packaged installation after explicit owner
consent:

```bash
computefield-machine sharing enable
computefield-machine sharing disable
```

Each workload runs in a fresh credential-free OS sandbox. Linux uses namespaces
through Bubblewrap 0.9 or newer; macOS uses the built-in Seatbelt facility.
Neither route needs BIOS virtualization, Docker Desktop, the Mac App Store, or
a paid runtime license. Installers provision and verify the sandbox before
reporting success.

## Operate

```bash
journalctl -u computefield-machine -f          # Ubuntu logs
sudo systemctl stop computefield-machine       # Ubuntu stop
sudo systemctl stop computefield-machine-cpu   # Ubuntu CPU stop
computefield-machine stop                      # macOS stop
computefield-machine unpair --yes              # remove local identity
```

Unlinking a Machine in the browser immediately revokes its server credential.
Installers honor standard proxy and TLS environment variables.

## Develop and release

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
.venv/bin/ruff check .
python tools/generate-python-reference.py
docker build --target cpu -t computefield-machine:dev .
```

Release checks test and audit Python, validate packaging, and build the CPU
image. A reviewed release is built explicitly with
`packaging/build-release.sh VERSION`;
GitHub automation is intentionally disabled. Upload the generated GPU/CPU
Debian packages, macOS source archive, three bootstrap installers, and their
SHA-256 files to the matching manual release. This directory is an independent
public repository; the private platform repository is not needed to build or run it.
The generated Python interface index is in
[`docs/reference/python-api.md`](docs/reference/python-api.md).
