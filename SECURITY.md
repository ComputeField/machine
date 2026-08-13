<!-- Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved. -->
# Security

Machine connects outbound over TLS and stores one revocable, host-scoped
credential with owner-only permissions. Pairing requires confirmation of a
short-lived code and matching fingerprint. Task downloads use short-lived
URLs; task directories are exclusive and erased before the next lease.

The native packaged controller never executes workload code and never passes
its credential or environment to a workload. Each task runs in a new
OS-sandboxed interpreter:
Linux uses separate user, mount, PID, IPC, UTS, cgroup and network namespaces;
macOS uses Seatbelt to deny network access, controller/identity files, Keychain,
SSH data, external volumes and writes outside the task directory. Shards cross
the boundary through a framed stdio request to the controller. Install and
startup perform an escape self-test.

On Linux the installer accepts only an ordinary, non-setuid Bubblewrap with
`--disable-userns` support and working unprivileged user namespaces. The
published package therefore requires Ubuntu 24.04 or newer. It aborts if the
kernel denies that boundary. Ubuntu packages install a root-owned private copy
of Bubblewrap and a path-specific AppArmor profile granting only that executable
the `userns` permission required by Ubuntu 24.04. They never change a global
user-namespace sysctl, make a binary setuid, disable AppArmor, enable
virtualization, or change BIOS settings.

The Docker Compose definition is a private development harness, not a shared
host boundary. It advertises no workload isolation and cannot opt into
cross-account work.

The Python syntax gate remains defense in depth, not the security boundary.
Model archives are treated as data: extraction is bounded, executable
serialized modules are rejected, and remote model code is never imported. The
platform treats every Machine and result as untrusted and enforces leases,
limits, accounting, tensor validation, and aggregation on the server.

Shared mode is opt-in. As with any local compute runtime, OS- and GPU-driver
vulnerabilities or resource exhaustion remain outside the guarantee of the
process sandbox; keep the host updated and disable sharing when it is not in
use. The Apache-2.0 license supplies the applicable warranty and liability
terms.

Report security issues privately to security@computefield.com. Do not include
credentials or customer data.
