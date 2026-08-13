# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

"""OS-backed workload sandboxing without a VM or desktop container product."""

from __future__ import annotations

import glob
import os
import platform
import secrets
import shutil
import stat
import subprocess  # nosec B404
import sys
import sysconfig
import tempfile
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    """The required host sandbox could not be constructed."""


def backend_name() -> str:
    system = platform.system()
    if system == "Linux":
        return "bubblewrap"
    if system == "Darwin":
        return "seatbelt"
    raise SandboxUnavailable(f"ComputeField workload isolation is not supported on {system}")


def _clean_environment(task_dir: str) -> dict[str, str]:
    source = os.environ
    keep = {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "PYTORCH_ENABLE_MPS_FALLBACK",
    }
    environment = {key: source[key] for key in keep if source.get(key)}
    environment.update(
        {
            "HOME": os.path.join(task_dir, "home"),
            "TMPDIR": os.path.join(task_dir, "tmp"),
            "TEMP": os.path.join(task_dir, "tmp"),
            "TMP": os.path.join(task_dir, "tmp"),
            "PATH": os.path.dirname(sys.executable) + os.pathsep + "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(
                (
                    str(Path(__file__).resolve().parent),
                    str(Path(sysconfig.get_path("purelib")).resolve()),
                )
            ),
        }
    )
    return environment


def _linux_command(command: list[str], task_dir: str) -> list[str]:
    configured = os.environ.get("COMPUTEFIELD_BWRAP", "").strip()
    bwrap = configured or shutil.which("bwrap")
    if not bwrap:
        raise SandboxUnavailable("bubblewrap is required; reinstall ComputeField Machine")

    metadata = os.stat(bwrap)
    if metadata.st_mode & stat.S_ISUID:
        raise SandboxUnavailable("refusing a setuid bubblewrap installation")
    if configured and (metadata.st_uid != 0 or metadata.st_mode & 0o022):
        raise SandboxUnavailable("configured bubblewrap must be root-owned and not group/world-writable")
    args = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--disable-userns",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--unshare-net",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/run",
    ]
    code_root = str(Path(__file__).resolve().parent)
    python_root = str(Path(sys.prefix).resolve())
    python_base = str(Path(sys.base_prefix).resolve())
    for path in (
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/opt/computefield-machine",
        code_root,
        python_root,
        python_base,
    ):
        if os.path.exists(path):
            args.extend(("--ro-bind", path, path))
    for path in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/ssl", "/sys"):
        if os.path.exists(path):
            args.extend(("--ro-bind", path, path))
    for pattern in ("/dev/nvidia*", "/dev/dri/renderD*"):
        for device in sorted(glob.glob(pattern)):
            args.extend(("--dev-bind", device, device))
    args.extend(("--bind", task_dir, task_dir, "--chdir", task_dir))
    return args + command


def _quote_profile(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _macos_executable(executable: str) -> str:
    """Skip the Homebrew framework launcher so the sandbox starts Python itself."""
    executable = os.path.realpath(executable)
    version_root = Path(executable).parent.parent
    framework_python = version_root / "Resources/Python.app/Contents/MacOS/Python"
    if framework_python.is_file():
        return str(framework_python.resolve())
    return executable


def _macos_profile(task_dir: str, command: list[str]) -> str:
    executable = os.path.realpath(command[0])
    code_root = str(Path(__file__).resolve().parent)
    python_root = str(Path(sys.prefix).resolve())
    python_base = str(Path(sys.base_prefix).resolve())
    user_home = str(Path.home().resolve())
    protected_root = str(Path(task_dir).resolve().parent.parent)
    readable_runtime = " ".join(
        f"(subpath {_quote_profile(path)})" for path in sorted({code_root, python_root, python_base, task_dir})
    )
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write* (require-not (subpath {_quote_profile(task_dir)})))
(deny file-read*
  (require-all
    (subpath {_quote_profile(user_home)})
    (require-not (require-any {readable_runtime}))))
(deny file-read*
  (require-all
    (subpath {_quote_profile(protected_root)})
    (require-not (require-any {readable_runtime}))))
(deny file-read* (subpath "/Library/Keychains"))
(deny file-read*
  (require-all
    (subpath "/private/var/folders")
    (require-not (require-any {readable_runtime}))))
(deny file-read* (subpath "/private/var/root"))
(deny file-read* (subpath "/private/etc/ssh"))
(deny file-read* (subpath "/Volumes"))
(deny mach-lookup (regex #"^com\\.apple\\.(securityd|secd)(\\..*)?$"))
(deny mach-lookup (regex #"^com\\.apple\\.(pboard|pasteboard)(\\..*)?$"))
(deny mach-lookup (regex #"^com\\.apple\\.(tccd|audio|audiohald|cmio|locationd|Bluetooth|AddressBook|CalendarAgent)(\\..*)?$"))
(deny appleevent-send)
(deny process-fork)
(deny process-exec (require-not (literal {_quote_profile(executable)})))
"""


def sandbox_command(command: list[str], task_dir: str) -> list[str]:
    """Wrap command in the native host sandbox; never return it unwrapped."""
    task_dir = str(Path(task_dir).resolve())
    executable = os.path.realpath(command[0])
    if platform.system() == "Darwin":
        executable = _macos_executable(executable)
    command = [executable, *command[1:]]
    Path(task_dir, "home").mkdir(mode=0o700, exist_ok=True)
    Path(task_dir, "tmp").mkdir(mode=0o700, exist_ok=True)
    backend = backend_name()
    if backend == "bubblewrap":
        return _linux_command(command, task_dir)
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        raise SandboxUnavailable("macOS Seatbelt sandbox-exec is unavailable")
    profile = Path(task_dir, ".workload.sb")
    profile.write_text(_macos_profile(task_dir, command), encoding="utf-8")
    os.chmod(profile, 0o600)
    return [sandbox_exec, "-f", str(profile), *command]


def popen(command: list[str], task_dir: str) -> subprocess.Popen[str]:
    """Start a credential-free, network-free workload process."""
    wrapped = sandbox_command(command, task_dir)
    return subprocess.Popen(  # noqa: S603  # nosec B603
        wrapped,
        cwd=task_dir,
        env=_clean_environment(task_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )


def self_test(work_dir: str) -> str:
    """Prove that a new process can run but cannot read identity or use IP networking."""
    root = Path(work_dir).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    probe = root.parent / f".computefield-sandbox-denied-{os.getpid()}-{secrets.token_hex(8)}"
    probe_value = "must-not-be-readable"
    probe.write_text(probe_value, encoding="utf-8")
    task = Path(tempfile.mkdtemp(prefix=".sandbox-self-test-", dir=root))
    program = f"""import pathlib, socket, subprocess, sys
failures = []
probe = pathlib.Path({str(probe)!r})
try:
    probe.read_text()
    failures.append('external-read')
except Exception:
    pass
try:
    probe.write_text('escaped')
except Exception:
    pass
try:
    sock = socket.socket()
    sock.settimeout(1)
    sock.connect(('1.1.1.1', 53))
    failures.append('network')
except Exception:
    pass
if sys.platform == 'darwin':
    try:
        keychain = subprocess.run(
            ['/usr/bin/security', 'list-keychains'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if keychain.returncode == 0:
            failures.append('keychain')
    except Exception:
        pass
pathlib.Path('write-ok').write_text('ok')
if failures:
    print(','.join(failures))
sys.exit(0 if not failures else 71)
"""
    try:
        process = popen([sys.executable, "-c", program], str(task))
        stdout, stderr = process.communicate(timeout=20)
        if process.returncode != 0:
            detail = (stderr or stdout).strip()[:2000]
            raise SandboxUnavailable(f"workload sandbox self-test failed ({process.returncode}): {detail}")
        if probe.read_text(encoding="utf-8") != probe_value:
            raise SandboxUnavailable("workload sandbox self-test failed: external-write")
        return backend_name()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxUnavailable(f"workload sandbox self-test failed: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)
        shutil.rmtree(task, ignore_errors=True)
