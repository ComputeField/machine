# Copyright (c) 2026 Compute Field Lab, LLC, Abu-Dhabi. All rights reserved.

import argparse
import asyncio
import json
import sys
from pathlib import Path

from config import settings
from pairing import pair
from sandbox_runtime import self_test as sandbox_self_test


def _prepare_sharing() -> None:
    if not settings.sharing_supported:
        raise SystemExit(
            "Shared mode requires the packaged per-workload sandbox. Reinstall ComputeField Machine to provision it."
        )
    Path(settings.work_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
    sandbox_self_test(settings.work_dir)


def _pair_sharing_consent(args: argparse.Namespace) -> bool:
    if args.share:
        return True
    if args.private:
        return False
    if not sys.stdin.isatty():
        return False
    answer = input("Allow verified cross-account workloads on this machine? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="computefield-machine")
    subcommands = parser.add_subparsers(dest="command", required=True)
    pair_command = subcommands.add_parser("pair", help="connect this machine to an account")
    pair_command.add_argument("code", help="short code created on the Machines page")
    pair_command.add_argument(
        "--api",
        default="https://computefield.net",
        help="ComputeField HTTPS base URL (default: https://computefield.net)",
    )
    pair_command.add_argument("--name", default="", help="name shown on the Machines page")
    sharing_choice = pair_command.add_mutually_exclusive_group()
    sharing_choice.add_argument("--share", action="store_true", help="consent to cross-account workloads")
    sharing_choice.add_argument("--private", action="store_true", help="keep this Machine account-private")
    subcommands.add_parser("status", help="show local connection state")
    subcommands.add_parser("run", help="run the foreground compute service")
    subcommands.add_parser("doctor", help="verify the workload sandbox on this host")
    unpair_command = subcommands.add_parser(
        "unpair",
        help="delete the local identity before pairing to another account",
    )
    unpair_command.add_argument("--yes", action="store_true", help="confirm local credential deletion")
    sharing_command = subcommands.add_parser(
        "sharing",
        help="control explicit participation in cross-account work",
    )
    sharing_command.add_argument("mode", choices=("enable", "disable", "status"))
    args = parser.parse_args()

    if args.command == "pair":
        pair(args.api, args.code, name=args.name)
        sharing = _pair_sharing_consent(args)
        if sharing:
            _prepare_sharing()
        settings.update_identity(sharing_enabled=sharing)
        print(f"Cross-account sharing {'enabled' if sharing else 'disabled'}.")
    elif args.command == "status":
        identity = settings.saved_identity
        print(
            json.dumps(
                {
                    "paired": bool(identity.get("credential")),
                    "host_id": identity.get("host_id"),
                    "broker": identity.get("broker_url"),
                    "sharing": settings.sharing_enabled,
                    "isolation": settings.machine_isolation_mode,
                },
                indent=2,
            )
        )
    elif args.command == "unpair":
        if not args.yes:
            raise SystemExit("Refusing to delete the local identity without --yes")
        settings.clear_identity()
        print("Local Machine identity deleted. The server-side Machine must be unlinked separately.")
    elif args.command == "sharing":
        if args.mode == "status":
            print("enabled" if settings.sharing_enabled else "disabled")
        else:
            enabled = args.mode == "enable"
            if enabled:
                _prepare_sharing()
            settings.update_identity(sharing_enabled=enabled)
            print(
                f"Cross-account sharing {'enabled' if enabled else 'disabled'}. "
                "Restart computefield-machine to apply the change."
            )
    elif args.command == "doctor":
        Path(settings.work_dir).mkdir(mode=0o700, parents=True, exist_ok=True)
        backend = sandbox_self_test(settings.work_dir)
        print(f"Workload sandbox ready: {backend}")
    else:
        from main import main as run_service

        asyncio.run(run_service())


if __name__ == "__main__":
    main()
