"""Let the frozen Pack16 HOST consume the exact group-6 business send grant."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import arbiter


BASELINE = Path(__file__).resolve().parents[1] / "zimage-square-rhs-pack16-host-v1/host.py"


def load_baseline():
    source = BASELINE.read_bytes()
    arbiter.require(arbiter.signature(source) == arbiter.BASELINE_SHA, "PACK16_BASELINE_SOURCE")
    spec = importlib.util.spec_from_file_location("pack16_ordinary_adopted_host", BASELINE)
    host = importlib.util.module_from_spec(spec)
    exec(compile(source, str(BASELINE), "exec"), vars(host))
    return host


def configure(host, grant_path, grant_sha, scope_path, scope_sha, target_output, *, clock=time.time):
    arbiter.require(all(arbiter.HEX.fullmatch(value or "") for value in (grant_sha, scope_sha)),
                    "PACK16_BUSINESS_GRANT_PINS")
    grant, _ = arbiter.read(grant_path, grant_sha)
    scope, _ = arbiter.read(scope_path, scope_sha)
    arbiter.validate_grant(grant, grant_sha, scope, scope_sha, clock())
    arbiter.require(target_output == grant["host_output"], "PACK16_EXCLUSIVE_OUTPUT")

    def exact_business_exclusion(repo, calendar, node):
        arbiter.require(node == arbiter.NODE and repo.resolve() == BASELINE.parents[3].resolve(),
                        "PACK16_BUSINESS_TARGET")
        arbiter.read(calendar, arbiter.CALENDAR_SHA)
        arbiter.validate_grant(grant, grant_sha, scope, scope_sha, clock())

    writer = host.exclusive

    def bind_business_grant(path, value):
        if path.name in {"intent.json", "startup.json", "host-terminal.json"}:
            value = dict(value, business_grant_SHA256=grant_sha)
        return writer(path, value)

    host.ordinary_exclusion = exact_business_exclusion
    host.exclusive = bind_business_grant


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--business-grant", type=Path)
    parser.add_argument("--business-grant-sha256")
    adapter, baseline_arguments = parser.parse_known_args()
    host = load_baseline()
    if baseline_arguments and baseline_arguments[0] == "run":
        required = argparse.ArgumentParser(add_help=False)
        required.add_argument("--scope", required=True, type=Path)
        required.add_argument("--scope-sha256", required=True)
        required.add_argument("--target-output", required=True)
        args, _ = required.parse_known_args(baseline_arguments[1:])
        arbiter.require(adapter.business_grant is not None, "PACK16_BUSINESS_GRANT_REQUIRED")
        configure(host, adapter.business_grant, adapter.business_grant_sha256, args.scope, args.scope_sha256,
                  args.target_output)
    sys.argv[1:] = baseline_arguments
    return host.main()


if __name__ == "__main__":
    raise SystemExit(main())
