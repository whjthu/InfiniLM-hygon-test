"""Enforce the kernel's exact numbered cgroup before calling the frozen Slurm attestation."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


ATTESTED_SHA = "c7f13669918f720ea195cb49d0be843681d3155b025520caa8b1887babbd6ee6"


def exact(path, pin, maximum):
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular staged file required")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if not 0 < len(raw) <= maximum or sha256(raw).hexdigest() != pin:
        raise ValueError("exact staged file differs")
    return raw


def run(args):
    here = Path(__file__).resolve(strict=True)
    exact(here, args.launcher_sha256, 65536)
    path = here.with_name("node_entry.py")
    if args.entry_sha256 != ATTESTED_SHA:
        raise ValueError("approved attested entry differs")
    module = ModuleType("original_o01_attested_entry")
    module.__file__ = str(path)
    exec(compile(exact(path, ATTESTED_SHA, 65536), str(path), "exec"), vars(module))
    scope = json.loads(exact(args.run / "scope.json", args.scope_sha256, 16384))
    if module.local_identity(scope)["cgroup_names_step"] is not True:
        raise ValueError("kernel cgroup does not name this numbered Slurm step")
    return module.host(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--entry-sha256", required=True)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    raise SystemExit(run(parser.parse_args()))
