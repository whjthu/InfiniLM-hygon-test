"""Create an exact source-only packet and remote Python command without connecting."""

from __future__ import annotations

import argparse
from base64 import b64encode
import json
from pathlib import Path
import shlex
from uuid import uuid4

import install


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
RENDERED = REPO / "local-only/O01-pack16-2930-device-runtime-candidate-20261125"


def packet(runtime_mode):
    closure_raw = install.read_exact(PACKAGE / "SOURCE_CLOSURE.json", 16384, install.CLOSURE_SHA)
    closure = json.loads(closure_raw)
    install.exact_runtime(RENDERED, closure)
    files = {}
    for name, expected in closure["members"].items():
        runtime = name.startswith("runtime/")
        if runtime and runtime_mode == "reuse":
            continue
        path = RENDERED / Path(name).name if runtime else REPO / name
        data = install.read_exact(path, expected["bytes"], expected["SHA256"])
        install.require(len(data) == expected["bytes"], "LOCAL_MEMBER_SIZE")
        files[name] = b64encode(data).decode("ascii")
    raw = install.encode({"closure": b64encode(closure_raw).decode("ascii"),
                          "files": files, "runtime_mode": runtime_mode})
    install.decode_packet(raw, install.pin(raw)["SHA256"])
    return raw


def remote_argv(stage_id, raw, original=None):
    install.stage_name(stage_id)
    loader = (PACKAGE / "install.py").read_text(encoding="utf-8")
    argv = ["/usr/bin/timeout", "--kill-after=5s", "55s", "/usr/local/python3.12/bin/python3",
            "-I", "-S", "-B", "-c", loader, "--stage-id", stage_id, "--packet-sha256", install.pin(raw)["SHA256"]]
    if original is not None:
        argv.extend(["--existing-original-envelope", original])
    return argv


def prepare(output, stage_id, runtime_mode, original=None):
    install.require(not output.exists() and not output.is_symlink(), "OUTPUT_NOT_EXCLUSIVE")
    raw = packet(runtime_mode)
    argv = remote_argv(stage_id, raw, original)
    remote = (shlex.join(argv) + "\n").encode()
    summary = {
        "schema": install.SCHEMA + "-local-packet",
        "source_closure_SHA256": install.CLOSURE_SHA, "packet": install.pin(raw),
        "remote_command": install.pin(remote), "installer": install.pin((PACKAGE / "install.py").read_bytes()),
        "stage_id": stage_id, "target_namespace": str(install.ROOT / "incoming" / install.stage_name(stage_id)),
        "runtime_mode": runtime_mode, "target_contacted": False,
        "transmitted_members": 20 if runtime_mode == "reuse" else 26,
        "private_calendar_in_packet": False, "original_raw_envelope_in_packet": False,
    }
    output.mkdir()
    install.save(output / "packet.json", raw)
    install.save(output / "remote-command.txt", remote)
    install.save(output / "packet-pins.json", install.encode(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage-id", default=None)
    parser.add_argument("--runtime-mode", choices=("reuse", "include"), default="reuse")
    parser.add_argument("--existing-original-envelope", default=None)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.stage_id or uuid4().hex, args.runtime_mode,
                            args.existing_original_envelope), sort_keys=True))


if __name__ == "__main__":
    main()
