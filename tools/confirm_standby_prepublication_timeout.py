"""Verify the one original refill that stopped in publication checking before GPU send."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


WRAPPER = "0bec61edb5e84266b545b4cda1441f09"
ADMISSION = "f84de0c6888d44a3b467703d1d183452"
JOURNAL = "training80-1738-ordinary-renewal-dispatch"
NODES = {"f13r4n15": "1194900.793", "f13r3n04": "1194900.795"}
OUTCOME_SHA = "c18858d6c5c4875bc5396e04130449c959e32d2840829fb03ffef73ecf6a3406"
STDOUT_SHA = "9c1318b9024125d7157bfe3e0f2a269434ce4c0c13146052d446cd9917023e6c"
STDERR_SHA = "4983cfdd246c128a5af0d7538b352e190f234d4ed71b1641ef0ae513d8d749a3"
REFILL_SHA = "04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73"
PUBLICATION_CHECK_SHA = "e5453b1aa9d20bb298952f53bd307f9d20e52dbbcde7874d4ffb4510de2fd025"
DISPATCH_SHA = "1d849fe990294c77b11b812992ab752ef23f3a47c612f2b9c2818302f64a92c7"


def read_pin(path, expected, cap):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or substituted original evidence")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap or sha256(raw).hexdigest() != expected:
        raise ValueError("original evidence digest differs")
    return raw


def confirmed_nodes(journal, previous):
    """Return the exact nodes proven to have stopped before the GPU send."""
    if not isinstance(previous, dict) or previous.get("wrapper_invocation") != WRAPPER:
        return frozenset()
    try:
        return verify(journal, previous)
    except (OSError, ValueError, KeyError, TypeError):
        return frozenset()


def verify(journal, previous):
    if journal.name != JOURNAL or journal.parent.name != "local-only" or journal.is_symlink():
        raise ValueError("unexpected standby journal")
    journal = journal.resolve(strict=True)
    repo = journal.parent.parent
    outcome = read_pin(journal / f"{WRAPPER}-outcome.json", OUTCOME_SHA, 4096)
    if (json.loads(outcome) != previous or previous["batch"] != 1 or previous["refill_rc"] != 1
            or previous["transport_observation"] != "REFILL_PROCESS_RETURNED"
            or previous["no_GPU_send_confirmed"] is not False
            or previous["terminal_ready_nodes"] != list(NODES)):
        raise ValueError("unexpected original wrapper state")
    stdout = read_pin(journal / f"{WRAPPER}-stdout.private", STDOUT_SHA, 8192)
    stderr = read_pin(journal / f"{WRAPPER}-stderr.private", STDERR_SHA, 8192).decode()
    actual = json.loads(stdout)
    if set(actual) != {"admission", "current_running_numbered_DCU", "all_reported_numbered_DCU_lower_bound"}:
        raise ValueError("the only standard output must be the previous device admission")
    admission = actual["admission"]
    if admission["request"] != ADMISSION or admission["clear_cards"] != 16 or admission["fresh"] is not True:
        raise ValueError("unexpected original admission")
    if ("File \"<stdin>\", line 119, in <module>" not in stderr
            or "git('ls-remote', '--exit-code', 'origin', BRANCH)" not in stderr
            or not re.search(r"(?m)^subprocess\.TimeoutExpired: .*'ls-remote'.*timed out after 12 seconds\s*$", stderr)):
        raise ValueError("the original failure does not identify the sole publication-check call")
    refill = read_pin(repo.parent / "run-dcu-standby-cycle-20261123.ps1", REFILL_SHA, 50000)
    read_pin(repo / "tools/prepare_training80_vggt_admission.py", PUBLICATION_CHECK_SHA, 30000)
    read_pin(repo / "code/bw1000/fleet-dcu-standby-v2/dispatch.py", DISPATCH_SHA, 50000)
    markers = (b"mod.published(repo,publication", b"request=uuid4().hex",
               b'dest/"standby-intent.json"', b'dest/"bridge-route.json"',
               b"q=subprocess.run(inner+[command]")
    offsets = [refill.index(marker) for marker in markers]
    if offsets != sorted(offsets) or len(set(offsets)) != len(markers):
        raise ValueError("unexpected GPU request or send order in original source")
    directory = journal.parent / f"training80-replacement-admission-{ADMISSION}"
    if directory.is_symlink() or not (directory / "receipt.json").is_file():
        raise ValueError("original device admission directory is missing")
    downstream = ("standby-intent.json", "bridge-route.json", "standby-stdout.json", "standby-stderr.private")
    if any((directory / name).exists() or (directory / name).is_symlink() for name in downstream):
        raise ValueError("unexpected downstream request or transport artifacts")
    for node, step in NODES.items():
        lease = journal / f"{node}-{step}-0.json"
        if lease.is_symlink() or not lease.is_file() or lease.stat().st_size > 8192:
            raise ValueError("original node journal is missing")
        raw = json.loads(lease.read_bytes())
        if (raw["wrapper_invocation"], raw["node"], raw["predecessor"], raw["batch"]) != (WRAPPER, node, step, 1):
            raise ValueError("original per-node journal does not bind the timeout")
    return frozenset(NODES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    args = parser.parse_args()
    previous = json.loads(read_pin(args.journal / f"{WRAPPER}-outcome.json", OUTCOME_SHA, 4096))
    nodes = verify(args.journal, previous)
    print(json.dumps({"wrapper": WRAPPER, "original_device_admission": ADMISSION,
                      "prepublication_timeout_confirmed_before_gpu_send": sorted(nodes),
                      "new_device_admission_required": True}, sort_keys=True))


if __name__ == "__main__":
    main()
