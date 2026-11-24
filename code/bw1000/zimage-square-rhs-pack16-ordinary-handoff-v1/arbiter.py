"""Single-node Pack16 business ticket on the existing 05:44 ordinary journal."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from uuid import uuid4


PARENT, NODE, PREDECESSOR = "1103606", "f12r2n10", "1103606.2930"
REQUEST = "a31421758fc74d319923dbce7045cc12"
LABEL = "O01-zimage-square-rhs-pack16-v1"
CALENDAR_SHA = "0730b099d674de7b195d825f5f81c9a12b656e50d79e4e2c1117c1156828863e"
BASELINE_SHA = "316a5ea88ca53484e8e8f10dcf2df1e779b02b041c04445c234ec9e5d6010f59"
PACKAGE = Path(__file__).resolve().parent
SCOPE = "pack16-1103606-f12r2n10-2930"
RESULTS = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
SCOPE_SCHEMA = "bw1000.o01.pack16-ordinary-handoff-v1"
HEX = re.compile("[a-f0-9]{64}")
STEP = re.compile(r"1103606\.[0-9]+")
MAX_TICKET_AGE = 30
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def require(ok, gate):
    if not ok:
        raise ValueError(gate)


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def signature(data):
    return sha256(data).hexdigest()


def read(path, wanted=None):
    require(path.is_file() and not path.is_symlink(), "MISSING_OR_ALIASED_EVIDENCE")
    with path.open("rb") as stream:
        data = stream.read(131073)
    require(0 < len(data) <= 131072 and (wanted is None or signature(data) == wanted), "EVIDENCE_PIN")
    return json.loads(data), signature(data)


def fsync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def publish(path, value):
    data = encode(value)
    pending = path.parent / ("." + uuid4().hex + ".pending")
    require(not path.exists() and not path.is_symlink(), "NO_REPLAY")
    try:
        with pending.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # All writers for this exact node hold the same journal lock.
        os.replace(pending, path)
        fsync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)
    return signature(data)


def qualified_scope(scope):
    target_output = scope["host_output"]
    output = PurePosixPath(target_output)
    require(output.parent == RESULTS and ".." not in output.parts and str(output) == target_output
            and re.fullmatch(r"O01-pack16-2930-[a-f0-9]{16,32}", output.name), "PACK16_EXCLUSIVE_OUTPUT")
    require((scope["parent"], scope["node"], scope["request"], scope["label"]) ==
            (PARENT, NODE, REQUEST, LABEL) and scope["root_admitted"] is True
            and scope["deadline_epoch"] == scope["epoch"] + 240 and scope["gfx"] == "gfx936",
            "PACK16_SCOPE_DIFFERS")


class Ledger:
    def __init__(self, ordinary_journal):
        self.root = ordinary_journal
        self.directory = self.root / SCOPE
        self.lock = self.root / (SCOPE + ".lock")

    @contextmanager
    def guarded(self):
        # This lock never spans transport, Slurm queries, or scheduler waits.
        self.root.mkdir(exist_ok=True)
        try:
            self.lock.mkdir()
        except FileExistsError:
            raise ValueError("EXACT_NODE_ARBITRATION_BUSY") from None
        try:
            yield
        finally:
            self.lock.rmdir()

    def get(self, name):
        path = self.directory / (name + ".json")
        return read(path) if path.exists() or path.is_symlink() else (None, None)

    def write(self, name, value):
        self.directory.mkdir(exist_ok=True)
        fsync_directory(self.root)
        return publish(self.directory / (name + ".json"), value)

    def ordinary(self, node, predecessor, action):
        if node != NODE:
            return action()
        require(predecessor == PREDECESSOR, "EXACT_O01_PREDECESSOR_CHANGED")
        try:
            with self.guarded():
                try:
                    accepted, _ = self.get("accepted")
                    release, _ = self.get("release")
                except (OSError, ValueError):
                    return False
                if accepted and not release:
                    return False
                return action()
        except ValueError as error:
            if str(error) == "EXACT_NODE_ARBITRATION_BUSY":
                return False
            raise

    def accept(self, scope, scope_sha, adoption, adoption_sha, group, now):
        qualified_scope(scope)
        require(HEX.fullmatch(scope_sha) and HEX.fullmatch(adoption_sha)
                and scope["epoch"] - MAX_TICKET_AGE <= now < scope["epoch"] + MAX_TICKET_AGE, "STALE_SCOPE")
        expected = {"schema": SCOPE_SCHEMA, "parent": PARENT, "node": NODE, "predecessor": PREDECESSOR,
                    "calendar_SHA256": CALENDAR_SHA, "group": group, "sole_coordinator_confirmed": True,
                    "legacy_target_dispatch_drained": True,
                    "coordinator_SHA256": signature((PACKAGE / "ordinary.py").read_bytes())}
        require({key: adoption.get(key) for key in expected} == expected
                and 0 <= now - adoption["observed_epoch"] <= 120
                and now < adoption["valid_until_epoch"] <= adoption["observed_epoch"] + 180,
                "COORDINATOR_ADOPTION_NOT_PROVEN")
        with self.guarded():
            for slot in range(3):
                old = self.root / f"{NODE}-{PREDECESSOR}-{slot}.json"
                require(not old.exists() and not old.is_symlink(), "EXISTING_ORDINARY_TICKET")
            require(self.get("accepted") == self.get("release") == (None, None), "NO_REPLAY")
            accepted = {**expected, "request": REQUEST, "label": LABEL, "scope_SHA256": scope_sha,
                        "adoption_SHA256": adoption_sha, "accepted_epoch": now,
                        "host_output": scope["host_output"],
                        "valid_until_epoch": min(adoption["valid_until_epoch"], scope["epoch"] + MAX_TICKET_AGE),
                        "state": "ACCEPTED_UNSENT", "replay_allowed": False}
            return accepted, self.write("accepted", accepted)

    def grant(self, accepted_sha, now):
        with self.guarded():
            accepted, actual = self.get("accepted")
            require(accepted and actual == accepted_sha and self.get("release") == (None, None)
                    and self.get("grant") == (None, None)
                    and accepted["accepted_epoch"] <= now < accepted["valid_until_epoch"], "NO_SEND_GRANT")
            grant = dict(accepted, state="SEND_GRANTED_UNKNOWN", accepted_SHA256=accepted_sha, granted_epoch=now)
            return grant, self.write("grant", grant)

    def release_unsent(self, accepted_sha, now):
        with self.guarded():
            accepted, actual = self.get("accepted")
            require(accepted and actual == accepted_sha and self.get("grant") == (None, None)
                    and self.get("release") == (None, None), "SEND_POSSIBLE_OR_NO_ACCEPTANCE")
            release = dict(state="RELEASED_POSITIVE_UNSENT", accepted_SHA256=accepted_sha,
                           evidence="sole_send_grant_was_never_issued", epoch=now, group=accepted["group"])
            return release, self.write("release", release)

    def release_post(self, accepted_sha, host, host_sha, final, final_sha, now):
        with self.guarded():
            accepted, actual = self.get("accepted")
            grant, grant_sha = self.get("grant")
            require(accepted and actual == accepted_sha and grant and grant["accepted_SHA256"] == accepted_sha
                    and self.get("release") == (None, None), "NO_SUBMITTED_BUSINESS_TICKET")
            require(host["state"] == "TERMINAL" and host["actual_resources_match"] is True
                    and host["scope_SHA256"] == accepted["scope_SHA256"]
                    and host["business_grant_SHA256"] == grant_sha
                    and len(host["accounting"]) == 1, "ORIGINAL_HOST_NOT_TERMINAL")
            row = host["accounting"][0]
            require(len(row) == 7 and STEP.fullmatch(row[0]) and row[0] != PREDECESSOR
                    and row[1] == LABEL and row[2].split()[0].rstrip("+") in TERMINAL and row[5] == NODE
                    and len(row[6].split(",")) == 4
                    and set(row[6].split(",")) in ({"cpu=4", "gres/dcu=1", "mem=8192M", "node=1"},
                                                   {"cpu=4", "gres/dcu=1", "mem=8G", "node=1"}), "ORIGINAL_TERMINAL_ALLOCATION")
            evidence = final["raw_worker_audit"]
            require(final["request"] == REQUEST and final["actual_gpu_step"] == row[0]
                    and final["independent_post"] == "CLEAN"
                    and evidence["actual_accounting"] == row
                    and evidence["scope_SHA256"] == accepted["scope_SHA256"]
                    and evidence["accounting_SHA256"] == host_sha, "INDEPENDENT_POST_REQUIRED")
            release = dict(state="RELEASED_AFTER_TERMINAL_AND_POST", accepted_SHA256=accepted_sha,
                           host_terminal_SHA256=host_sha, independent_final_SHA256=final_sha,
                           actual_gpu_step=row[0], epoch=now, group=accepted["group"])
            return release, self.write("release", release)


def validate_grant(grant, grant_sha, scope, scope_sha, now):
    qualified_scope(scope)
    require(HEX.fullmatch(grant_sha) and grant["schema"] == SCOPE_SCHEMA
            and (grant["parent"], grant["node"], grant["predecessor"], grant["request"], grant["label"]) ==
            (PARENT, NODE, PREDECESSOR, REQUEST, LABEL) and grant["calendar_SHA256"] == CALENDAR_SHA
            and grant["group"] == 6 and grant["sole_coordinator_confirmed"] is True
            and grant["legacy_target_dispatch_drained"] is True
            and grant["coordinator_SHA256"] == signature((PACKAGE / "ordinary.py").read_bytes())
            and grant["state"] == "SEND_GRANTED_UNKNOWN" and grant["replay_allowed"] is False
            and HEX.fullmatch(grant["accepted_SHA256"]) and HEX.fullmatch(grant["adoption_SHA256"])
            and grant["scope_SHA256"] == scope_sha and grant["host_output"] == scope["host_output"]
            and grant["accepted_epoch"] <= grant["granted_epoch"] <= now
            and now < grant["valid_until_epoch"] <= scope["epoch"] + MAX_TICKET_AGE, "PACK16_BUSINESS_GRANT")


def main():
    import ordinary
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("accept", "grant", "release-unsent", "release-post", "status"))
    for option in ("scope", "adoption", "host-terminal", "independent-final"):
        parser.add_argument("--" + option, type=Path)
        parser.add_argument("--" + option + "-sha256")
    parser.add_argument("--accepted-sha256")
    args = parser.parse_args()
    coordinator, ledger, group = ordinary.original()
    now = time.time()
    if args.mode == "status":
        state = {}
        with ledger.guarded():
            for name in ("accepted", "grant", "release"):
                document, digest = ledger.get(name)
                if document:
                    state[name] = {"SHA256": digest, "state": document["state"]}
        print(json.dumps({"group": group, "tickets": state}))
        return 0
    if args.mode == "accept":
        require(args.scope and args.adoption and HEX.fullmatch(args.scope_sha256 or "")
                and HEX.fullmatch(args.adoption_sha256 or ""), "ACCEPTANCE_PINS_REQUIRED")
        scope, scope_sha = read(args.scope, args.scope_sha256)
        adoption, adoption_sha = read(args.adoption, args.adoption_sha256)
        value, digest = ledger.accept(scope, scope_sha, adoption, adoption_sha, group, now)
    else:
        require(HEX.fullmatch(args.accepted_sha256 or ""), "ACCEPTANCE_PIN_REQUIRED")
        if args.mode == "grant":
            value, digest = ledger.grant(args.accepted_sha256, now)
        elif args.mode == "release-unsent":
            value, digest = ledger.release_unsent(args.accepted_sha256, now)
        else:
            require(args.host_terminal and args.independent_final and HEX.fullmatch(args.host_terminal_sha256 or "")
                    and HEX.fullmatch(args.independent_final_sha256 or ""), "TERMINAL_POST_PINS_REQUIRED")
            host, host_sha = read(args.host_terminal, args.host_terminal_sha256)
            final, final_sha = read(args.independent_final, args.independent_final_sha256)
            value, digest = ledger.release_post(args.accepted_sha256, host, host_sha, final, final_sha, now)
    result = {"state": value["state"], "SHA256": digest, "group": group}
    if args.mode.startswith("release-"):
        result["ordinary_refill_argv"] = ordinary.refill_argv(group)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
