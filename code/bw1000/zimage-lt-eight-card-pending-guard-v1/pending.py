"""Protect the exact O01 original dispatch across controller and transport exits."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
from types import ModuleType


PARENT, NODE = "1103606", "f12r2n11"
REQUEST = "e23d24ea0c9c4d00af06154d82dc22e8"
OLD_STANDBY_REQUEST = "3228d0f2042f4224bb72bdf2313617c5"
PRE_REQUEST = "f0674bbd407942a4a2455634c8cb44c4"
POST_REQUEST = "a57632cbc5534cf781abbde00746a7a9"
ORIGINAL = "O01-zimage-lt-eight-card-supervisor-v1-run"
PRE = "O01-zimage-hybrid-pre-" + PRE_REQUEST[:12]
POST = "O01-zimage-hybrid-post-" + POST_REQUEST[:12]
LOCK = ".training80-dcu-standby.lock"
MARKER = ".training80-o01-pending-" + REQUEST
HOST_SCOPE_SHA = "a50bfad3b2e2a7616c464b44ddd5adacbabb8979e79bf94e2c64ae15963d811f"
PCI = frozenset(f"0000:{bus}:00.0" for bus in ("09", "36", "55", "77", "85", "b5", "d5", "f5"))
REQUEST_ID = re.compile("[0-9a-f]{32}")
SHA = re.compile("[0-9a-f]{64}")


def require(test, reason):
    if not test:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, cap=16 << 10):
    require(path.is_file() and not path.is_symlink(), "EXACT_PENDING_RECORD_REQUIRED")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap, "PENDING_RECORD_BOUND")
    return raw


def present(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable(path, record):
    raw = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)
    return digest(raw)


def shared_lock(results, held):
    import fcntl

    require(not held.closed and Path(held.name).resolve(strict=True) == (results / LOCK).resolve(strict=True),
            "REAL_SHARED_STANDBY_LOCK_REQUIRED")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)


def marker_document(results):
    directory = results / MARKER
    require(directory.is_dir() and not directory.is_symlink(), "CANONICAL_PENDING_DIRECTORY")
    path = directory / "claim.json"
    raw = read(path)
    value = json.loads(raw)
    require(value["request"] == REQUEST and value["parent"] == PARENT and value["node"] == NODE
            and value["replay_allowed"] is False and value["no_expiration"] is True, "PENDING_CLAIM_IDENTITY")
    return digest(raw)


def reserve_locked(results, held):
    """Call under the real standby lock before producing or sending any O01 phase."""
    shared_lock(results, held)
    require(not any(present(results / leaf) for leaf in (PRE, ORIGINAL, POST, MARKER)),
            "ORIGINAL_REQUIRES_RECONCILIATION")
    directory = results / MARKER
    directory.mkdir(mode=0o700)
    # An empty marker directory already reserves the exact node after a crash.
    fsync_directory(results)
    return durable(directory / "claim.json", {"request": REQUEST, "parent": PARENT, "node": NODE,
                   "replay_allowed": False, "no_expiration": True, "epoch": time.time()})


def refill_request(value):
    return bool(REQUEST_ID.fullmatch(value) and value not in (REQUEST, OLD_STANDBY_REQUEST, PRE_REQUEST, POST_REQUEST))


def read_grant(results, marker_sha):
    raw = read(results / MARKER / "refill-grant.json")
    grant = json.loads(raw)
    require(grant["original_request"] == REQUEST and grant["marker_SHA256"] == marker_sha
            and refill_request(grant["refill_request"]) and grant["replay_allowed"] is False
            and SHA.fullmatch(grant["scope_SHA256"]), "REFILL_GRANT_IDENTITY")
    eligibility = grant["eligibility"]
    require(eligibility["root_refill_eligibility"] is True and set(eligibility["PCI"]) == PCI
            and len(eligibility["PCI"]) == 8 and SHA.fullmatch(eligibility["post_SHA256"])
            and SHA.fullmatch(eligibility["boot_id_SHA256"]), "REFILL_GRANT_POST")
    return grant, digest(raw)


def decision(results, request, proposal):
    """Called inside the actual controller lock, with no scheduler or time inference."""
    marker_dir = results / MARKER
    try:
        if not any(present(results / leaf) for leaf in (PRE, ORIGINAL, POST, MARKER)):
            return {"blocked": False, "reason": "NO_O01_PENDING_INTENT"}
        marker_sha = marker_document(results)
        if not present(marker_dir / "refill-grant.json"):
            return {"blocked": True, "reason": "ORIGINAL_PENDING_WITHOUT_REFILL_GRANT", "marker_SHA256": marker_sha}
        grant, grant_sha = read_grant(results, marker_sha)
        if present(marker_dir / "released.json"):
            release = json.loads(read(marker_dir / "released.json"))
            require(release["grant_SHA256"] == grant_sha and release["refill_request"] == grant["refill_request"]
                    and release["counted_cards"] == 8 and release["original_request"] == REQUEST
                    and SHA.fullmatch(release["refill_evidence_SHA256"]), "REFILL_RELEASE_IDENTITY")
            return {"blocked": False, "reason": "INDEPENDENT_EIGHT_CARD_REFILL_CONSUMED"}
        allowed = (request == grant["refill_request"] and proposal["parent"] == PARENT
                   and len(proposal["pci"]) == 8 and set(proposal["pci"]) == PCI)
        return {"blocked": not allowed, "reason": "EXACT_REFILL_DELEGATED" if allowed else
                "O01_RESERVED_FOR_EXACT_REFILL", "grant_SHA256": grant_sha}
    except (OSError, ValueError, KeyError, TypeError):
        # Malformed, inaccessible, legacy, and partial records hold this one fixed node.
        return {"blocked": True, "reason": "O01_PENDING_RECEIPT_INCOMPLETE_OR_UNACCESSIBLE"}


def install(namespace, inputs, evidence_directory):
    """Install a final admission filter in the byte-identical standby controller."""
    if NODE not in inputs["nodes"]:
        return
    original = namespace["live_inventory"]
    results = namespace["RESULTS"]

    def checked_live_inventory():
        owners, active = original()
        result = decision(results, inputs["request"], inputs["nodes"][NODE])
        durable(evidence_directory / "o01-pending-decision.json",
                {**result, "original_request": REQUEST, "standby_request": inputs["request"],
                 "node": NODE, "epoch": time.time()})
        if result["blocked"]:
            active = set(active) | {NODE}
        return owners, active

    namespace["live_inventory"] = checked_live_inventory


def bound_host(path, results):
    raw = read(path, 64 << 10)
    require(digest(raw) == HOST_SCOPE_SHA, "PUBLISHED_HOST_SCOPE_SHA")
    scope = ModuleType("published_o01_host_scope")
    scope.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), vars(scope))
    require(Path(scope.RESULTS) == results and scope.REQUEST == REQUEST and scope.NODE == NODE,
            "PUBLISHED_HOST_SCOPE_PLACEMENT")
    return scope


def authorize_refill_locked(results, held, host_path, collect, scope, evidence, refill):
    shared_lock(results, held)
    require(refill_request(refill), "DISTINCT_REFILL_REQUEST")
    marker_sha = marker_document(results)
    host = bound_host(host_path, results)
    scope_pin = host.digest(host.encoded(scope))
    host.read_exact(results / ORIGINAL / "scope.json", scope_pin)
    eligibility = host.refill_eligibility_locked(held, collect, scope, evidence)
    return durable(results / MARKER / "refill-grant.json",
                   {"original_request": REQUEST, "refill_request": refill, "marker_SHA256": marker_sha,
                    "scope_SHA256": scope_pin, "eligibility": eligibility, "replay_allowed": False})


def finish_refill_locked(results, held, host_path, collect, evidence):
    shared_lock(results, held)
    host = bound_host(host_path, results)
    grant, grant_sha = read_grant(results, marker_document(results))
    require(evidence["request"] == grant["refill_request"], "REFILL_DIFFERS_FROM_GRANTED_REQUEST")
    actual = host.consume_refill(collect(), time.time(), os.getuid(), grant["eligibility"], evidence)
    return durable(results / MARKER / "released.json",
                   {"original_request": REQUEST, "refill_request": evidence["request"], "grant_SHA256": grant_sha,
                    "counted_cards": actual["actual_computing_cards"], "refill_step": actual["refill_step"],
                    "refill_evidence_SHA256": host.digest(host.encoded(evidence))})
