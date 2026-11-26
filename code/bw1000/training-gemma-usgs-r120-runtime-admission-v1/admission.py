"""One-shot Gemma R120 successor admission, independent of the elapsed release receipt."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import re


PARENT, NODE, PORT = "1103606", "f13r1n12", 32128
LABEL = "training-gemma-usgs-r120-runtime-admission-v1"
SCOPE, DISPATCH = "568be365fb2444a8bf0c4a30bb2b8d13", "67fd59f0a8574fdaab24324c5dea04cf"
OLD_SCOPE = "0c2fc133102849748fb3e97e4613889c"
BASELINE = "d5479f181bee07f3bf2fff658b1a05619536f030"
BASE_MANIFEST = "976f80922950ea78bee79aded9062fe40b21d3881cb219196ad7528ca5a9f210"
HISTORY = "eec018a8fa58c8f6803aef1aaf0a781964933e57a5dc9e89b6af635a0eaac679"
HOLDER, POST = "1103606.4081", "1103606.4167"
WORKER_RECEIPT = "08b2c4b6a066e08df37b4b77773dcee96e418375ae1cf31857b644cd279cddf7"
RELEASE_PROJECTION = "977ca204f44b980ea1e4806549245d010110339bc1711fa8297d16e8cac32e70"
LIMIT, WHOLE = 65, 2400


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def pin(raw):
    return dict(bytes=len(raw), SHA256=digest(raw))


def is_sha(value):
    return isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))


def stamp(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def histories(history_raw, release_raw):
    require(digest(history_raw) == HISTORY and 0 < len(release_raw) <= 16384,
            "ORIGINAL_ARCHIVE_OR_RELEASE_EXTENT")
    history, release = map(json.loads, (history_raw, release_raw))
    holder, post = release["holder_accounting"], release["post_accounting"]
    require(release["request_id"] == OLD_SCOPE and release["node"] == NODE and release["parent"] == PARENT
            and holder == history["holder_accounting"] and post == history["post_accounting"]
            and holder[0] == HOLDER and post[0] == POST and holder[2:4] == post[2:4] == ["COMPLETED", "0:0"]
            and tuple(holder[5:7]) == (NODE, "cpu=8,gres/dcu=8,mem=16G,node=1")
            and tuple(post[5:7]) == (NODE, "cpu=1,mem=256M,node=1")
            and stamp(holder[8]) <= stamp(post[7]) <= stamp(post[8])
            and history["status"] == "ORIGINAL_STANDBY_NATURAL_TERMINAL_AND_INDEPENDENT_POST_CONFIRMED"
            and history["worker_terminal_SHA256"] == WORKER_RECEIPT and history["accounting_SHA256"] == RELEASE_PROJECTION
            and release["independent"] is release["released"] is release["same_boot_and_pci"] is True
            and release["open_kfd_holders"] == release["scope_owned_processes_remaining"] == 0
            and release["UTC_accounting"]["TZ"] == "UTC"
            and release["UTC_accounting"]["rows"][-2:] == [holder, post]
            and all(is_sha(release[k]) for k in ("original_release_receipt_SHA256", "physical_projection_SHA256")),
            "EXACT_NATURAL_4081_AND_INDEPENDENT_4167_REQUIRED")
    return release


def authorize(history_raw, release_raw, natural_raw, approval_raw, publication, stable_pin, now):
    release = histories(history_raw, release_raw)
    require(all(0 < len(raw) <= 16384 for raw in (natural_raw, approval_raw)), "BOUNDED_CURRENT_EVIDENCE")
    natural, approval = map(json.loads, (natural_raw, approval_raw))
    terminal, post, fresh = (natural[key] for key in ("terminal", "post", "fresh"))
    require(natural["schema"] == "GEMMA_R120_NATURAL_RELEASE_AND_FRESH_N12_V1"
            and natural["scope"] == SCOPE and natural["node"] == NODE
            and terminal["step"] == HOLDER and terminal["accounting"] == release["holder_accounting"]
            and terminal["worker_terminal_verified"] is True and terminal["termination_kind"] == "NATURAL"
            and terminal["handoff_signal_delivered"] is False and terminal["worker_receipt_SHA256"] == WORKER_RECEIPT
            and post["step"] == POST and post["receipt_SHA256"] == release["original_release_receipt_SHA256"]
            and post["independently_verified"] is True and post["kfd_holders"] == post["owned_processes_remaining"] == 0,
            "ORIGINAL_TERMINAL_OR_POST_PROJECTION_CHANGED")
    ports = fresh["ports"]
    pci = post["pci"]
    step = re.fullmatch(r"1103606\.([0-9]+)", fresh["step"])
    observed = fresh["observed_epoch"]
    accounting = fresh["accounting"]
    require(type(accounting) is list and len(accounting) == 9 and accounting[0] == fresh["step"]
            and accounting[2:4] == ["COMPLETED","0:0"] and accounting[5] == NODE
            and fresh["accounting_TZ"] == "UTC"
            and set(accounting[6].split(",")) in ({"cpu=1","mem=256M","node=1"},{"cpu=1","mem=1G","node=1"})
            and accounting[4].isdecimal() and 0 <= int(accounting[4]) <= 65
            and stamp(release["post_accounting"][8]) <= stamp(accounting[7]) <= observed <= stamp(accounting[8])+1
            and stamp(accounting[8]) <= now and abs(stamp(accounting[8])-stamp(accounting[7])-int(accounting[4])) <= 1,
            "FRESH_ACTUAL_NUMBERED_CPU_ACCOUNTING_REQUIRED")
    require(type(observed) in (float,int) and math.isfinite(observed) and 0 <= now-observed <= LIMIT
            and step and int(step[1]) > 4167 and observed >= stamp(release["post_accounting"][8])
            and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", post["boot_id"])
            and fresh["boot_id"] == post["boot_id"] and len(pci) == len(set(pci)) == len(fresh["pci"]) == 8
            and set(pci) == set(fresh["pci"]) and all(re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) for x in pci)
            and fresh["node"] == NODE and fresh["all_relevant_fd_inventory_complete"] is True
            and fresh["kfd_holders"] == fresh["owned_processes_remaining"] == 0
            and fresh["current_workload_conflict_excluded"] is fresh["controller_and_queue_both_clear"] is True
            and set(ports) == {"tcp", "tcp6", "udp", "udp6"}
            and all(x["port"] == PORT and x["clear"] is True for x in ports.values())
            and is_sha(fresh["receipt_SHA256"]), "CURRENT_SAME_BOOT_8PCI_FD_PORT_AND_BOTH_VIEWS_REQUIRED")
    issued, expires = approval["issued_epoch"], approval["authorize_until"]
    require(all(type(x) in (int,float) and math.isfinite(x) for x in (issued, expires, approval["valid_until"]))
            and observed <= issued <= now <= expires <= min(observed+LIMIT,issued+LIMIT)
            and approval["valid_until"] >= now + WHOLE
            and (approval["request_id"],approval["dispatch_request_id"],approval["node"],approval["parent"],approval["master_port"])
                == (SCOPE,DISPATCH,NODE,PARENT,PORT)
            and re.fullmatch("[0-9a-f]{40}",publication) and publication != BASELINE
            and approval["publication"] == publication and approval["stable_source_manifest"] == stable_pin
            and approval["original_release_SHA256"] == digest(release_raw)
            and approval["natural_release_gate_SHA256"] == digest(natural_raw)
            and approval["old_holder_step"] == HOLDER and approval["latest_holder_accounting"] == release["holder_accounting"]
            and all(approval[k] is True for k in ("approved_eight_card_single_attempt","ordinary_exclusion",
                "coordinator_identity_verified","manual_adoption_authorized","prior_native_accounting_verified",
                "release_requires_terminal_independent_post","latest_explicit_holder_verified_terminal")),
            "FRESH_SOLE_RUNTIME_PUBLICATION_AUTHORITY_REQUIRED")
    rebound = encode(dict(release, request_id=SCOPE))
    runtime_approval = encode(dict(approval, latest_release_SHA256=digest(rebound),
                                   latest_holder_observation_SHA256=digest(rebound)))
    return runtime_approval, rebound
