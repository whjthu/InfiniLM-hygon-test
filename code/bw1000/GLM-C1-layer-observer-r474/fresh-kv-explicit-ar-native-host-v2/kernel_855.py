"""Bind the independently verified fresh-KV kernel on its original compute mount."""

from hashlib import sha256
import json
from pathlib import Path
import runpy


LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v2"
REQUEST = "d0b051b52122d2858ca72058d217d915"
OBJECT_SHA = "f25802c55db08aa3f4bdb0c8aa8f88e89bc5b88420ecbe078cae332db0ed27c6"
WORKER_SHA = "5c54e46f3762df2a9a93e28ca000b1f52254a7f8fac6069b86fe0f4407d37632"
DEPS_SHA = "d866bf966908555973deb0defd70451393416e78dbebdc219e97948f4d3d4a26"
INTENT_SHA = "fd641128b7661b1fe256ad4d7954e50af37b367c648b09337b799f1539e812dc"
PUBLIC = "results/bw1000/2026-11-22/GLM474-fresh-kv-pack-v2-original855-compute-reader-terminal.json"
PUBLIC_SHA = "b113149ebf3e4df70a9f34ae5ca7c7b609e7cf708ab2753b40eeaa6cff89cd6b"
PACK_CLAIM_SHA = "95e7cb7accf7cdc14ab028d58f2d78b460dd6a06a2d1e541439380193f3bafb4"
PACK_HOST_SHA = "f6a276edc2fe58c59e57072ea723156b0b432846327ac47ca8f5f34197830901"
STAGE_SHA = "3089672d48d7aaad128b53c8bdb2f8014d334cf94d846c9f12cea7e2cee8bf20"
PROOF = {"receipt_SHA256": WORKER_SHA, "object_SHA256": OBJECT_SHA,
         "fresh_M_SHA256": DEPS_SHA, "actual_MD_SHA256": DEPS_SHA,
         "terminal_row_SHA256": "027f4860f76190c706636006bfa561097e519c7bc40b8eb2ba3f1587813ada5c",
         "planned_device_TUs": 1, "actual_device_TUs": 1}
ROW = ["1103606.855", LABEL, "COMPLETED", "0:0", "9", "f12r2n13", "cpu=4,mem=8G,node=1"]
COLUMNS = ["JobIDRaw", "JobName%100", "State", "ExitCode", "ElapsedRaw", "NodeList", "AllocTRES"]


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, pin, size=None, limit=2 * 1024**2):
    path = Path(path)
    need(path.resolve(strict=True) == path, "frozen artifact must be at its own canonical path")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    need(0 < len(raw) <= limit and (size is None or len(raw) == size) and digest(raw) == pin,
         f"frozen artifact bytes differ: {path.name}")
    return raw


def archive(raw):
    need(len(raw) == 3226 and digest(raw) == PUBLIC_SHA, "independent .855 public reference differs")
    data = json.loads(raw)
    observed = data["independent_compute_reader"]
    result = observed["result"]
    history = data["actual_accounting"]
    need(data["request"] == REQUEST and data["state"] == "TARGET_OBJECT_VERIFIED_ON_ORIGINAL_COMPUTE_NODE"
         and data["original_controller_state"] == "TERMINAL_UNVERIFIED_LOGIN_MOUNT_NO_RECEIPT"
         and data["accounting_columns"] == COLUMNS and data["actual_GPU_steps"] == 0
         and observed["reader_rc"] == 0 and observed["failure"] is None and observed["original"] == ROW
         and observed["original_intent_SHA256"] == INTENT_SHA
         and result == dict(passed=True, original=ROW, result=PROOF, independent_read_only=True)
         and [row[0] for row in history] == ["1103606.854", "1103606.855", "1103606.856", "1103606.857"]
         and data["final_cleanup"]["observed"] is True and data["final_cleanup"]["step"] == "1103606.857"
         and digest(json.dumps(ROW, separators=(",", ":")).encode()) == PROOF["terminal_row_SHA256"],
         "actual .855/.856/.857 evidence scope differs")
    return data


def kernel_intent(root):
    path = root / "results" / (LABEL + "-controller") / "intent.json"
    raw = read(path, INTENT_SHA, limit=16384)
    intent = json.loads(raw)
    epoch = intent["epoch"]
    need(intent["request"] == REQUEST and intent["parent"] == "1103606" and intent["node"] == "f12r2n13"
         and intent["source_manifest_SHA256"] == STAGE_SHA and intent["claim_SHA256"] == PACK_CLAIM_SHA
         and type(epoch) is float and 0 < epoch < float("inf") and intent["deadline"] == epoch + 300
         and intent["replay_allowed"] is False, "actual .855 intent or epoch differs")
    return dict(intent, actual_intent_SHA256=digest(raw))


def login_proof(root, reviewed, read_accounting):
    """Read the exact archive and current original rows without touching the hidden login object."""
    data = archive(read(reviewed / PUBLIC, PUBLIC_SHA, size=3226))
    intent = kernel_intent(root)
    rows = read_accounting()
    for archived in data["actual_accounting"][1:]:
        need([row for row in rows if row[0] == archived[0]] == [archived], "current original accounting differs")
    return dict(PROOF, request=REQUEST, state="COMPILE_VERIFIED", independent_observation=True,
                actual_GPU_steps=0, row=ROW, intent_SHA256=intent["actual_intent_SHA256"],
                public_result_SHA256=PUBLIC_SHA, independent_reader_step="1103606.856",
                independent_cleanup_step="1103606.857")


def verify_compute(root, proof):
    """Use the already frozen v2 passive reader in the actual compute/container mount."""
    need(proof["request"] == REQUEST and proof["row"] == ROW and proof["intent_SHA256"] == INTENT_SHA
         and proof["public_result_SHA256"] == PUBLIC_SHA
         and {key: proof[key] for key in PROOF} == PROOF, "proposed predecessor proof does not bind .855")
    intent = kernel_intent(root)
    host = (root / "incoming" / (LABEL + "-reviewed-checkout") /
            "code/bw1000/GLM-C1-layer-observer-r474/fresh-kv-pack-host-compile-v2/host_once.py")
    read(host, PACK_HOST_SHA, size=8116)
    pack = runpy.run_path(str(host))
    claim, plan, original = pack["package"](PACK_CLAIM_SHA)
    need(claim["request"] == REQUEST and claim["stage_manifest_SHA256"] == STAGE_SHA,
         "original target stage differs")
    helpers, _, _ = pack["prepare_controller"](claim, plan, original)
    result = helpers["terminal_pass"](root / "candidates" / LABEL, ROW, intent["epoch"], plan)
    need(result == PROOF, "compute-mounted original relocatable or M/MD/log evidence changed")
    return result
