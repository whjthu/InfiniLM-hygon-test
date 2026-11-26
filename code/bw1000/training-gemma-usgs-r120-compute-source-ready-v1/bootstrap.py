"""Same-step stdlib readback before the original Gemma zero-GRES device probe."""

import ast
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time


INTERVAL = .25
MAX_WAIT = 20.
PRE_RESERVE = 8.
REVIEW_RESERVE = 20.
AUTHORITY_MAX = 90.
SCRIPT = "devices.py"


class Bad(RuntimeError):
    pass


def need(ok, why):
    if not ok:
        raise Bad(why)


def pin(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def identity(meta):
    return tuple(getattr(meta, f"st_{name}") for name in ("dev", "ino", "size", "mtime_ns", "ctime_ns"))


def read(path, expected, cap):
    # lstat and descriptor identity avoid following a replaced or aliased file.
    expected_size = expected["bytes"]
    need(type(expected_size) is int and 0 < expected_size <= cap
         and re.fullmatch("[a-f0-9]{64}", expected["SHA256"]), "BAD_MEMBER_PIN")
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode), "MEMBER_NOT_REGULAR")
    need(before.st_size == expected_size, "MEMBER_EXTENT_MISMATCH")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(expected_size + 1)
        after = os.fstat(stream.fileno())
    # ENOENT during a read is a missing observation, never a silently ignored member.
    # Windows can report different ctime semantics for its path and descriptor APIs.
    path_before, path_after = identity(before), identity(path.lstat())
    fd_before, fd_after = identity(opened), identity(after)
    need(path_before == path_after and fd_before == fd_after
         and (path_before[:4] == fd_before[:4] if os.name == "nt" else path_before == fd_before),
         "MEMBER_CHANGED_DURING_READ")
    need(pin(raw) == expected, "MEMBER_SHA256_MISMATCH")
    return raw


def contract_projection(raw):
    tree = ast.parse(raw)
    wanted = {"LABEL", "TRAINING_REQUEST", "DISPATCH_REQUEST", "PARENT", "NODE", "PORT", "R"}
    values = {}
    for instruction in tree.body:
        if not isinstance(instruction, ast.Assign) or len(instruction.targets) != 1:
            continue
        dest, = instruction.targets
        if isinstance(dest, ast.Name) and dest.id == "R":
            v = instruction.value
            need(isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "Path"
                 and len(v.args) == 1 and not v.keywords, "CONTRACT_ROOT_LITERAL")
            values["R"] = ast.literal_eval(v.args[0])
        elif isinstance(dest, ast.Name) and dest.id in wanted:
            need(dest.id not in values, "DUPLICATE_CONTRACT_FIELD")
            values[dest.id] = ast.literal_eval(instruction.value)
        elif isinstance(dest, ast.Tuple) and all(isinstance(x, ast.Name) for x in dest.elts):
            keys = [x.id for x in dest.elts]
            if set(keys) & wanted:
                need(keys == ["PARENT","NODE","PORT"] and not (set(keys) & values.keys()), "CONTRACT_TUPLE_LITERAL")
                values.update(zip(keys,ast.literal_eval(instruction.value),strict=True))
        elif isinstance(dest, ast.Name) and dest.id == "ROOT":
            # This exact derived expression is also used by the original launcher.
            need(ast.unparse(instruction.value) == "R / 'results' / LABEL", "CONTRACT_OPERATION_ROOT")
            values["ROOT"] = True
    need(set(values) == wanted | {"ROOT"}, "CONTRACT_SOURCE_FIELDS")
    return values


def inspect(root, manifest, target, report, still_before):
    """Verify one coherent snapshot. No user module loads before success."""
    for part in [*reversed(root.parents),root]:
        need(stat.S_ISDIR(part.lstat().st_mode), "SOURCE_DIRECTORY_NOT_REAL")
    report["last_member"] = "SOURCE_FILES.json"
    manifest_raw = read(root / "SOURCE_FILES.json", manifest, 16384)
    listed = json.loads(manifest_raw)["files"]
    need(len(listed) == 32 and {"contract.py","CLAIM.json",SCRIPT} <= listed.keys()
         and all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",name) and name!="SOURCE_FILES.json" for name in listed),
         "EXACT_32_CANONICAL_SOURCE_NAMES")
    docs = {}
    for name, expected in listed.items():
        still_before()
        report["last_member"] = name
        raw = read(root/name, expected, 65536)
        report["checked_last_round"] += 1
        if name in {"CLAIM.json","contract.py"}:
            docs[name] = raw
    require_fields = contract_projection(docs["contract.py"])
    require_fields.pop("ROOT")
    claim = json.loads(docs["CLAIM.json"])
    need(require_fields == target and (claim["label"],claim["training_request_id"],claim["request_id"],claim["new_master_port"])
         == (target["LABEL"],target["TRAINING_REQUEST"],target["DISPATCH_REQUEST"],target["PORT"])
         and claim["root_GO"] is claim["GPU_GO"] is claim["training_GO"] is True
         and claim["Model_GO"] is claim["formal17_GO"] is False,"COMPUTE_CLAIM_AND_CONTRACT_SCOPE")
    need(read(root/"SOURCE_FILES.json",manifest,16384)==manifest_raw,"MANIFEST_CHANGED_DURING_VERIFICATION")
    still_before()


def budget(now, issued, pre_deadline):
    need(all(type(x) in (float,int) and math.isfinite(x) for x in (now,issued,pre_deadline))
         and issued<=now and now < pre_deadline <= now+60, "PRE_STEP_EPOCH_BOUNDS")
    residual = min(MAX_WAIT, pre_deadline-now-PRE_RESERVE, issued+AUTHORITY_MAX-now-REVIEW_RESERVE)
    need(residual > 0,"NO_BUDGET_FOR_PRE_AND_ORIGINAL_SECOND_ADMISSION")
    return residual


def wait_ready(root, manifest, target, issued, pre_deadline, status):
    start = time.monotonic()
    limit = start + budget(time.time(),issued,pre_deadline)

    def still_before():
        need(time.monotonic() < limit and pre_deadline-time.time() > PRE_RESERVE
             and issued+AUTHORITY_MAX-time.time() > REVIEW_RESERVE,"SOURCE_VISIBILITY_TIMEOUT")

    while True:
        still_before()
        status["rounds"] += 1
        status["checked_last_round"] = 0
        try:
            inspect(root,manifest,target,status,still_before)
            status["waited_ms"] = int((time.monotonic()-start)*1000)
            return
        except OSError as e:
            if e.errno != errno.ENOENT:
                status["errno"] = e.errno
                raise Bad("SOURCE_OS_FAILURE_NO_RETRY") from None
            status["enoent_rounds"] += 1
            remaining = limit-time.monotonic()
            if remaining > 0:
                time.sleep(min(INTERVAL,remaining))


def main(raw):
    setup = json.loads(raw)
    target = setup["target"]
    root = Path(setup["root"])
    status = dict(event="GEMMA_R120_COMPUTE_SOURCE_READY_V1", state="FAIL", rounds=0,
                  enoent_rounds=0, checked_last_round=0, last_member=None, errno=None,
                  request=target["TRAINING_REQUEST"],manifest=setup["manifest"],planned=32)
    try:
        need(root.is_absolute() and root.parts[-3:]==("results",target["LABEL"],"src")
             and root.parents[2].as_posix()==target["R"]
             and os.environ["SLURM_JOB_ID"]==target["PARENT"]
             and re.fullmatch(r"[0-9]+",os.environ["SLURM_STEP_ID"])
             and os.uname().nodename.split(".",1)[0]==target["NODE"],"WRONG_REAL_PRE_STEP_SCOPE")
        status["step"] = target["PARENT"]+"."+os.environ["SLURM_STEP_ID"]
        wait_ready(root,setup["manifest"],target,setup["issued"],setup["pre_deadline"],status)
        status["state"] = "READY_EXEC_ORIGINAL_DEVICES"
        print(json.dumps(status,sort_keys=True),flush=True)
        os.execv(sys.executable,[sys.executable,"-B",str(root/SCRIPT),"pre"])
    except Exception as error:
        status.update(state="FAIL",error_code=str(error) if type(error) is Bad else type(error).__name__)
        print(json.dumps(status,sort_keys=True),flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
