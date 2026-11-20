"""Exact published packet -> one foreground controller in the existing SSH call."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
LABEL="training-qwen-usgs-exposure25-resume20-hip-v1"
REQUEST="f572fda6784140cbbdcae7bb91a916b9"


def write(path,raw):
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def main(payload):
    os.umask(0o077)
    epoch,deadline=float(payload["epoch"]),float(payload["deadline"])
    assert all(math.isfinite(x) for x in (epoch,deadline)) and deadline-epoch==2400
    assert 0<epoch<=time.time()<deadline-30
    source={name:base64.b64decode(raw,validate=True) for name,raw in payload["files"].items()}
    assert sum(map(len,source.values()))<=1024**2
    claim=json.loads(source["CLAIM.json"])
    assert payload["request"]==REQUEST
    assert (claim["request"],claim["label"],claim["device_count"],claim["optimizer_updates"])==(REQUEST,LABEL,8,30)
    assert all(claim[k] is True for k in ("root_GO","GPU_GO","training_GO"))
    assert all(claim[k] is False for k in ("Model_GO","formal_training_GO","effect_GO","performance_GO"))
    mf=source["SOURCE_FILES.json"]
    assert hashlib.sha256(mf).hexdigest()==payload["source_manifest_SHA256"]
    assert hashlib.sha256(source["CLAIM.json"]).hexdigest()==payload["claim_SHA256"]
    pins=json.loads(mf)["files"]
    assert set(source)==set(pins)|{"SOURCE_FILES.json"}
    for name,pin in pins.items():
        assert Path(name).name==name and len(source[name])==pin["bytes"]
        assert hashlib.sha256(source[name]).hexdigest()==pin["SHA256"]
    root=ROOT/LABEL
    root.mkdir(mode=0o700)
    area=root/"src"
    area.mkdir(mode=0o700)
    for name,raw in source.items():
        write(area/name,raw)
    intent=dict(request_id=REQUEST,publication=payload["publication"],epoch=epoch,deadline=deadline,
                source_manifest_SHA256=payload["source_manifest_SHA256"],claim_SHA256=payload["claim_SHA256"],
                node="f13r1n10",parent="1103606",arm="continuous",planned_optimizer_updates=30,
                planned_updates_by_arm={"continuous":25,"resumed":5},replay_allowed=False)
    write(root/"intent.json",json.dumps(intent,allow_nan=False).encode())
    child=None
    rc=None
    error_type=None
    try:
        with (root/"controller.log").open("xb") as log:
            child=subprocess.Popen([sys.executable,"-B","-u",str(area/"controller.py"),"host"],
                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            fields=Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ",1)[1].split()
            owner=dict(pid=child.pid,startticks=int(fields[19]),
                       host=os.uname().nodename,boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
            write(root/"controller-owner.json",json.dumps(owner).encode())
            rc=child.wait(timeout=max(.1,deadline-time.time()-8))
    except BaseException as exc:
        error_type=type(exc).__name__
    finally:
        if child is not None and child.poll() is None:
            # This Popen handle and saved lifetime belong to this exact new scope.
            # No foreign process or old run is selected.
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
    outcome=root/"controller-outcome.json"
    passed=rc==0 and error_type is None and outcome.is_file() and json.loads(outcome.read_bytes())["passed"] is True
    receipt=dict(request=REQUEST,epoch=epoch,deadline=deadline,controller_rc=rc,error_type=error_type,
                 normal_held_controller_reaped=child is not None and child.poll() is not None,
                 passed=passed,transport_or_failure_does_not_imply_GPU_terminal=True,
                 original_namespace_preserved=True,replay_allowed=False)
    write(root/"receiver-outcome.json",json.dumps(receipt,allow_nan=False).encode())
    print(json.dumps(dict(request=REQUEST,passed=passed,execution_state="TERMINAL",planned_updates=30,
                          actual_model_or_GPU_qualification_not_implied=True)),flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main(PAYLOAD))
