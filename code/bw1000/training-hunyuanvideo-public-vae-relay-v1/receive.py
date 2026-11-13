"""Receive an exact public VAE without modifying the Transformer transfer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

def timeout(*_):
    raise TimeoutError("public VAE relay deadline")

def save(path, value):
    with path.open("x") as stream:
        json.dump(value,stream)
        stream.flush()
        os.fsync(stream.fileno())

root=Path(PLAN["root"])
assert len(CONFIG)==PLAN["config_bytes"] and hashlib.sha256(CONFIG).hexdigest()==PLAN["config_SHA256"]
assert 0 < DEADLINE-time.time() <= PLAN["seconds"] and PLAN["offset"]==0
assert shutil.disk_usage(root.parent).free > PLAN["total"]+1024**3
signal.signal(signal.SIGALRM,timeout)
signal.alarm(max(1,int(DEADLINE-time.time())))
root.mkdir(mode=0o700)
base=dict(claim=CLAIM,request_id=REQUEST,deadline=DEADLINE)
save(root/"intent.json",base)
position=0
digest=hashlib.sha256()
try:
    stage=root/"vae.incomplete"
    stage.mkdir(mode=0o700)
    with (stage/"config.json").open("xb") as config:
        config.write(CONFIG)
        config.flush()
        os.fsync(config.fileno())
    with (stage/"diffusion_pytorch_model.safetensors").open("xb") as stream:
        print(json.dumps(dict(**base,ready=True,offset=0)),flush=True)
        checkpoint=64*1024**2
        while position<PLAN["total"]:
            block=sys.stdin.buffer.read(min(1024**2,PLAN["total"]-position))
            if not block:
                raise EOFError("public VAE ended early")
            stream.write(block)
            digest.update(block)
            position+=len(block)
            if position>=checkpoint:
                stream.flush()
                os.fsync(stream.fileno())
                temp=root/"progress.tmp"
                save(temp,dict(**base,offset=position,prefix_SHA256=digest.hexdigest()))
                temp.replace(root/"progress.json")
                checkpoint=position+64*1024**2
        stream.flush()
        os.fsync(stream.fileno())
    if sys.stdin.buffer.read(1) or digest.hexdigest()!=PLAN["full_SHA256"]:
        raise ValueError("whole-file verification failed")
    stage.rename(root/"vae")
    outcome=dict(**base,passed=True,offset=position,whole_SHA256=digest.hexdigest(),
                 config_SHA256=PLAN["config_SHA256"])
except Exception as exc:
    outcome=dict(**base,passed=False,offset=position,error_type=type(exc).__name__)
save(root/"outcome.json",outcome)
print(json.dumps(outcome),flush=True)
raise SystemExit(0 if outcome["passed"] else 1)
