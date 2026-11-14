import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

epoch = time.time()
require = lambda ok: None if ok else (_ for _ in ()).throw(ValueError("published run package differs"))
require(re.fullmatch("[0-9a-f]{40}", CLAIM) and re.fullmatch("[0-9a-f]{32}", REQUEST))
data = {}
for name, record in FILES.items():
    require(Path(name).name==name)
    raw = base64.b64decode(record["b64"],validate=True)
    require(len(raw)==record["bytes"] and hashlib.sha256(raw).hexdigest()==record["SHA256"])
    data[name] = raw
inventory = json.loads(data["SOURCE_FILES.json"])["files"]
require(set(inventory)==set(data)-{"SOURCE_FILES.json"})
for name,row in inventory.items():
    require(row == {key:FILES[name][key] for key in ("bytes","SHA256")})
root = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
run = root/"GLM474-selected-value-GPU-v1"
run.mkdir(mode=0o700)
with (run/"intent.json").open("x") as stream:
    json.dump(dict(claim=CLAIM,request_id=REQUEST,epoch=epoch,deadline=epoch+360,
                   PID=os.getpid(),source_files=inventory),stream)
    stream.flush()
    os.fsync(stream.fileno())
src = run/"src"
src.mkdir(mode=0o700)
for name,raw in data.items():
    with (src/name).open("xb") as stream:
        stream.write(raw)
sys.path.insert(0,str(src))
from launch import execute
outcome = execute(CLAIM,REQUEST,epoch)
print(json.dumps(dict(claim=CLAIM,request_id=REQUEST,accounting=outcome["stages"],
                      passed=outcome["passed"],error=outcome.get("error"),
                      post_release_verified=outcome.get("post_release_verified",False))),
      flush=True)
