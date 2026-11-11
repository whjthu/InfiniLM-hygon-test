"""Run one bounded original A100 backward capture with its original environment."""

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base = Path("/home/whj/nvidia-baseline")
directory = base/"alignment-assets/hygon-trellis-lora-operands-v1"
assert not directory.exists()

def query(columns,kind="gpu"):
    raw = subprocess.check_output(["nvidia-smi","--query-"+kind+"="+columns,"--format=csv,noheader,nounits"],
                                  text=True,timeout=10)
    return [line.split(",") for line in raw.splitlines() if line.strip()]

cards = query("index,uuid,memory.used")
first, = [row for row in cards if row[0].strip()=="0"]
assert int(first[2].strip()) < 100
assert query("gpu_uuid,pid,used_gpu_memory","compute-apps") == []
directory.mkdir(mode=0o700)
for name, spec in FILES.items():
    assert name in ("capture_a100_lora.py","lora_projection_reference.py")
    data = base64.b64decode(spec["b64"],validate=True)
    assert hashlib.sha256(data).hexdigest()==spec["SHA256"]
    with (directory/name).open("xb") as stream:
        stream.write(data)

with (directory/"intent.json").open("x") as stream:
    json.dump({"claim_commit":CLAIM,"physical_card":0,"timeout_seconds":260,"GPU_optimizer_updates":0},stream)

env = dict(os.environ,CUDA_VISIBLE_DEVICES="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
           NVIDIA_BASELINE_DETERMINISTIC="1",CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
started = time.monotonic()
log = directory/"capture.log"
with log.open("xb") as stream:
    child = subprocess.Popen([sys.executable,"-B",str(directory/"capture_a100_lora.py"),str(directory/"capture")],
                             stdout=stream,stderr=subprocess.STDOUT,env=env)
    stat = Path(f"/proc/{child.pid}/stat").read_text()
    start_ticks = stat[stat.rfind(")")+2:].split()[19]
    try:
        rc = child.wait(timeout=260)
    except subprocess.TimeoutExpired:
        child.kill()
        rc = child.wait(timeout=10)

post_cards = query("index,uuid,memory.used")
post, = [row for row in post_cards if row[0].strip()=="0"]
assert post[1].strip()==first[1].strip()
raw = log.read_bytes()
assert len(raw)<131072
summary = directory/"capture/receipt.json"
result={"claim_commit":CLAIM,"PID":child.pid,"start_ticks":start_ticks,"returncode":rc,
        "wall_seconds":time.monotonic()-started,"log_SHA256":hashlib.sha256(raw).hexdigest(),
        "post_card_memory_MiB":post[2].strip(),"post_compute_process_count":len(query("pid","compute-apps")),
        "error_lines":[line[-200:] for line in raw.decode(errors="replace").splitlines()
                       if line.startswith(("AssertionError","RuntimeError","TypeError","KeyError","AttributeError"))][-5:]}
if summary.is_file():
    data = summary.read_bytes()
    assert len(data)<32768
    result["summary_SHA256"] = hashlib.sha256(data).hexdigest()
    result["summary"] = json.loads(data)
with (directory/"outcome.json").open("x") as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result))
