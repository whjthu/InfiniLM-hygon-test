"""Run the patched native model-authority entry on one separately placed CPU step."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import sys
import time
from types import ModuleType


BASE=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
OLD_PREPARE=BASE/"training-gemma-usgs-r39-two-updates-native-v1/src/prepare.py"
OLD_PREPARE_SHA="4c09188c713bb5b25e7e0597f1e95b78a2bebdfa10e81da19fd3bc4669403b76"
ASSETS_SHA="2cd0c3a463d2efeebd6d41fb05308f87da998380b9d305ee5b3c07825a90626b"
CPU_WORK=120


def need(ok,why):
    if not ok:
        raise ValueError(why)


def module(name,path,expected):
    meta=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(meta.st_mode) and 0<meta.st_size<=65536,
         "R40_EXACT_CANONICAL_SOURCE_PATH")
    with path.open("rb") as stream:
        raw=stream.read(65537)
    need(len(raw)==meta.st_size and sha256(raw).hexdigest()==expected and name not in sys.modules,
         "R40_EXACT_SOURCE_BYTES_BEFORE_EXEC")
    value=ModuleType(name)
    value.__file__=str(path)
    sys.modules[name]=value
    try:
        exec(compile(raw,str(path),"exec",dont_inherit=True),value.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return value


def run(epoch):
    env=os.environ
    numeric=env.get("SLURM_STEP_ID","")
    host=socket.gethostname().split(".")[0]
    gid=env.get("SLURM_JOB_ID")
    cgroup=Path("/proc/self/cgroup").read_text("ascii")
    need(gid=="1103606" and re.fullmatch(r"[0-9]+",numeric) and host=="f13r1n12"
         and any({"job_1103606","step_"+numeric}<=set(line.split(":",2)[-1].split("/")) for line in cgroup.splitlines())
         and env.get("SLURM_CPUS_PER_TASK")=="4" and len(os.sched_getaffinity(0))>=4,
         "R40_GEMMA_EXACT_NUMERIC_CPU_STEP")
    need(env.get("SLURM_STEP_GPUS","") in ("","(null)","none","N/A")
         and all(env.get(k,"0") in ("","0","(null)") for k in ("SLURM_GPUS_ON_NODE","SLURM_GPUS_PER_TASK"))
         and not any(name in env.get("SLURM_TRES_PER_TASK","").lower() for name in ("gres/","gpu","dcu"))
         and all(env.get(k)=="-1" for k in ("HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES","CUDA_VISIBLE_DEVICES")),
         "R40_GEMMA_ZERO_VISIBLE_DEVICE")
    begin=time.time()
    need(math.isfinite(epoch) and epoch<=begin<epoch+CPU_WORK,"R40_GEMMA_EXACT_120S_CPU_EPOCH")
    until=time.monotonic()+epoch+CPU_WORK-begin
    def check():
        need(time.time()<epoch+CPU_WORK and time.monotonic()<until,"R40_GEMMA_CPU_DEADLINE")
    def expire(*_):
        raise TimeoutError("R40_GEMMA_CPU_DEADLINE")
    signal.signal(signal.SIGALRM,expire)
    signal.setitimer(signal.ITIMER_REAL,max(.001,until-time.monotonic()))
    try:
        script=Path(__file__).absolute()
        need(script.resolve(strict=True)==script and sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode,
             "R40_GEMMA_ISOLATED_CPU_CANDIDATE")
        module("prepare",OLD_PREPARE,OLD_PREPARE_SHA)
        patched=module("_r40_native_assets_candidate",script.with_name("assets.py"),ASSETS_SHA)
        seen=patched.from_target(check)
        return {"step":gid+"."+numeric,"node":host,"epoch":epoch,"CPU":4,"target_GPU":0,"original_3417_replayed":False,
                "historic_full_file_shas_bound":True,"native_same_entry":True,"members":len(seen),
                "new_shard_payload_bytes_rehashed":0,"actual_model_loaded":False,"optimizer_updates":0,
                "current_assets_identity_SHA256":sha256(json.dumps(seen,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
                "actual_step_TRES_independent_confirmation_required":True}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch",required=True,type=float)
    args=parser.parse_args()
    try:
        proof=run(args.epoch)
        print(json.dumps({"passed":True,**proof},sort_keys=True))
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
        reason=str(error) if type(error) is ValueError and re.fullmatch(r"R(?:39|40)_[A-Z0-9_]{1,95}",str(error)) else None
        print(json.dumps({"passed":False,"error_class":type(error).__name__,"predicate":reason,"actual_training":0},sort_keys=True))
        raise SystemExit(1) from None
