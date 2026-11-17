"""Frozen CPU-only true-main run and cache write resources; no GPU dispatch path."""
from pathlib import Path,PurePosixPath
import hashlib
import json
import math
ROOT=PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS=ROOT/"results"
NAMESPACE="training-qwen-treasury-cache-host-main-candidate-v1"
LABEL="training-qwen-treasury-cache-host-main-CPU-v1"
RUN=Path(str(RESULTS/NAMESPACE))
SRC=RUN/"source"
CPU=RUN/"cpu"
PREP=RESULTS/"training-qwen-treasury-explicit-dev-CPU-v1"
MEDIA=RESULTS/"training-qwen-treasury-dev-native-CPU-v1/dev/media"
MODEL=RESULTS/"training-qwen-original-public-model-v1/model"
IMAGE=ROOT/"containers/dtk26-sandbox-r1"
ARMS={"base":dict(parent="1087656",node="a01r4n14"),"old_domain":dict(parent="1087656",node="a01r4n16"),"domain":dict(parent="1103606",node="f13r1n10")}
ADAPTERS=[str(RESULTS/"training-qwen-treasury-sft-v3/continuous/train/checkpoint-5"),str(RESULTS/"training-qwen-treasury-explicit-sft-v1/continuous/train/checkpoint-5")]
FLAGS=["/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--contain","--rocm","--no-home","--pwd","/"]
COLS="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"
CPU330_PINS={"intent.json":"6c0ea879c9d884f3938482b8ee613cd51233182420f742de5e37fd35f36d1e0c","outcome.json":"e10122997c1418eed20431f56419b9242e90428f16f52d1372fe231ee3600ee0","step.log":"5fc38b966f4c121a1f1ad961621109d35b605a5e0b4df2adb70eef9bd26a31b8","source/SOURCE_FILES.json":"c0efd831d4882f1b4dc7fbc1efed49a96fb3093abdefd56cf5d371cb9f292e9a","inventory.json":"07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"}


def require(ok,why):
    if not ok:raise ValueError(why)


def pin(path,limit=2*1024**2):
    require(path.is_file() and not path.is_symlink() and 0<=path.stat().st_size<=limit,"unsafe source/small receipt")
    b=path.read_bytes();return dict(bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())


def read(path):return json.loads(path.read_bytes())


def remaining(deadline,mono_end,now,mono,cap):
    require(all(math.isfinite(v) for v in (deadline,mono_end,now,mono,cap)) and cap>0,"invalid original deadline")
    left=min(deadline-now,mono_end-mono);require(left>0,"original CPU epoch expired");return min(cap,left)


def stage_limits(deadline,mono_end,now,mono):
    left=remaining(deadline,mono_end,now,mono,900)
    slurm=60*(int(min(180,left-60))//60)
    require(slurm>=120,"not enough original CPU epoch for main and accounting")
    work=min(150,slurm-30)
    return dict(Slurm_seconds=slurm,work_seconds=work,kill_grace_seconds=10,client_seconds=min(slurm+10,left-40))


def runtime_bind_source(host):
    path=Path(host)
    if path==Path("/opt/hyhal"):
        target=Path("/usr/local/hyhal");require(path.resolve(strict=True)==target and target.is_dir() and not target.is_symlink(),"hyhal system alias differs");return str(target)
    require(path.is_dir() and not path.is_symlink(),"required exact runtime bind missing or linked");return str(path)


def declared_binds(arm):
    own=str(RESULTS/NAMESPACE/"arms"/arm);src=str(RESULTS/NAMESPACE/"source")
    binds=[[own,own,"rw"],[src,src,"ro"],[str(MODEL),str(MODEL),"ro"],[str(PREP),str(PREP),"ro"],[str(MEDIA),str(PREP/"dev/media"),"ro"],
       [str(RESULTS/"training-vlm-swift-offline-runtime-v1"),"/swift-runtime","ro"],[str(RESULTS/"training-qwen-vision014-CPU-v2/vision"),"/vision","ro"],
       [str(RESULTS/"training-minicpm-hygon-processor-r2/cpu/runtime-site"),"/minicpm-transformers","ro"],["/opt/hyhal","/opt/hyhal","ro"]]
    # HOST API test constructs both real AdapterRequests; both immutable checkpoint dirs are RO for all three CPU arms.
    binds.extend([[path,path,"ro"] for path in ADAPTERS]);return binds


def verify_dependencies(accounting):
    p=Path(str(PREP))
    for name,digest in CPU330_PINS.items():require(pin(p/name)["SHA256"]==digest,"original CPU330 input prerequisite differs")
    original=read(p/"outcome.json");row=accounting("1087656","training-qwen-treasury-explicit-dev-CPU-v1")
    require(row==original["accounting"]["row"] and row[:4]==["1087656.330","training-qwen-treasury-explicit-dev-CPU-v1","COMPLETED","0:0"]
        and original["passed"] is True and original["metrics"]["actual_GPU_cards"]==original["metrics"]["actual_generations"]==0
        and all(original["metrics"]["arms"][k]["rows"]==108 for k in ("visual","text_only")),"actual CPU330 native input reference differs")
    mf=read(p/"source/SOURCE_FILES.json")["files"];require(len(mf)+1==13,"CPU330 source13 differs")
    for name,info in mf.items():require(Path(name).name==name and pin(p/"source"/name)==info,"CPU330 source member differs")
