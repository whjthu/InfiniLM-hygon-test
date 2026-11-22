"""T03 engineering diagnostic scope and reusable input/optimizer boundaries."""
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time


REQUEST = "c01034b21e554afa909ccff5388b2a68"
LABEL = "training-internvl-native64-first2-hip-v5"
PARENT, NODE, PORT = "1087656", "a14r3n04", 29931
WHOLE, NATIVE, SLURM = 1800, 1440, 1500
HERE = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT/"results"
RUN = RESULTS/LABEL
FRAMEWORK_PARENT = RESULTS/"training-qwen-a68952c-swift-initial-CPU-v1"
FRAMEWORK = FRAMEWORK_PARENT/"framework"
COMPUTE = RESULTS/"training-internvl-modelroot17-compute-CPU-v1"
CPU360 = RESULTS/"training-internvl-target-native64-CPU-v5"
ASSETS = RESULTS/"training-vlm-original-assets-v1"
DATASET = ASSETS/"vlm_report_coco64.hip.jsonl"
REFERENCE = RESULTS/"training-internvl-rank00-reference-provision-v1/payload"
REF_INPUTS = REFERENCE/"reference/input-fingerprints.json"
INITIAL = REFERENCE/"initial/rank-00/initial.pt"
INITIAL_BYTES = 174821899
INITIAL_SHA = "2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f"
NAMED_SHA = "086ece33e01ff3454ba54f90cb0d30478ed71f172dedc70b1c076e2babd27137"
INPUT_SHA = "a52d848f052dfc0ccb42335a7ffa6a1982723fc2a82064fc612d7e1bfca1d959"
BASELINE_SHA = "d731f9889c22381514c2181c27f99ac86051930ab2e7792958db5b34e34f27e8"
LAUNCH_SHA = "7fefdc16f5519a998a94fdef703c2dd326b3c15e2e31d5a3735f6c4558269c67"
FINGERPRINT_SHA = "816ef694537273ac04d8f7d40df5694a16e8a869797d2d2f65664d21e6a0ea49"
CALLBACK = "internvl_first2_engineering"
MODEL = PurePosixPath("/internvl-model")
HEX = re.compile(r"[0-9a-f]{64}")
CAPTURE_NAMES = frozenset(["initial.pt"]
    + [f"forward-{i:02d}-{side}.pt" for i in range(1,9) for side in ("input","output")]
    + [f"update-{i:02d}-{side}.pt" for i in (1,2) for side in ("preclip","postclip","before-optimizer","after-optimizer")])


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, *, expected=None, extent=None, cap=2<<20):
    need(path.is_absolute() and path.resolve(strict=True)==path and not path.is_symlink(), "Canonical original path required")
    with path.open("rb",buffering=0) as handle:
        before=os.fstat(handle.fileno())
        raw=handle.read(cap+1)
        after=os.fstat(handle.fileno())
    keep=lambda value: (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
    need(stat.S_ISREG(before.st_mode) and keep(before)==keep(after)==keep(path.lstat())
         and 0<len(raw)==before.st_size<=cap and (extent is None or len(raw)==extent)
         and (expected is None or digest(raw)==expected),"Original metadata length or digest differs: "+path.name)
    return raw


def write_once(path, data):
    raw=encoded(data)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(raw)


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def seconds_left(epoch, limit, now=None):
    now=time.time() if now is None else now
    need(math.isfinite(epoch) and math.isfinite(now) and 0 <= now-epoch < limit <= WHOLE,
         "Original T03 epoch exhausted or differs")
    return limit-(now-epoch)


def input_roster(document):
    need(document["schema"]=="native_first_two_inputs_v1" and document["case"]=="T03.sft"
         and document["reference_request_id"]=="031e04f24efa429595364e6f0da1e5b6"
         and document["world_size"]==8 and document["accumulation"]==4
         and document["fingerprint_source_SHA256"]==FINGERPRINT_SHA
         and document["projection"]==["args","kwargs"] and len(document["rows"])==64,"Original native64 reference differs")
    result={str(rank):[] for rank in range(8)}
    for item in document["rows"]:
        rank, ordinal, value = item["rank"],item["micro_ordinal"],item["input_fingerprint_SHA256"]
        need(type(rank) is type(ordinal) is int and 0<=rank<8 and ordinal==len(result[str(rank)])+1
             and ordinal<=8 and item["global_step"]==(ordinal-1)//4 and HEX.fullmatch(value),
             "Incomplete or reordered reference microbatch denominator")
        result[str(rank)].append(value)
    need(set(map(len,result.values()))=={8},"All eight rank-specific input sequences required")
    return result


def verify_shape(binding, actual, target_signature):
    manifest=binding["trainable_manifest"]
    need(binding["spec"]["case"]=="T03.sft" and len(manifest)==504
         and sum(row["numel"] for row in manifest)==43646976
         and actual["trainable_manifest"]==manifest
         and actual["actual_trainable_tensors"]==504 and actual["actual_trainable_parameters"]==43646976,
         "Original .360 actual PEFT structure differs")
    need(target_signature(actual["target_modules"])==target_signature(binding["target_modules"]),
         "Original .360 PEFT target selector differs")
    return manifest


def finish_step(step, *, updates, inputs, outputs, clips, names):
    need(step==2 and updates==2 and inputs==outputs==8 and clips==2 and set(names)==CAPTURE_NAMES
         and len(names)==25,"Incomplete first-two optimizer updates or capture slots")


def launch_recipe(builder, original, *, package=HERE, run=RUN):
    need(digest(original)==LAUNCH_SHA,"Exact original A100 Swift recipe required")
    launch=builder("T03.sft",original,PurePosixPath(str(FRAMEWORK/"vlm/runtime/swift_entry.py")),
                   PurePosixPath("/t03-recipe-model"),PurePosixPath(str(DATASET)),
                   PurePosixPath(str(ASSETS/"plugins")),PurePosixPath(str(run/"train")),PORT,
                   diagnostic_plugin=PurePosixPath(str(package/"callback.py")),diagnostic_callback=CALLBACK,
                   diagnostic_callback_position="before")
    argv=launch["command"]
    expected={"--model":"/t03-recipe-model/InternVL3_5-8B-HF","--dataset":str(DATASET),
              "--max_steps":"25","--gradient_accumulation_steps":"4","--gradient_checkpointing":"true",
              "--per_device_train_batch_size":"1","--attn_impl":"sdpa","--max_length":"4096"}
    need(all(argv.count(name)==1 and argv[argv.index(name)+1]==value for name,value in expected.items())
         and launch["environment_overrides"]["NPROC_PER_NODE"]=="8"
         and launch["environment_overrides"]["MAX_STEPS"]=="25","Original full COCO64 eight-rank training schedule differs")
    argv[argv.index("--model")+1]=str(MODEL)
    launch["environment_overrides"]["MODELS_DIR"]=str(MODEL.parent)
    return launch
