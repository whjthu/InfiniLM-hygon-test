"""Pinned .220/.1231 runtime, original data stage and fresh placement inputs."""

from datetime import datetime, timezone
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import socket
import stat


PROJECT=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS=PROJECT/"results"
STAGE=RESULTS/"training-qwen-a68952c-swift-initial-CPU-v1/source"
FRAMEWORK=STAGE.parent/"framework"
MODEL=RESULTS/"training-qwen-original-public-model-v1/model"
INITIAL=RESULTS/"training-qwen-initial-trainables-v1/initial.pt"
SWIFT=RESULTS/"training-vlm-swift-offline-runtime-v1"
VISION=RESULTS/"training-qwen-vision014-CPU-v2/vision"
PROCESSOR=RESULTS/"training-minicpm-hygon-processor-r2/cpu/runtime-site"
INPUT=RESULTS/"training-qwen-textvqa-native-input-CPU-04a2af4f4b5b4e44bf5411cf32606bb9"
APPTAINER="/public/software/apps/apptainer/1.3.4/bin/apptainer"
SANDBOX=PROJECT/"containers/dtk26-sandbox-r1"
PYHOST="/usr/local/python3.12/bin/python3"
PREFIX="qwen-textvqa-first-two-"
NATIVE,GPU,WHOLE,INSIDE=1320,1500,1720,1380
CPUS,MEMORY,DCU=64,224000,8
AUTHOR_MAX_AGE=60
SHA64=re.compile(r"[0-9a-f]{64}")
SHA40=re.compile(r"[0-9a-f]{40}")
REQUEST=re.compile(r"[0-9a-f]{32}")
CPU_STEP_GRES="none"
OLD_SCOPE_SHA="6aa67a5a1cecb0376a53bf7312e3cab42f93e2fa34ed53212ad95d3412fe4ee5"
OLD_SNAPSHOT_SHA="98b45c22130c1e39a14ebb7393d85f2a763ffc2417de35a4f57fa303f3f05b8a"
OLD_NATIVE_SHA="56eda87754f64bd46b88f4b8a7236475294f52ded464984f6f8f1e92d59af443"
ASSETS_SHA="3abef5782032d9c6fe41dcc876c27320b0ff9314771259b4b5eec563b4a2599f"
FRAMEWORK_SHA="e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"
SAMPLER_SHA="d8809334da1398d0fcd73f03b02f69bc88f8c91d9037a678fbdc01ade2d6449c"
TERMINAL1231_SHA="9a2f85d87bb7245686c2bf4fd77c1d1fc882c9149a614daa746bbed0d662d745"
VISION_MANIFEST_SHA="75ee6a875b49a53c749cf7091de9c070c3276ef488a1af6ba3d2dab0571d551b"
FRAMEWORK_CACHE_ALLOWLIST=frozenset({
    "tools/__pycache__/hygon_vlm_recipe.cpython-312.pyc",
    "vlm/runtime/__pycache__/checkpointing_lifecycle.cpython-312.pyc",
    "vlm/runtime/__pycache__/native_recipe_contract.cpython-312.pyc",
})
CACHE_PREFIX_NAME="never-populated-python-cache"
CLOSED_ASSET_REQUESTS=frozenset({"19a0e8677692481ead4c75f76a0361b9","e39ca4817cb7451aa66092d35f0c6b03"})
EMPTY_SHA256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
VISION_REQUESTED="qwen_vl_utils-0.0.14.dist-info/REQUESTED"
PROVIDERS={
    SWIFT/"site/swift/cli/main.py":("d84a44f9911c01c9af83db19baf2068ad39a8ec926f6e2c58f6f69779bfdad68",None),
    SWIFT/"site/swift/cli/sft.py":("97162d10eb5b82cdfadf1ec30a0eac50cc0bc9646140b8342806df9e8e6e07f0",None),
    PROCESSOR/"transformers/modeling_rope_utils.py":("f370d8588169c07fc5245ddb83d9a29282f670f98d99a4d7fa814a777649f3d5",58066),
    PROCESSOR/"transformers/integrations/sdpa_attention.py":("87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564",4851),
    PROCESSOR/"transformers/models/qwen3_vl/modeling_qwen3_vl.py":("f39b3621fea3f757a01d2e00ebd3a2be3516b7df3048d862a365988aba9fa994",74723),
    PROCESSOR/"transformers/modeling_utils.py":("98a17fb9c5a04127f770da4d686932a5c5321cfb25833c057df8ab9f9577ce43",267972),
    VISION.parent/"outcome.json":("235851389a1a6a39c1cf5762c89af2ce77802ae7ed45b45365f5863524a983f3",None)
}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(obj):
    return (json.dumps(obj,sort_keys=True,allow_nan=False)+"\n").encode()


def digest(raw):
    return sha256(raw).hexdigest()


def stamp(path):
    now=path.lstat()
    need(stat.S_ISREG(now.st_mode) and path.resolve(strict=True)==path.absolute(),"EXACT_UNLINKED_ASSET_FILE")
    return [now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns,stat.S_IMODE(now.st_mode)]


def small(path, expected=None, limit=524288):
    before=stamp(path)
    need(0<before[2]<=limit,"EXACT_SMALL_FILE_EXTENT")
    raw=path.read_bytes()
    need(before==stamp(path) and len(raw)==before[2] and (expected is None or digest(raw)==expected),
         "EXACT_SMALL_FILE_IDENTITY")
    return raw


def stream_check(path, expected, extent=None, mode=None):
    before=stamp(path)
    exact_empty=(type(extent) is int and extent==0 and expected==EMPTY_SHA256 and path==VISION/VISION_REQUESTED)
    need((before[2]>0 or exact_empty) and (extent is None or before[2]==extent)
         and (mode is None or before[-1]==mode),"EXACT_ASSET_EXTENT_OR_MODE")
    with path.open("rb") as source:
        opened=os.fstat(source.fileno())
        need([opened.st_dev,opened.st_ino,opened.st_size]==before[:3],"OPENED_ASSET_IDENTITY_CHANGED")
        actual=file_digest(source,"sha256").hexdigest()
    need(actual==expected and before==stamp(path),"EXACT_ASSET_BYTES")
    return before


def save(path, item):
    with path.open("xb") as out:
        out.write(encode(item))
        out.flush()
        os.fsync(out.fileno())
    if os.name=="posix":
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def cache_prefix(root):
    value=root/CACHE_PREFIX_NAME
    need(root.is_absolute() and not value.exists() and not value.is_symlink(),
         "EXACT_UNUSED_PRIVATE_PYCACHE_PREFIX")
    return value


def placement(item, proof, now):
    req=item["request"]
    need(REQUEST.fullmatch(req) and re.fullmatch(r"(?:1103606|1194900)",item["parent"])
         and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}",item["node"])
         and SHA40.fullmatch(item["publication"]) and 1024<=item["port"]<=65535
         and (item["DCU"],item["CPU"],item["memory_MiB"],item["whole_s"])==(DCU,CPUS,MEMORY,WHOLE)
         and item["GPU_GO"] is True and item["exclusive_node"] is True and item["replay_allowed"] is False
         and req not in CLOSED_ASSET_REQUESTS
         and req not in {"04a2af4f4b5b4e44bf5411cf32606bb9","ab25db1eebbc433ea36dbd8266e9c4d4",
                         "8286a02a2f514fe2ac19636220dbbe33","2a13708f221c4a75890de2912c3d0810"},
         "EXACT_NEW_OWNER_PLACEMENT_REQUIRED")
    keys=("request","parent","node","publication")
    need(all(proof[key]==item[key] for key in keys) and SHA64.fullmatch(proof["coordinator_evidence_SHA256"])
         and proof["exclusive_no_other_dispatch"] is True
         and proof["legacy_bridge_state"] in {"VERIFIED_NO_SEND","VERIFIED_TERMINAL_AND_RELEASED","VERIFIED_NO_LEGACY_OWNER"}
         and proof["ordinary_reservation_reconciled"] is True
         and 0<=now-proof["observed_epoch"]<AUTHOR_MAX_AGE
         and proof["expires_epoch"]>=now+WHOLE,"FRESH_AUTHORITY_AND_RESOLVED_LEGACY_BRIDGE_REQUIRED")


def utc(value):
    return datetime.strptime(value,"%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def framework_inventory(framework, pinned):
    actual={p.relative_to(framework).as_posix() for p in framework.rglob("*") if p.is_file() or p.is_symlink()}
    names=set(pinned)
    extra=actual-names
    need(names<=actual and extra in (set(),FRAMEWORK_CACHE_ALLOWLIST),
         "EXACT_220_FRAMEWORK_393_WITH_ONLY_OBSERVED_CACHES")
    for name in extra:
        stamp(framework/name)
    return sorted(extra)


def provider_values(root):
    src=root/"src"
    assets=json.loads(small(src/"assets220.json",ASSETS_SHA))
    vision=json.loads(small(src/"vision220.json",VISION_MANIFEST_SHA))
    framework_path=STAGE/"FRAMEWORK_FILES.json"
    framework=json.loads(small(framework_path,FRAMEWORK_SHA))["files"]
    need(len(framework)==393
         and {p.name for p in MODEL.iterdir()}==set(assets["model"]) and len(assets["model"])==14
         and len(vision)==7 and {p.relative_to(VISION).as_posix() for p in VISION.rglob("*") if p.is_file()}==set(vision),
         "EXACT_220_FRAMEWORK_14_MODEL_AND_VISION_INVENTORY")
    ignored=framework_inventory(FRAMEWORK,framework)
    pins={path:(sha,size,None) for path,(sha,size) in PROVIDERS.items()}
    pins[framework_path]=(FRAMEWORK_SHA,None,None)
    for top,items in ((FRAMEWORK,framework),(MODEL,assets["model"]),(VISION,vision)):
        for name,item in items.items():
            pins[top/name]=(item["SHA256"],item["bytes"],item.get("mode"))
    for name,sha in assets["staged_source"].items():
        pins[STAGE/name]=(sha,None,None)
    pins[INITIAL]=("2a0df4d5fdd4d793d47859f7e6ee9102cd3324199f90d0ed34760d3c7198612f",174821899,None)
    return pins,ignored


def original_evidence(src):
    sampler=json.loads(small(src/"sampler1252.json",SAMPLER_SHA))
    old=json.loads(small(src/"terminal1231.json",TERMINAL1231_SHA))
    need(sampler["original_terminal"][:4]==["1194900.1252","qwen-textvqa-sampler-cpu-ab25db1eebbc","COMPLETED","0:0"]
         and [(row["actual_CPU_sampler_slots"],row["unique_original"],row["padding"]) for row in sampler["epochs"]]
             ==[(48,41,7)]*2
         and old["actual_slurm"][2][0]=="1103606.1231" and old["post_clean"] is True
         and old["optimizer_updates"]==2,"EXACT_ORIGINAL_SAMPLER_AND_8_DCU_PRECEDENT")
    return sampler["installed_sampler_source_SHA256"]


def inventory(root, *, hash_bytes):
    pins,ignored=provider_values(root)
    ids={str(path):(stream_check(path,*value) if hash_bytes else stamp(path)) for path,value in pins.items()}
    return {"state":"EXACT_ORIGINAL_RUNTIME_FILES_REHASHED","host":socket.gethostname(),"files":ids,
            "model_files":14,"framework_files":393,"ignored_framework_cache_paths":ignored,
            "original_initializer":True,"GPU":0,
            "model_bytes":sum(ids[str(path)][2] for path in pins if path.parent==MODEL)}
