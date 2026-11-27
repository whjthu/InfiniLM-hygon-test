"""Fixed ingress and bounded metadata contract for an independent T09 cached run."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time
import model_mount

LABEL = "training-hunyuanvideo-t09-abo-domain-n14-fd-v3"
REQUEST = "1b22ddd9e5854853a43e646807e92b3f"
from placement import PARENT, NODE, PLACEMENT
ORIGINAL_NODE = "f11r4n17"
ROOT = PurePosixPath("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RUN = ROOT / "results" / LABEL
FRAMEWORK = ROOT / "results/training-qwen-a68952c-swift-initial-CPU-v1/framework"
FRAMEWORK_MANIFEST = FRAMEWORK.parent / "source/FRAMEWORK_FILES.json"
NATIVE = ROOT / "results/training-hunyuanvideo-original-source-v1/source"
NATIVE_ALIAS = PurePosixPath("/t09-native/hunyuanvideo-1.5")
RUNTIME = ROOT / "results/training-t06-sdxl-runtime-v1"
MODEL = ROOT / "results/training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5"
WEIGHT = MODEL / "transformer/480p_t2v/diffusion_pytorch_model.safetensors"
CACHE = ROOT / "results/hunyuan-4282-storage-recovery-v1/bb1dbf2b7f414a5c9c77771ba3ed573a/cache-output/sample.private.pt"
CONFIG = ROOT / "results/training-hunyuanvideo-original-inputs-v1/config.json"
MODEL_SHA = "71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86"
MODEL_STAMP = [53, 11025132338, 33306632192, 1794715911000000000, 1794715912000000000]
ORIGINAL_EXECUTION_STAMP = [49, *MODEL_STAMP[1:]]
HOST_MODEL_STAMP = [PLACEMENT["host_model_dev"], *MODEL_STAMP[1:]]
COMPUTE_MODEL_STAMP = [PLACEMENT["execution_model_dev"], *MODEL_STAMP[1:]]
FULL_SHA_RECEIPT = ROOT / "results/T09-native25-original-transformer-rebind-fullsha-CPU-v1/normalized-independent-terminal.json"
FULL_SHA_RECEIPT_SHA = "8311536400438e4aac50a361b89030633d952ebb5b320ad4e9771b8d3759aff4"
CACHE_SHA = "8c9d090755a7ec461302232227dd1529f48b1de7066e711d900429e5ddcd1b0c"
CACHE_BYTES = 3521945
CLI_SHA = "b218b9a6c1b5b80aaf6df5a73006b06c1529a30a77b1e4ed7b6a8a1d7a9b4f96"
CONFIG_SHA = "689b2168380a16e50560dece59edfb7cc8b4de4d0438ffc450b9f494a949a6ae"
LAUNCH_SHA = "3fdd29e7ea1a09fc54872c044699199cd5ef16bd73af6e7d522fe2ff1b32662f"
WHOLE, WORK = 1440, 1240
EXPECTED_CLAIM_FLAGS = {"root_GO": True, "GPU_GO": True, "training_GO": True, "READ_GO": True,
                        "CPU_GO": False, "Dataset_GO": False, "Model_GO": False,
                        "formal17_qualified": False, "domain_effect_qualified": False,
                        "performance_qualified": False}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def stamp(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read(path, limit=65536):
    path = Path(path)
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "Metadata file type/extent differs")
    with path.open("rb") as stream:
        need(stamp(os.fstat(stream.fileno())) == stamp(before), "Metadata open identity differs")
        raw = stream.read(limit + 1)
        need(stamp(os.fstat(stream.fileno())) == stamp(before), "Metadata changed during read")
    need(len(raw) == before.st_size and stamp(path.lstat()) == stamp(before), "Metadata path changed")
    return raw, {"path": str(path), "bytes": len(raw), "SHA256": hashlib.sha256(raw).hexdigest()}


def load(path, limit=65536):
    raw, pin = read(path, limit)
    return json.loads(raw), pin


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Budget:
    def __init__(self, origin, now=None, monotonic=None):
        wall = time.time() if now is None else now
        mono = time.monotonic() if monotonic is None else monotonic
        need(type(origin) in (int, float) and math.isfinite(origin) and 0 < origin <= wall < origin + WHOLE,
             "Original epoch invalid, future or expired")
        self.origin, self.mono_end = origin, mono + origin + WHOLE - wall

    def left(self, offset=WHOLE, now=None, monotonic=None):
        wall = time.time() if now is None else now
        mono = time.monotonic() if monotonic is None else monotonic
        return max(0, min(self.origin + offset - wall, self.mono_end - (WHOLE - offset) - mono))


def validated_login_preflight(p):
    binding,proof=load(RUN/"src/MODEL_NFS_BINDING.json",8192)
    need(proof["SHA256"]==p["login_mount_binding_SHA256"],"EXACT_ROOT_NFS_BINDING_IN_PREFLIGHT")
    model=p["model"]
    fresh=model_mount.compare(model["login_mount_observation"],binding,REQUEST,p["fresh_transport"],
                              "fresh",p["publication"])
    need(fresh["file_identity"]==model["host_file_identity"] and fresh["observed_epoch"]<=p["epoch"]
         and p["epoch"]-fresh["observed_epoch"]<=60,"FRESH_SCOPED_LOGIN_MODEL_DESCRIPTOR")
    return binding


def validate_preflight(p, origin):
    need(p.get("state") == "T09_CACHED_NATIVE_TARGET_PREFLIGHT_PASS" and p.get("passed") is True
         and p.get("root_GO") is True, "No approved original asset preflight")
    need((p.get("request_id"), p.get("parent"), p.get("node"), p.get("run"))
         == (REQUEST, PARENT, NODE, str(RUN)), "Preflight scope differs")
    need(p.get("epoch") == origin and p.get("deadline") == origin + WHOLE, "Preflight epoch differs")
    model = p["model"]
    need(model.get("path") == str(WEIGHT) and model.get("bytes") == MODEL_STAMP[2]
         and model.get("SHA256") == MODEL_SHA, "Original weight identity differs")
    current = model.get("file_identity")
    need(type(current) is list and len(current) == 5 and all(type(v) is int and v >= 0 for v in current),
         "Weight fingerprint malformed")
    model_mount.execution_identity(current)
    need(current==COMPUTE_MODEL_STAMP,"Exact compute observation differs")
    validated_login_preflight(p)
    need(model.get("verification") == "EXECUTION_MOUNT_FULL_SHA256"
         and current[1:] == model["host_file_identity"][1:] == MODEL_STAMP[1:],
         "Actual full-SHA compute rediscovery is required")
    need(type(model.get("full_SHA_receipt")) is dict, "Actual execution-mount full-SHA receipt missing")
    receipt = model["full_SHA_receipt"]
    need(receipt.get("path") == str(FULL_SHA_RECEIPT) and receipt.get("bytes") == 932
         and receipt.get("SHA256") == FULL_SHA_RECEIPT_SHA, "Actual qualified full-SHA receipt pin differs")
    for key, path, size, sha in [("cache", CACHE, CACHE_BYTES, CACHE_SHA), ("config", CONFIG, 968, CONFIG_SHA)]:
        need(p[key].get("path") == str(path) and p[key].get("bytes") == size
             and p[key].get("SHA256") == sha, "Original cached asset proof differs")
    need(p.get("framework_manifest_SHA256") ==
         "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41", "Framework identity differs")
    for key, parent in [("native_source_manifest", RUN / "src"), ("runtime_manifest", RUNTIME)]:
        item = p[key]
        path = PurePosixPath(item["path"])
        need(path.is_absolute() and path.parent == parent and type(item["bytes"]) is int
             and 0 < item["bytes"] <= 2 * 1024**2 and len(item["SHA256"]) == 64,
             "Source/runtime manifest binding differs")
    need(p["native_source_manifest"]["path"] == str(RUN / "src/NATIVE_BINDING.json")
         and p["native_source_manifest"]["SHA256"] ==
         "26f1c4e5ef857bb04f5287b4386d4aa9a17ff8cd0bf65d0cece23a5a402633a7",
         "Original scoped native source binding differs")
    need(p["runtime_manifest"]["path"] == str(RUNTIME / "receipt.json")
         and p["runtime_manifest"]["bytes"] == 2124 and p["runtime_manifest"]["SHA256"] ==
         "56f65e63b28f8ec30c7dc1efb3f6dc1df291d4a5d981f91d681f2fad41a03b6a",
         "Original installed runtime receipt differs")


def validate_model_observation(p, actual, *, execution_mount, host_observation=None):
    need(type(execution_mount) is bool,"Explicit HOST versus execution mount selection required")
    binding=validated_login_preflight(p)
    if execution_mount:
        return model_mount.execution_identity(actual)
    observation=model_mount.compare(host_observation,binding,REQUEST,REQUEST,"owner",p["publication"])
    need(observation["file_identity"]==actual and p["epoch"]<=observation["observed_epoch"]<=p["epoch"]+90,
         "EXACT_NEW_CONTROLLER_OWNER_SAME_FD_OBSERVATION")
    return actual[:]


def validate_assets(p, *, execution_mount):
    model = p["model"]
    owner=None if execution_mount else model_mount.observe(REQUEST,REQUEST,"owner",p["publication"])
    candidate=stamp(Path(WEIGHT).lstat()) if execution_mount else owner["file_identity"]
    current_identity=validate_model_observation(p,candidate,execution_mount=execution_mount,host_observation=owner)
    need(not Path(WEIGHT).is_symlink(), "Current weight is not a canonical regular path")
    full_sha_pin = None
    if model["verification"] == "EXECUTION_MOUNT_FULL_SHA256":
        item = model["full_SHA_receipt"]
        path = PurePosixPath(item["path"])
        need(path.is_absolute() and ROOT / "results" in path.parents, "Full-SHA receipt outside approved results")
        actual, pin = load(path, 16384)
        need(pin["bytes"] == item["bytes"] and pin["SHA256"] == item["SHA256"], "Full-SHA receipt binding differs")
        need(actual.get("passed") is True and actual.get("node") == ORIGINAL_NODE
             and actual.get("weight_path") == str(WEIGHT) and actual.get("weight_bytes") == MODEL_STAMP[2]
             and actual.get("weight_SHA256") == MODEL_SHA and actual.get("file_identity") == ORIGINAL_EXECUTION_STAMP
             and actual.get("stable_stat") is True and actual.get("payload_bytes_read") == MODEL_STAMP[2],
             "Actual execution-mount full-SHA evidence differs")
        full_sha_pin = pin
    need(Path(NATIVE / "hyvideo/__init__.py").is_file()
         and not Path(NATIVE).is_symlink(), "Original native checkout is missing")
    for key in ("cache", "config", "native_source_manifest", "runtime_manifest"):
        item = p[key]
        raw, pin = read(Path(item["path"]), max(item["bytes"], 1))
        need(pin["bytes"] == item["bytes"] and pin["SHA256"] == item["SHA256"], "Current small asset identity differs")
        if key == "native_source_manifest":
            binding = json.loads(raw)
            need(len(binding["files"]) == 38, "Native scoped source member denominator differs")
            for relative, declared in binding["files"].items():
                rel = PurePosixPath(relative)
                need(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == relative,
                     "Unsafe native source relative path")
                _, actual = read(NATIVE.joinpath(*rel.parts), declared["bytes"])
                need(actual["bytes"] == declared["bytes"] and actual["SHA256"] == declared["SHA256"],
                     "Current scoped native provider source differs")
    _, model_config = read(WEIGHT.parent / "config.json", 968)
    need(model_config["bytes"] == 968 and model_config["SHA256"] == CONFIG_SHA,
         "Actual native model config differs from frozen original")
    return {"ingress_weight_body_bytes_read": 0, "weight_verification": model["verification"],
            "full_SHA_receipt": full_sha_pin, "weight_file_identity": current_identity,
            "execution_mount_checked": execution_mount, "login_mount_owner_observation": owner,
            "model_config": model_config, "backend_weight_load_and_tree_hash_IO_bytes": "NOT_MEASURED",
            "implicit_loader_GPU_use": "UNKNOWN", "cache_bytes_SHA_verified": True,
            "native_source_files_verified": 38, "runtime_receipt_bytes_verified": True,
            "native_provider_current_qualification": "PENDING_ACTUAL_NATIVE_CALL"}


def verify_package(source, expected):
    manifest, pin = load(source / "SOURCE_FILES.json")
    need(pin["SHA256"] == expected, "Source manifest differs")
    need(set(p.name for p in source.iterdir()) == set(manifest["files"]) | {"SOURCE_FILES.json"},
         "Unexpected delivered source member")
    for name, item in manifest["files"].items():
        need(Path(name).name == name and name not in (".", ".."), "Unsafe source member")
        raw, actual = read(source / name, item["bytes"])
        need(actual["bytes"] == item["bytes"] and actual["SHA256"] == item["SHA256"], "Source member differs")
    claim, _ = load(source / "CLAIM.json")
    need(claim.get("request_id") == REQUEST and claim.get("label") == LABEL
         and claim.get("parent") == PARENT and claim.get("node") == NODE,
         "Root operative claim scope differs")
    need(all(claim.get(name) is expected for name, expected in EXPECTED_CLAIM_FLAGS.items()),
         "Operative root claim flag mismatch")
    return pin
