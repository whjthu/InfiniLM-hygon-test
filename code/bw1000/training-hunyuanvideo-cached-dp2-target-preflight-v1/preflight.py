"""Read-only binding of the exact T09 DP2 recipe to existing target assets."""
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys


REQUEST = "223e1b257df746aeb37779d2376ef49f"
LABEL = "training-hunyuanvideo-cached-dp2-target-v1"
PARENT, NODE = "1087656", "f11r4n17"
HERE = Path(__file__).resolve().parent
ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = ROOT / "results"
BASE = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1"
FRAMEWORK = BASE / "framework"
ENTRY = FRAMEWORK / "tools/hygon_training_recipe.py"
NATIVE = RESULTS / "training-hunyuanvideo-original-source-v1/source"
NATIVE_ALIAS = PurePosixPath("/t09-native/hunyuanvideo-1.5")
MODEL = RESULTS / "training-hunyuanvideo-original-model-v1/hunyuanvideo-1.5"
MODEL_SHA = "71f9affa1115fef2b14bd41fba30eab966fe80c9ed98e0fcba495dbc6d8fff86"
WEIGHT = MODEL / "transformer/480p_t2v/diffusion_pytorch_model.safetensors"
CPU_WEIGHT_RECEIPT = RESULTS / "T09-native25-original-transformer-rebind-fullsha-CPU-v1/normalized-independent-terminal.json"
CACHE = RESULTS / "training-hunyuanvideo-original-inputs-v1/sample.pt"
CACHE_SHA = "9bdbaccaada6be6080ace96aa464f70adc8041bd43e58eae58052c5b2eab541b"
DP2_SHA = "181b1ca706a7d62741298192c3391a0336acf0c60624aa9f2645e90a55934a6f"
ORIGINAL_NATIVE_LAUNCH = "node/results/hunyuan-cuda-dp2-baseline-r1-repeat1/launch.json"
OVERLAY = RESULTS / "training-hunyuanvideo-cached-native25-target-v3/overlay/flash_attn_no_pad.py"
NATIVE_FLASH = "hyvideo/utils/flash_attn_no_pad.py"
OVERLAY_SHA = "f9762977bf93bf7ea41258b32eafa6985b9461702274882b42cb71d112e084a0"
ORIGINAL_PATHS = dict(cache="/mnt/data/nvidia-baseline-20261010/inputs/datasets/hunyuanvideo-1.5/sample.pt",
                      model="/mnt/data/nvidia-baseline-20261010/inputs/models/hunyuanvideo-1.5",
                      native="/mnt/data/nvidia-baseline-20261010/sources/hunyuanvideo-1.5")
PATCHES = {
    "generation/src/gencuda/backends/hunyuanvideo_data_parallel.py":
        "f21dfc2a5234c0882de9fadbc042f3fb4ab015a22dab8d02f66c3f4a6239fd0b",
    "generation/src/gencuda/hunyuanvideo_distributed_checkpoint.py":
        "f39849ffe7ed3659ed3865ac180aa9f9ee086dc75355119fb55021560692f87d",
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read_pinned(path, wanted, size=None):
    require(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
            "Expected regular, unaliased original " + path.name)
    with path.open("rb") as stream:
        raw = stream.read((size if size is not None else 1 << 20) + 1)
    require((size is None or len(raw) == size) and sha256(raw).hexdigest() == wanted,
            "Frozen original length or SHA differs: " + path.name)
    return raw


def source_check(directory=HERE):
    for rel, wanted in PATCHES.items():
        path = directory / Path(rel).name
        raw = read_pinned(path, wanted)
        compile(raw, rel, "exec")
    return dict(local_overlay_files=PATCHES, deployed_verified=False)


def validate_recipe(launch):
    config = launch["config"]
    require(launch["weights"] == ORIGINAL_PATHS["model"] and launch["source"] == ORIGINAL_PATHS["native"]
            and config["training_cache"] == ORIGINAL_PATHS["cache"] and config["training_cache_sha"] == CACHE_SHA,
            "Frozen DP2 assets differ")
    require((config["world_size"],config["sp"],config["seed"],config["transformer_version"])
            == (2,1,20260911,"480p_t2v") and (config["warmup_steps"],config["max_steps"]) == (5,20)
            and config["checkpoint_steps"] == [10,20,25] and (config["lora_rank"],config["lora_alpha"]) == (4,8)
            and config["lora_target_modules"] == ["img_attn_q","img_attn_v"],
            "Original two-rank training recipe differs")
    return {ORIGINAL_PATHS["model"]:str(MODEL), ORIGINAL_PATHS["native"]:str(NATIVE_ALIAS),
            ORIGINAL_PATHS["cache"]:str(CACHE)}


def native_argv(original, asset_map, cards, phase, checkpoint=None):
    require(re.fullmatch(r"(?:0|[1-9][0-9]*),(?:0|[1-9][0-9]*)", cards) and
            len(set(cards.split(","))) == 2 and all(x.is_absolute() for x in (original,asset_map)),
            "Exactly two distinct assigned card selectors and original absolute files are required")
    require(phase in ("continuous","resume10","resume20"), "Unknown DP2 phase")
    run = RESULTS / LABEL
    argv = [str(ENTRY),"T09.dp2","--a100-launch",str(original),"--asset-map",str(asset_map),
            "--run","--cards",cards,"--source-root",str(NATIVE_ALIAS.parent),"--output-root",str(run/"outputs"),
            "--run-name",phase,"--lock-root",str(run/"locks"),"--lock-host",NODE]
    if phase != "continuous":
        step = 10 if phase == "resume10" else 20
        expected = run/"outputs"/"hunyuanvideo-1.5"/"train"/"continuous"/f"checkpoint-{step}"
        require(checkpoint == expected, "Resume must use this new DP2 continuous transaction")
        argv += ["--resume-from-checkpoint",str(checkpoint)]
    return argv


def host_assets():
    cache = read_pinned(CACHE,CACHE_SHA,3521630)
    del cache
    read_pinned(OVERLAY,OVERLAY_SHA,4212)
    read_pinned(NATIVE/NATIVE_FLASH,
                "e9835a76bb2908a1160fec435a9b2e230b6db0588199ea48e6ae66309b2b5b67",3909)
    require(WEIGHT.resolve(strict=True) == WEIGHT and stat.S_ISREG(WEIGHT.lstat().st_mode)
            and WEIGHT.stat().st_size == 11025132338, "Original target transformer extent differs")
    read_pinned(CPU_WEIGHT_RECEIPT,
                "8311536400438e4aac50a361b89030633d952ebb5b320ad4e9771b8d3759aff4",932)
    manifest = json.loads(read_pinned(BASE/"source/FRAMEWORK_FILES.json",
                       "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",112315))
    require(len(manifest["files"]) == 393 and manifest["framework_tree"] ==
            "7373ac3f6eee95828af4eb2e457ff4ea56a5d888", "Original registered framework differs")
    for path in [ENTRY,*[FRAMEWORK / rel for rel in PATCHES]]:
        pin = manifest["files"][path.relative_to(FRAMEWORK).as_posix()]
        read_pinned(path, pin["SHA256"],pin["bytes"])
    return dict(cache_sha256=CACHE_SHA, cache_bytes=3521630, transformer_stat_bytes=11025132338,
                prior_transformer_full_sha256=MODEL_SHA, original_model_body_bytes_read=0,
                existing_framework_files_pinned=393, existing_framework_files_rechecked_here=3,
                exact_native_unpad_sha256=OVERLAY_SHA,
                mounts=[f"{NATIVE}:{NATIVE_ALIAS}:ro",f"{OVERLAY}:{NATIVE_ALIAS/NATIVE_FLASH}:ro",
                        *[f"{HERE/Path(rel).name}:{FRAMEWORK/rel}:ro" for rel in PATCHES]])


def mounted_check(original, asset_path):
    for rel,wanted in PATCHES.items():
        read_pinned(FRAMEWORK/rel,wanted)
    read_pinned(Path(NATIVE_ALIAS)/NATIVE_FLASH,OVERLAY_SHA,4212)
    receipt = subprocess.run([sys.executable,"-B",str(ENTRY),"T09.dp2","--a100-launch",str(original),
                              "--asset-map",str(asset_path)],check=True,capture_output=True,timeout=30)
    plan = json.loads(receipt.stdout)
    require((plan["case"],plan["cards"],plan["a100_launch_sha256"])==("T09.dp2",2,DP2_SHA)
            and plan["config"]["training_cache"]==str(CACHE), "Actual common CLI rejected original DP2 binding")
    return dict(common_CLI_Plan=plan, backend_overlay_files=PATCHES, exact_unpad_sha256=OVERLAY_SHA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("source","host","mounted"))
    p.add_argument("--a100-launch",type=Path)
    p.add_argument("--asset-map",type=Path)
    args = p.parse_args()
    result = dict(source_check(),request=REQUEST,label=LABEL,parent=PARENT,node=NODE,training_READY=False,
                  gpu_steps_started=0,optimizer_updates=0)
    if args.mode in ("host","mounted"):
        if args.a100_launch is None:
            p.error("--a100-launch requires the exact 1,563-byte A100 DP2 original")
        raw = read_pinned(args.a100_launch,DP2_SHA,1563)
        mapping = validate_recipe(json.loads(raw))
        if args.mode == "host":
            result.update(host_assets(),asset_map=mapping)
        else:
            if args.asset_map is None or json.loads(args.asset_map.read_bytes()) != mapping:
                p.error("--asset-map must match the original target asset binding")
            result.update(mounted_check(args.a100_launch,args.asset_map))
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__ == "__main__":
    main()
