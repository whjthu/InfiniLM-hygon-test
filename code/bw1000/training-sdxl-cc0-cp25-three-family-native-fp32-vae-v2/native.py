"""Generate the locked three-family paired pilot with the actual SDXL pipeline."""

import argparse
import ctypes
import importlib.util
from importlib import metadata
from functools import wraps
import inspect
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import sys
import time

from plan import ARMS, CONFIG, CP25_BYTES, CP25_SHA, HEX, INDEX_SHA, PIPE_SHA, SCHEDULER_SHA
from plan import digest, frozen, need, original_run, put, read, seal, wire


ADAPTER = "r33_cp25"
DIFFUSERS = Path("/training-runtime/site/diffusers/__init__.py")
DIFFUSERS_SHA = "27b79203d8e25aa693aefceed1a03838b51cfe6359317e16cb50e941a79e2ac9"
PIPELINE_SOURCE = DIFFUSERS.parent / "pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py"
PIPELINE_SOURCE_SHA = "9a5a1d2dd98f38b0171a1bffd39d80e93bc1a8bdbeea5f36444f383c067d286b"
STAGE_MEMBERS = {
    "run.py": "15145c880bd9d3ce08c09607b8ed43a93fae7ddb9f3c73459a81e7c42bcf4c63",
    "sdxl_cc0_witness.py": "6305e21a7a85a11b6706b5fe17e7d9a6a852a68a49b7a5f4c8cf804706856568",
}


def readonly(path):
    need(path.resolve(strict=True) == path and os.statvfs(path).f_flag & os.ST_RDONLY, "EXACT_READONLY_BIND")


def asset_verifier(path):
    for name, pin in STAGE_MEMBERS.items():
        read(path / name, 256 << 10, pin)
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location("frozen_sdxl_asset_verifier", path / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.assets


def authority(args, plan_sha):
    owner = json.loads(read(args.authority, 16 << 10, args.authority_sha))
    readonly(args.authority)
    job, number = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    output_parent = args.output.parent
    need(set(owner) == {"request", "step", "node", "boot", "pci", "epoch", "deadline", "GPU_GO",
                       "plan_SHA256", "output", "gpu_count", "CP25_SHA256"}
         and re.fullmatch(r"[0-9a-f]{32}", args.request) and owner["request"] == args.request
         and owner["GPU_GO"] is True and type(owner["gpu_count"]) is int and owner["gpu_count"] == 1
         and job.isdecimal() and number.isdecimal() and owner["step"] == job + "." + number
         and re.fullmatch(r"[0-9]+", os.environ.get("SLURM_STEP_GPUS", ""))
         and owner["node"] == socket.gethostname().split(".")[0]
         and owner["boot"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
         and re.search(rf"/job_{job}(?:\.scope)?/step_{number}(?:\.scope)?(?:/|$)",
                       Path("/proc/self/cgroup").read_text(), re.MULTILINE)
         and sorted(set(owner["pci"])) == owner["pci"] and len(owner["pci"]) == 8
         and all(re.fullmatch(r"[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", x) for x in owner["pci"])
         and owner["plan_SHA256"] == plan_sha and owner["CP25_SHA256"] == CP25_SHA
         and owner["output"] == str(args.output) and args.output.is_absolute()
         and output_parent.resolve(strict=True) == output_parent and output_parent.stat().st_uid == os.geteuid()
         and stat.S_IMODE(output_parent.stat().st_mode) == 0o700
         and type(owner["epoch"]) in (int, float) and type(owner["deadline"]) in (int, float)
         and owner["epoch"] <= time.time() < owner["deadline"] and 0 < owner["deadline"] - owner["epoch"] <= 2400,
         "EXACT_NEW_SINGLE_CARD_PAIRED_AUTHORITY")
    return owner


def selected_pci():
    hip = ctypes.CDLL("libamdhip64.so")
    count, result = ctypes.c_int(), ctypes.create_string_buffer(32)
    need(hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
         and hip.hipDeviceGetPCIBusId(result, len(result), 0) == 0, "ONE_ACTUAL_HIP_ORDINAL")
    value = result.value.decode("ascii").lower()
    return "0000:" + value if re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", value) else value


def tensor_bytes(tensor, torch):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def latents(samples, folder, torch):
    folder.mkdir(mode=0o700)
    result = {}
    pins = []
    for sample in samples:
        seed, pair = sample["seed"], sample["pair"]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.randn(tuple(CONFIG["latent_shape"]), generator=generator, device="cpu",
                            dtype=torch.float32).to(dtype=torch.bfloat16)
        raw = tensor_bytes(value, torch)
        sha = digest(raw)
        name = f"pair-{pair:02d}.bf16.private"
        need(len(raw) == 8192 and bool(torch.isfinite(value).all()), "EXACT_FINITE_INITIAL_LATENT")
        put(folder / name, raw)
        read(folder / name, len(raw), sha)
        result[pair] = (value, sha)
        pins.append(dict(pair=pair, file=name, seed=seed, dtype=str(value.dtype), shape=list(value.shape),
                         bytes=len(raw), SHA256=sha))
    need(len(set(sha for _, sha in result.values())) == 6, "SIX_DISTINCT_ACTUAL_LATENTS")
    raw = wire(pins)
    put(folder / "LATENTS.json", raw)
    return result, digest(raw)


def adapter(pipe, raw, torch):
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict
    from safetensors.torch import load

    saved = load(raw)
    need(bool(saved) and all(name.startswith("unet.") and bool(torch.isfinite(item).all())
                             for name, item in saved.items()), "UNET_ONLY_ORIGINAL_CP25_WEIGHTS")
    pipe.load_lora_weights(dict(saved), adapter_name=ADAPTER, local_files_only=True)
    pipe.set_adapters([ADAPTER], adapter_weights=[1.0])
    need(pipe.get_active_adapters() == [ADAPTER] and
         {key: value for key, value in pipe.get_list_adapters().items() if value} == {"unet": [ADAPTER]},
         "CP25_SOLE_ACTIVE_NATIVE_ADAPTER")
    live = {"unet." + name: tensor for name, tensor in convert_state_dict_to_diffusers(
            get_peft_model_state_dict(pipe.unet, adapter_name=ADAPTER)).items()}
    need(set(live) == set(saved) and all(value.device.type == "cuda" and value.shape == saved[name].shape
         and torch.equal(value.detach().cpu().float(), saved[name].float()) for name, value in live.items()),
         "EXACT_LIVE_CP25_NAMED_VALUES")
    return dict(active=ADAPTER, strength=1.0, tensors=len(live), SHA256=CP25_SHA)


def fp32_decode(decode, current, witnesses, torch):
    @wraps(decode)
    def consume(z, *args, **kwargs):
        need(current["phase"] == "generation" and current["prepared"] is not None
             and current["decode_count"] == 0 and z.dtype == torch.bfloat16 and z.device.type == "cuda"
             and list(z.shape) == CONFIG["latent_shape"], "ONE_NATIVE_BF16_VAE_OPERAND")
        converted = z.to(dtype=torch.float32)
        need(converted.dtype == torch.float32 and converted.device == z.device
             and converted.shape == z.shape, "EXACT_FP32_VAE_OPERAND")
        current["decode_count"] += 1
        witnesses.append(dict(arm=current["arm"], pair=current["pair"], source_dtype=str(z.dtype),
                              decode_dtype=str(converted.dtype)))
        return decode(converted, *args, **kwargs)
    return consume


def render(args, plan, owner, facts, records, current):
    for path in (args.plan, args.r33_proof, args.pipeline, args.asset_source, args.pipeline_manifest, args.adapter):
        readonly(path)
    need(os.environ.get("PYTHONPATH") == "/training-runtime/site" and
         all(os.environ.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE")),
         "EXACT_OFFLINE_FROZEN_RUNTIME")
    tmp = Path("/tmp")
    need((tmp / ".sdxl-owner-request").read_text().strip() == owner["request"]
         and tmp.stat().st_uid == os.geteuid() and stat.S_IMODE(tmp.stat().st_mode) == 0o700,
         "EXACT_REQUEST_PRIVATE_TMP")
    home = tmp / "paired-home"
    home.mkdir(mode=0o700)
    db, cache = home / ".config/miopen/user-db", home / ".cache/miopen"
    for path in (db, cache):
        path.mkdir(mode=0o700, parents=True)
    os.environ.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_CACHE_HOME=str(home / ".cache"),
                      MIOPEN_USER_DB_PATH=str(db), MIOPEN_CUSTOM_CACHE_DIR=str(cache), TMPDIR=str(tmp))
    current["phase"] = "official_assets"
    original_run(args.r33_proof)
    need(args.adapter.name == "pytorch_lora_weights.safetensors" and args.adapter.parent.name == "checkpoint-25"
         and args.adapter.parent.parent.name == "continuous", "CONTINUOUS_ONLY_CP25_PATH")
    raw_adapter = read(args.adapter, CP25_BYTES, CP25_SHA)
    need(len(raw_adapter) == CP25_BYTES, "ORIGINAL_CP25_EXTENT")
    asset_verifier(args.asset_source)(args.pipeline, args.pipeline_manifest, PIPE_SHA)
    read(args.pipeline / "model_index.json", 16 << 10, INDEX_SHA)
    read(args.pipeline / "scheduler/scheduler_config.json", 16 << 10, SCHEDULER_SHA)
    readonly(DIFFUSERS)
    read(DIFFUSERS, 2 << 20, DIFFUSERS_SHA)
    readonly(PIPELINE_SOURCE)
    read(PIPELINE_SOURCE, 256 << 10, PIPELINE_SOURCE_SHA)

    import diffusers
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline
    from PIL import Image
    import torch

    need(Path(diffusers.__file__) == DIFFUSERS and metadata.version("diffusers") == "0.39.0"
         and torch.version.hip == "6.3.26113" and torch.version.cuda is None and torch.cuda.device_count() == 1,
         "ACTUAL_ORIGINAL_DIFFUSERS_AND_HIP")
    torch.cuda.set_device(0)
    facts["PCI"] = selected_pci()
    need(facts["PCI"] in owner["pci"], "ACTUAL_PCI_IN_PINNED_ROSTER")
    current["phase"] = "frozen_noise_before_output"
    material, facts["latent_manifest_SHA256"] = latents(plan["samples"], args.output / "latents", torch)
    current["phase"] = "load_base"
    vae = AutoencoderKL.from_pretrained(str(args.pipeline), subfolder="vae", torch_dtype=torch.float32,
                                        local_files_only=True, use_safetensors=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(str(args.pipeline), vae=vae, torch_dtype=torch.bfloat16,
                                                     local_files_only=True, use_safetensors=True)
    facts["model_loads"] += 1
    need(type(pipe) is StableDiffusionXLPipeline and type(pipe.scheduler) is EulerDiscreteScheduler,
         "EXACT_NATIVE_PIPELINE_AND_EULER_SCHEDULER")
    pipe.to("cuda:0")
    need(pipe.vae is vae, "DIRECT_FP32_VAE_IDENTITY")
    for model, want in ((pipe.unet, torch.bfloat16), (pipe.text_encoder, torch.bfloat16),
                        (pipe.text_encoder_2, torch.bfloat16), (pipe.vae, torch.float32)):
        model.eval()
        need({param.device.type for param in model.parameters()} == {"cuda"}
             and {param.dtype for param in model.parameters() if param.is_floating_point()} == {want},
             "EXACT_INFERENCE_COMPONENT_DTYPE_DEVICE")
    original_scheduler = digest(wire(dict(pipe.scheduler.config)))
    facts["effective_scheduler_config_SHA256"] = original_scheduler
    native_prepare = pipe.prepare_latents
    signature = inspect.signature(native_prepare)

    def witness_prepare(*args_, **kwargs):
        supplied = signature.bind(*args_, **kwargs).arguments.get("latents")
        need(supplied is not None and current["prepared"] is None and list(supplied.shape) == CONFIG["latent_shape"]
             and str(supplied.dtype) == CONFIG["latent_dtype"]
             and digest(tensor_bytes(supplied, torch)) == current["latent"], "ACTUAL_NATIVE_SUPPLIED_LATENT_BYTES")
        prepared = native_prepare(*args_, **kwargs)
        need(list(prepared.shape) == CONFIG["latent_shape"] and str(prepared.dtype) == CONFIG["latent_dtype"]
             and bool(torch.isfinite(prepared).all()), "ACTUAL_NATIVE_PREPARED_LATENT")
        current["prepared"] = digest(tensor_bytes(prepared, torch))
        timesteps = pipe.scheduler.timesteps
        need(tuple(timesteps.shape) == (CONFIG["steps"],) and bool(torch.isfinite(timesteps).all()),
             "ACTUAL_NATIVE_EULER_TIMESTEPS")
        current["timesteps"] = digest(tensor_bytes(timesteps, torch))
        if current["arm"] == "cp25":
            old = records["base", current["pair"]]
            need(old["prepared_latent_SHA256"] == current["prepared"] and old["timesteps_SHA256"] == current["timesteps"],
                 "PAIRED_NATIVE_PREPARED_LATENT_OR_TIMESTEPS_DIFFERS")
        return prepared

    pipe.prepare_latents = witness_prepare
    need(Path(inspect.getsourcefile(StableDiffusionXLPipeline)).resolve() == PIPELINE_SOURCE,
         "EXACT_NATIVE_VAE_DECODE_CALLER")
    facts["vae_decode_casts"] = []
    pipe.vae.decode = fp32_decode(pipe.vae.decode, current, facts["vae_decode_casts"], torch)
    for arm in ARMS:
        current.update(phase="adapter" if arm == "cp25" else "base", arm=arm, pair=None)
        if arm == "cp25":
            facts["live_adapter"] = adapter(pipe, raw_adapter, torch)
        else:
            need(not any(pipe.get_list_adapters().values()) and pipe.get_active_adapters() == [],
                 "ORIGINAL_UNADAPTED_BASE")
        for item in plan["samples"]:
            pair = item["pair"]
            latent, sha = material[pair]
            current.update(phase="generation", arm=arm, pair=pair, latent=sha, prepared=None, timesteps=None, decode_count=0)
            row = {key: item[key] for key in ("pair", "family", "replicate", "seed", "prompt_SHA256")}
            row.update(arm=arm, outcome="error")
            try:
                need(digest(wire(dict(pipe.scheduler.config))) == original_scheduler and
                     (pipe.get_active_adapters() == [ADAPTER] if arm == "cp25" else pipe.get_active_adapters() == []),
                     "EXACT_GENERATION_CONFIGURATION_AND_ARM")
                generator = torch.Generator(device="cpu").manual_seed(item["seed"])
                initial = latent.clone().to("cuda:0")
                with torch.inference_mode():
                    facts["actual_generation_calls"] += 1
                    output = pipe(prompt=item["prompt"], negative_prompt=CONFIG["negative_prompt"],
                                  height=CONFIG["height"], width=CONFIG["width"], guidance_scale=CONFIG["cfg"],
                                  num_inference_steps=CONFIG["steps"], latents=initial, generator=generator,
                                  num_images_per_prompt=1, output_type="pil")
                image, = output.images
                need(current["prepared"] is not None and current["decode_count"] == 1 and isinstance(image, Image.Image)
                     and image.mode == "RGB" and image.size == (256, 256), "EXACT_NATIVE_GENERATION_OUTPUT")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                raw = buffer.getvalue()
                name = f"{arm}-{pair:02d}.private.png"
                put(args.output / name, raw)
                row.update(outcome="completed", image=name, image_bytes=len(raw), image_SHA256=digest(raw),
                           raw_latent_SHA256=sha, prepared_latent_SHA256=current["prepared"], timesteps_SHA256=current["timesteps"])
            except Exception:
                records[arm, pair] = row
                put(args.output / f"{arm}-{pair:02d}.json", wire(row))
                raise
            records[arm, pair] = row
            put(args.output / f"{arm}-{pair:02d}.json", wire(row))


def run(args):
    plan, plan_sha = frozen(args.plan)
    owner = authority(args, plan_sha)
    args.output.mkdir(mode=0o700)
    facts = dict(step=owner["step"], model_loads=0, actual_generation_calls=0, live_adapter=None)
    current = dict(phase="admission")
    failure = None
    records = {}

    def timed_out(*_):
        raise TimeoutError("ORIGINAL_PAIRED_EPOCH")

    signal.signal(signal.SIGALRM, timed_out)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, owner["deadline"] - time.time() - 10))
        render(args, plan, owner, facts, records, current)
    except BaseException as error:
        failure = dict(phase=current["phase"], arm=current.get("arm"), pair=current.get("pair"),
                       type=type(error).__name__, error_SHA256=digest(str(error).encode()))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result = seal(args.output, plan, plan_sha, args.request, records, facts, failure)
    print(json.dumps({key:result[key] for key in ("request", "passed", "planned_slots", "complete_same_noise_pairs")}))
    return 0 if result["passed"] else 2


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    check = modes.add_parser("verify-plan")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--r33-proof", type=Path, required=True)
    launch = modes.add_parser("run")
    for key in ("plan", "r33-proof", "pipeline", "pipeline-manifest", "asset-source", "adapter", "output", "authority"):
        launch.add_argument("--" + key, type=Path, required=True)
    launch.add_argument("--request", required=True)
    launch.add_argument("--authority-sha", required=True)
    args = parser.parse_args()
    if args.mode == "verify-plan":
        _, sha = frozen(args.plan)
        original_run(args.r33_proof)
        print(json.dumps(dict(plan_SHA256=sha, planned_slots=12, planned_pairs=6, actual_generations=0)))
        return 0
    need(HEX.fullmatch(args.authority_sha), "PINNED_AUTHORITY_SHA")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(cli())
