"""Replay the frozen A100 SDXL UNet update on one explicitly selected CUDA or HIP device."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys


REFERENCE = "c59a71095924edfdfb9ad7041b8ef32f747e7825557a313f72ad09a3b0b53036"
POLICY = "075e86e54a3eeefb8d5aad219a23f16959d14bd42b22103cd5fe5302c823f6b5"
UNET_FILES = {
    "config.json": "30ebc70750223e59006f7f2b4e1e6c102570aa19a9c4ae3e1fbe7591332dbae6",
    "diffusion_pytorch_model.safetensors": "357650fbfb3c7b4d94c1f5fd7664da819ad1ff5a839430484b4ec422d03f710a",
}


def sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_reference(reference: Path, policy: Path) -> dict:
    require(reference.stat().st_size == 74257307 and sha(reference) == REFERENCE,
            "SDXL reference differs from the frozen A100 first-step r2 payload")
    policy_bytes = policy.read_bytes()
    require(sha256(policy_bytes).hexdigest() == POLICY, "SDXL equivalence policy differs")
    rules = json.loads(policy_bytes)
    require(rules["stage_id"] == "sdxl-sft" and rules["reference_payload_sha256"] == REFERENCE,
            "SDXL policy is not bound to this reference")
    return {"stage_id": "sdxl-sft", "reference_sha256": REFERENCE, "policy_sha256": POLICY}


def replay(reference_path: Path, weights: Path, output: Path, binding: dict, *, accelerator: str = "hip") -> None:
    require(accelerator in ("hip", "cuda"), "unsupported replay accelerator")
    output.mkdir()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation" / "src"))
    from gencuda.runtime import MEMORY_LIMIT_BYTES, get_torch, inspect_runtime, validate_accelerator

    observed_runtime = inspect_runtime()
    validate_accelerator(observed_runtime, accelerator, 1)
    for name, expected in UNET_FILES.items():
        require(sha(weights / "unet" / name) == expected, f"SDXL UNet identity differs: {name}")

    torch = get_torch()
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig

    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    cfg = reference["arguments"]
    initial = reference["initial_adapter"]
    initial_optimizer = reference["initial_optimizer"]
    group, = initial_optimizer["param_groups"]
    require(initial_optimizer["state"] == {} and group["params"] == list(range(len(initial))),
            "SDXL reference is not the first ordered AdamW update")
    require(torch.equal(reference["target"], reference["inputs"]["noise"]),
            "SDXL reference epsilon target differs from its noise")
    require(cfg["mixed_precision"] == "bf16" and cfg["gradient_accumulation_steps"] == 1
            and not cfg["gradient_checkpointing"] and cfg["snr_gamma"] is None
            and reference["scheduler_config"]["prediction_type"] == "epsilon",
            "SDXL first-step recipe differs")

    device = torch.device("cuda", 0)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, MEMORY_LIMIT_BYTES / total), device)
    torch.cuda.reset_peak_memory_stats(device)
    model = UNet2DConditionModel.from_pretrained(
        str(weights), subfolder="unet", use_safetensors=True, local_files_only=True
    ).requires_grad_(False).to(device, dtype=torch.bfloat16)
    model.add_adapter(LoraConfig(
        r=cfg["rank"], lora_alpha=cfg["rank"], init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    model.train()
    named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    require(list(named) == list(initial), "SDXL native parameter names or order differ")
    require(not any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()),
            "SDXL replay does not capture stochastic UNet forward")
    with torch.no_grad():
        for name, parameter in named.items():
            saved = initial[name]
            require(parameter.dtype == saved.dtype == torch.bfloat16 and parameter.shape == saved.shape,
                    f"SDXL parameter schema differs: {name}")
            parameter.copy_(saved)
            require(torch.equal(parameter.cpu(), saved), f"SDXL initial parameter copy differs: {name}")

    options = {key: value for key, value in group.items() if key not in ("params", "initial_lr")}
    require(options.pop("decoupled_weight_decay") is True, "SDXL reference does not use decoupled AdamW")
    optimizer = torch.optim.AdamW(list(named.values()), **options)
    optimizer.load_state_dict(initial_optimizer)

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu(item) for item in value)
        return value

    actual_initial = cpu(named)
    actual_initial_optimizer = cpu(optimizer.state_dict())

    def gradients():
        require(all(parameter.grad is not None for parameter in named.values()), "missing SDXL gradients")
        values = {name: cpu(parameter.grad) for name, parameter in named.items()}
        require(all(bool(torch.isfinite(value).all()) for value in values.values()), "nonfinite SDXL gradients")
        return values

    inputs = reference["inputs"]
    conditions = {key: value.to(device) for key, value in inputs["unet_added_conditions"].items()}
    noisy = inputs["noisy_model_input"].to(device)
    timesteps = inputs["timesteps"].to(device)
    prompts = inputs["prompt_embeds"].to(device)
    target = reference["target"].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw = model(noisy, timesteps, prompts, added_cond_kwargs=conditions, return_dict=False)[0]
        prediction = raw.float()
        loss = torch.nn.functional.mse_loss(prediction, target.float(), reduction="mean")
    require(raw.device == device and bool(torch.isfinite(loss)), "invalid SDXL forward")
    loss.backward()
    before = gradients()
    norm = torch.nn.utils.clip_grad_norm_(list(named.values()), cfg["max_grad_norm"])
    require(bool(torch.isfinite(norm)) and float(norm) > 0, "invalid SDXL preclip gradient norm")
    after = gradients()
    optimizer.step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_reserved(device)
    require(peak <= MEMORY_LIMIT_BYTES, "SDXL peak exceeds the configured device memory limit")
    final = {name: cpu(parameter) for name, parameter in named.items()}
    require(any(not torch.equal(initial[name], final[name]) for name in named), "zero SDXL update")

    # Bind the actual device inputs; the other captured inputs are frozen context.
    realized = {**inputs, "noisy_model_input": cpu(noisy), "timesteps": cpu(timesteps),
                "prompt_embeds": cpu(prompts), "unet_added_conditions": {
                    key: cpu(value) for key, value in conditions.items()}}
    observed = output / "observed.pt"
    torch.save({
        "initial_adapter": actual_initial, "initial_optimizer": actual_initial_optimizer,
        "inputs": realized, "batch": reference["batch"], "target": cpu(target),
        "raw_prediction": cpu(raw), "prediction": cpu(prediction), "loss": cpu(loss),
        "preclip_norm": cpu(norm), "gradients_before_clip": before, "gradients_after_clip": after,
        "final_adapter": final, "final_optimizer": cpu(optimizer.state_dict()),
    }, observed)
    (output / "summary.json").write_text(json.dumps({
        **binding, "boundary": "fixed_realized_inputs", "observed_sha256": sha(observed),
        "runtime": asdict(observed_runtime), "peak_reserved_bytes": peak,
        "unet_files": UNET_FILES, "replay_source_sha256": sha(Path(__file__)),
        "numerical_acceptance": "pending independent comparison",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--accelerator", choices=("hip", "cuda"), default="hip")
    args = parser.parse_args()
    if not args.verify_reference and (args.weights is None or args.output is None or args.output.exists()):
        parser.error("replay requires --weights and a new --output directory")
    binding = verify_reference(args.reference, args.policy)
    if args.verify_reference:
        print(json.dumps(binding))
        return
    replay(args.reference, args.weights, args.output, binding, accelerator=args.accelerator)


if __name__ == "__main__":
    main()
