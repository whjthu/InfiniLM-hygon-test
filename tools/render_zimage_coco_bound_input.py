"""Evaluate one bound Z-Image development input using its frozen BF16 noise."""

from hashlib import sha256
import inspect
import math
import sys


PIPE_SIGNATURE = ("shape", "seed", "rand_device", "rand_torch_dtype", "device", "torch_dtype")
START_SHAPE = (1, 16, 32, 32)
START_BYTES = 32768


def require(ok, message):
    if not ok:
        raise ValueError(message)


def render_bound_input(*, pipe, torch, noise_bank, entry, inference_steps, guidance):
    """Call the native pipeline once with one previously verified development row."""
    require(sys.byteorder == "little", "frozen noise requires little-endian tensor storage")
    require(type(inference_steps) is int and inference_steps > 0 and math.isfinite(guidance) and guidance > 0,
            "invalid render recipe")
    prompt = entry["prompt"]
    require(sha256(prompt.encode("utf-8")).hexdigest() == entry["prompt_SHA256"], "caption bytes changed")
    seed = entry["seed"]
    raw, expected = noise_bank.sample(entry["image_id"], entry["prompt_SHA256"], seed)
    require(len(raw) == START_BYTES and sha256(raw).hexdigest() == expected, "starting noise bytes changed")

    native = pipe.generate_noise
    require(tuple(inspect.signature(native).parameters) == PIPE_SIGNATURE, "native noise call signature changed")
    had_override = "generate_noise" in vars(pipe)
    original_override = vars(pipe).get("generate_noise")
    consumed = []

    def frozen(shape, seed=None, rand_device="cpu", rand_torch_dtype=None, device=None, torch_dtype=None):
        require(not consumed and tuple(shape) == START_SHAPE and seed == entry["seed"], "unexpected noise request")
        require(rand_torch_dtype in (None, torch.float32), "native generator source dtype differs")
        output_dtype = torch_dtype or pipe.torch_dtype
        output_device = torch.device(device if device is not None else pipe.device)
        require(output_dtype == torch.bfloat16 and output_device.type == "cuda",
                "expected BF16 starting noise on the active accelerator")
        target_index = output_device.index if output_device.index is not None else torch.cuda.current_device()
        tensor = torch.frombuffer(bytearray(raw), dtype=output_dtype).reshape(START_SHAPE).to(device=output_device)
        uploaded = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        observed = sha256(uploaded).hexdigest()
        require(tensor.dtype == output_dtype and tensor.device.type == "cuda" and tensor.device.index == target_index
                and observed == expected,
                "uploaded starting noise differs")
        consumed.append(observed)
        return tensor

    pipe.generate_noise = frozen
    try:
        with torch.no_grad():
            image = pipe(prompt=prompt, seed=seed, rand_device="cuda", height=256, width=256,
                         num_inference_steps=inference_steps, cfg_scale=guidance)
        require(consumed == [expected] and image.size == (256, 256), "native output or starting noise use differs")
    finally:
        if had_override:
            pipe.generate_noise = original_override
        else:
            del pipe.generate_noise

    return image, {"image_id": entry["image_id"], "ordinal": entry["ordinal"], "seed": seed,
                   "prompt_SHA256": entry["prompt_SHA256"], "initial_noise_SHA256": consumed[0],
                   "inference_steps": inference_steps, "guidance": guidance, "height": 256, "width": 256}
