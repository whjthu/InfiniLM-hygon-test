"""Verify official SDXL VAE CPU loading and latent extraction for two CC0 inputs."""

import argparse
from hashlib import sha256
from importlib import metadata
import json
import math
import os
from pathlib import Path
import stat


WEIGHTS = (334_643_268, "1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341")
CONFIG = (642, "0b331c8ac22ded5f9997a144a575c1d113d6169aff262c353f39015bd24a6264")
IMAGES = {
    "tool_chest": "df0806d6195e8b78ac45b5833356ae7ca656e6ce3bd2cce85434082d81cbf274",
    "painted_stool": "ebc9641bdc1c3b1db45821abb81585fa92ee65da97bbcef04bf227f47b82cd73",
}
SEED = 261128


def require(ok, message):
    if not ok:
        raise ValueError(message)


def regular_pin(path, expected, maximum):
    before = path.lstat()
    require(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
            and 0 < before.st_size <= maximum, "Expected canonical bounded input")
    digest = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            digest.update(block)
    after = path.lstat()
    require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            and digest.hexdigest() == expected, "Official input digest or identity differs")
    return digest.hexdigest()


def run(vae_root, image_root):
    require(os.environ.get("SLURM_STEP_GPUS") in (None, "", "(null)")
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Zero-GRES CPU environment required")
    for name, (size, pin) in (("config.json", CONFIG), ("diffusion_pytorch_model.safetensors", WEIGHTS)):
        path = vae_root / name
        require(path.stat().st_size == size, "Official VAE extent differs")
        regular_pin(path, pin, size)

    from PIL import Image
    import torch
    from torchvision import transforms
    from diffusers import AutoencoderKL

    require(not torch.cuda.is_initialized(), "Preflight must not initialize a GPU")
    torch.set_num_threads(4)
    vae = AutoencoderKL.from_pretrained(str(vae_root), local_files_only=True, use_safetensors=True)
    vae.eval().requires_grad_(False)
    require({(p.device.type, p.dtype) for p in vae.parameters()} == {("cpu", torch.float32)}
            and vae.config.in_channels == 3 and vae.config.latent_channels == 4
            and vae.config.force_upcast is True and vae.config.scaling_factor == 0.13025,
            "Official SDXL VAE configuration or provider differs")
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    rows = {}
    with torch.inference_mode():
        for name, pin in IMAGES.items():
            path = image_root / (name + ".png")
            regular_pin(path, pin, 128 << 10)
            with Image.open(path, formats=("PNG",)) as source:
                require(source.size == (256, 256) and source.mode == "RGB", "Expected exact RGB input extent")
                pixels = preprocess(source.convert("RGB")).unsqueeze(0).contiguous()
            posterior = vae.encode(pixels).latent_dist
            sample = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(SEED))
            latent = (sample * vae.config.scaling_factor).contiguous()
            require(tuple(latent.shape) == (1, 4, 32, 32) and latent.dtype == torch.float32
                    and bool(torch.isfinite(latent).all()) and torch.count_nonzero(latent).item() > 0,
                    "Native VAE latent differs from expected shape or numeric domain")
            repeat = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(SEED))
            require(torch.equal(sample, repeat), "Fixed posterior sample is not repeatable")
            l2 = latent.norm().item()
            require(math.isfinite(l2) and l2 > 0, "Invalid latent norm")
            rows[name] = dict(image_SHA256=pin, latent_SHA256=sha256(latent.numpy().tobytes()).hexdigest(),
                              pixels_sha256=sha256(pixels.numpy().tobytes()).hexdigest(), l2=l2,
                              latent_shape=list(latent.shape), latent_dtype="float32")
    require(len({row["latent_SHA256"] for row in rows.values()}) == 2
            and not torch.cuda.is_initialized(), "Expected distinct CPU-only data derivatives")
    return dict(schema="SDXL_DOMAIN_TWO_CC0_NATIVE_VAE_CPU_V1", image_rows=rows,
                versions={name: metadata.version(name) for name in ("torch", "torchvision", "diffusers")},
                model_frozen=True, gpu_initialized=False, native_vae_encodes=2, optimizer_updates=0,
                training_READY=False, seed=SEED, weight_SHA256=WEIGHTS[1], config_SHA256=CONFIG[1])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("vae_root", type=Path)
    cli.add_argument("image_root", type=Path)
    args = cli.parse_args()
    print(json.dumps(run(args.vae_root, args.image_root), sort_keys=True, allow_nan=False))
