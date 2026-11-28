"""Bind the qualified SDXL pair owner to the native FP32 VAE cast."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re


ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "incoming/sdxl-cc0-cp25-pairs-f957a4d93e33-stage/host.py"
PREVIOUS_SHA = "1df43c403edd53c9ab58c00932d6d8d1f55a1530abc37fbb55c20eb1216781b3"
PLAN = "b4287b48cff81e86c740d446fa6314514f4102bad86f323dee6d83d679299dc4"
RUNNER = "4368ee6b114fbbd93ccd79c30e142088229987d7c4d85de2c18c62905de41f23"


def label(request):
    if not re.fullmatch(r"[a-f0-9]{32}", request):
        raise ValueError("EXACT_NEW_FP32_VAE_REQUEST")
    return "sdxl-cc0-cp25-pairs-fp32-" + request[:12]


def prior():
    if PREVIOUS.resolve(strict=True) != PREVIOUS:
        raise ValueError("EXACT_CANONICAL_PAIRED_OWNER")
    with PREVIOUS.open("rb") as stream:
        raw = stream.read((128 << 10) + 1)
    if not 0 < len(raw) <= 128 << 10 or sha256(raw).hexdigest() != PREVIOUS_SHA:
        raise ValueError("EXACT_PREVIOUS_PAIRED_OWNER_BYTES")
    provider = module_from_spec(spec_from_file_location("previous_paired_owner", PREVIOUS))
    exec(compile(raw, str(PREVIOUS), "exec"), provider.__dict__)
    if provider.PLAN != PLAN:
        raise ValueError("EXACT_SAME_PAIRED_INPUT_PLAN")
    provider.HERE, provider.RUNNER, provider.label = HERE, RUNNER, label
    return provider


def bind():
    return prior().bind()


def paths(request):
    return prior().paths(request)


if __name__ == "__main__":
    raise SystemExit(prior().main())
