"""Replay realized Z-Image inputs at the original DiffSynth forward boundary."""

import ast
from contextlib import contextmanager
import hashlib
import inspect
from io import BytesIO
import json
from pathlib import Path
import textwrap


NATIVE_LOSS_SHA256 = "523b00ccfd815c2f4c07434348ebb79ad294dfea325171b80a452db2c9943de5"
PREPARED_FIELDS = {"inputs", "noise", "timestep_id", "timestep"}
LATENT_SHAPE = (1, 16, 32, 32)
REFERENCE_OPERANDS = {"prepared.pt", "target.pt"}
IMAGE_IDENTITY_FIELDS = {"kind", "mode", "size", "decoded_pixels_SHA256"}


def _checked_image(native, reference, key):
    from PIL import Image

    if (not isinstance(native, Image.Image) or reference.keys() != IMAGE_IDENTITY_FIELDS
            or native.mode != reference["mode"] or list(native.size) != reference["size"]
            or hashlib.sha256(native.tobytes()).hexdigest() != reference["decoded_pixels_SHA256"]):
        raise ValueError(f"Realized Z-Image image pixels differ at {key}")
    return native


def _pinned_bytes(path, expected, maximum, expected_bytes=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Original capture member is missing: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if (not 0 < len(raw) <= maximum or (expected_bytes is not None and len(raw) != expected_bytes)
            or hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError(f"Original capture member differs: {path.name}")
    return raw


def load_fixed_reference(torch, root, expected_progress_sha256):
    """Authenticate the two original operand archives in place."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Original capture directory cannot be a symlink")
    raw = _pinned_bytes(root / "progress.json", expected_progress_sha256, 65536)
    progress = json.loads(raw)
    step = {"HIP_COCO20_FIRST_UPDATE_RETURNED": 1, "NATIVE25_RETURNED": 25}.get(progress.get("phase"))
    if (progress.get("case") != "T07.sft" or progress.get("first_update_complete") is not True
            or progress.get("failed") is not False or step is None
            or progress.get("planned_global_steps") != step or progress.get("completed_global_steps") != step):
        raise ValueError("Original COCO20 first-update evidence is incomplete")
    pins = [item for item in progress["files"] if item["name"] in REFERENCE_OPERANDS]
    if len(pins) != len(REFERENCE_OPERANDS) or {p["name"] for p in pins} != REFERENCE_OPERANDS:
        raise ValueError("Expected exactly two first-update operand archives")
    records = {}
    for pin in pins:
        raw = _pinned_bytes(root / pin["name"], pin["SHA256"], 8 << 20, pin["bytes"])
        records[pin["name"]] = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    return FixedReference(torch, records["prepared.pt"], records["target.pt"])


class FixedReference:
    """One native forward consumes one already authenticated CPU capture."""

    def __init__(self, torch, prepared, target):
        if type(prepared) is not dict or prepared.keys() != PREPARED_FIELDS:
            raise ValueError("Unexpected original Z-Image prepared inputs")
        if type(target) is not dict or target.keys() != {"training_target"}:
            raise ValueError("Unexpected original Z-Image training target")
        if type(prepared["inputs"]) is not dict:
            raise ValueError("Expected the original native input map")
        tensors = (prepared["noise"], prepared["inputs"]["input_latents"],
                   prepared["inputs"]["latents"], target["training_target"])
        if any(not isinstance(t, torch.Tensor) or tuple(t.shape) != LATENT_SHAPE
               for t in tensors):
            raise ValueError("Reference noise, latent or target has wrong extent")
        index = prepared["timestep_id"]
        if (not isinstance(index, torch.Tensor) or tuple(index.shape) != (1,)
                or index.dtype != torch.int64 or int(index.item()) != 726):
            raise ValueError("Expected the frozen first COCO20 timestep")
        self.torch = torch
        self.reference = {**prepared, **target}
        self.applied = False

    def _place(self, native, reference, key):
        torch = self.torch
        if type(reference) is dict and reference.get("kind") == "PIL.Image":
            return _checked_image(native, reference, key)
        if isinstance(reference, torch.Tensor):
            if (not isinstance(native, torch.Tensor) or reference.device.type != "cpu"
                    or reference.dtype != native.dtype or reference.shape != native.shape
                    or reference.layout != torch.strided or native.layout != torch.strided
                    or reference.requires_grad or native.requires_grad
                    or not bool(torch.isfinite(reference).all())):
                raise ValueError(f"Realized Z-Image tensor schema differs at {key}")
            return reference.to(device=native.device, copy=True)
        if type(native) is not type(reference):
            raise ValueError(f"Realized Z-Image value type differs at {key}")
        if type(reference) is dict and native.keys() == reference.keys():
            return {name: self._place(native[name], value, name) for name, value in reference.items()}
        if type(reference) in (tuple, list) and len(native) == len(reference):
            return type(reference)(self._place(a, b, key) for a, b in zip(native, reference, strict=True))
        if (reference is None or type(reference) in (bool, int, float, str, bytes)) and native == reference:
            return reference
        raise ValueError(f"Realized Z-Image conditioning differs at {key}")

    def apply(self, inputs, noise, timestep_id, timestep, training_target):
        if self.applied:
            raise RuntimeError("The fixed Z-Image reference was already consumed")
        native = dict(inputs=inputs, noise=noise, timestep_id=timestep_id, timestep=timestep,
                      training_target=training_target)
        fixed = self._place(native, self.reference, "prepared")
        self.applied = True
        return tuple(fixed[k] for k in native)


def _compile_loss(original, replace, observer):
    source_path = Path(inspect.getsourcefile(original))
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != NATIVE_LOSS_SHA256:
        raise ValueError("DiffSynth loss source differs from the frozen COCO20 recipe")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    (function,) = tree.body
    if not isinstance(function, ast.FunctionDef) or function.name != "FlowMatchSFTLoss" or function.decorator_list:
        raise ValueError("Unexpected original DiffSynth loss function")
    anchor = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "training_target" for t in node.targets)]
    if len(anchor) != 1:
        raise ValueError("Missing unique realized input boundary")
    following = function.body[anchor[0] + 1:]
    model_calls = [node for line in following for node in ast.walk(line) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "model_fn"]
    if len(model_calls) != 1:
        raise ValueError("Missing unique native model forward")
    rewrite = ast.parse(
        "inputs, noise, timestep_id, timestep, training_target = "
        "_replace_reference(inputs, noise, timestep_id, timestep, training_target)"
    ).body[0]
    function.body.insert(anchor[0] + 1, rewrite)
    if observer is not None:
        function.body.insert(anchor[0] + 2, ast.parse(
            "_reference_observer.prepared(inputs, noise, timestep_id, timestep, training_target)"
        ).body[0])
        returns = [i for i, statement in enumerate(function.body) if isinstance(statement, ast.Return)]
        if len(returns) != 1:
            raise ValueError("Missing unique native model return")
        function.body.insert(returns[0], ast.parse(
            "_reference_observer.forward_value(noise_pred, loss)"
        ).body[0])
    scope = dict(original.__globals__, _replace_reference=replace, _reference_observer=observer)
    exec(compile(ast.fix_missing_locations(tree), str(source_path), "exec"), scope)
    return scope[function.name]


@contextmanager
def replay_native_loss(native_module, reference, *, observer=None):
    """Use the pinned native loss, replacing the realized operands before model_fn."""
    original = native_module.FlowMatchSFTLoss
    if (reference.applied or getattr(original, "_zimage_fixed_reference", False)
            or getattr(original, "_zimage_first_update_observer", False)):
        raise RuntimeError("Invalid overlapping Z-Image fixed replay")
    installed = _compile_loss(original, reference.apply, observer)
    installed._zimage_fixed_reference = True
    native_module.FlowMatchSFTLoss = installed
    try:
        yield
        if not reference.applied:
            raise RuntimeError("Native forward did not consume the fixed Z-Image input")
    finally:
        native_module.FlowMatchSFTLoss = original
