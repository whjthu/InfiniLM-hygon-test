"""Run one native DiffSynth update with a preceding no-update replay on Style18."""

import csv
from hashlib import sha256
from pathlib import Path
import json
from types import FunctionType


TRAIN = Path("/zimage-input/train")
LOSS = Path("/native-source/diffsynth/diffusion/loss.py")
LOSS_SHA256 = "523b00ccfd815c2f4c07434348ebb79ad294dfea325171b80a452db2c9943de5"


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def data_hash(tensor):
    import torch

    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256(raw).hexdigest()


def exact(path, pin):
    raw = path.read_bytes()
    need(path.is_file() and not path.is_symlink() and len(raw) == pin["bytes"]
         and sha256(raw).hexdigest() == pin["SHA256"], "Style18 frozen input changed")
    return raw


class LossDraws:
    """Substitute only the native loss function's two draws in one call."""

    def __init__(self, torch, fixed=None):
        self.torch = torch
        self.fixed = fixed
        self.seen = {}

    def __getattr__(self, name):
        return getattr(self.torch, name)

    def draw(self, name, generated):
        need(name not in self.seen, "Unexpected repeated native loss draw")
        if self.fixed is None:
            chosen = generated
        else:
            stored = self.fixed[name]
            need(stored.shape == generated.shape and stored.dtype == generated.dtype, "Replay operand metadata differs")
            chosen = stored.to(device=generated.device)
        self.seen[name] = chosen.detach().cpu().clone()
        return chosen

    def randint(self, low, high, size):
        need(size == (1,) and type(low) is int and type(high) is int and low < high, "Invalid native timestep draw")
        chosen = self.draw("timestep_id", self.torch.randint(low, high, size))
        need(low <= int(chosen.item()) < high, "Replay timestep outside scheduler bounds")
        return chosen

    def randn_like(self, sample):
        need(tuple(sample.shape) == (1, 16, 32, 32), "Native noise shape differs")
        return self.draw("noise", self.torch.randn_like(sample))


def bound_loss(native_loss, draws):
    globals_for_call = dict(native_loss.__globals__, torch=draws)
    return FunctionType(native_loss.__code__, globals_for_call, native_loss.__name__, native_loss.__defaults__)


def forward(model, data, native_loss, torch, seed, fixed):
    draws = LossDraws(torch, fixed)
    original = model.task_to_loss[model.task]
    frozen = bound_loss(native_loss, draws)
    status = {}

    def call(pipe, shared, pos, negative):
        status["input_latent_SHA256"] = data_hash(shared["input_latents"])
        return frozen(pipe, **shared, **pos)

    try:
        model.task_to_loss[model.task] = call
        torch.manual_seed(seed)
        loss = model(data)
    finally:
        model.task_to_loss[model.task] = original
    need(set(draws.seen) == {"timestep_id", "noise"}, "Original native loss did not consume both operands")
    status["timestep_id"] = int(draws.seen["timestep_id"].item())
    status["noise_SHA256"] = data_hash(draws.seen["noise"])
    status["timestep_SHA256"] = data_hash(model.pipe.scheduler.timesteps[status["timestep_id"]])
    return loss, status, draws.seen


def load_dataset(model, root, deadline):
    from diffsynth.core import UnifiedDataset
    from diffsynth.diffusion import loss

    deadline()
    need(Path(loss.__file__).resolve() == LOSS and sha256(LOSS.read_bytes()).hexdigest() == LOSS_SHA256,
         "Original native loss provider differs")
    need(model.task == "sft", "Original native task differs")
    binding = json.loads((root / "STYLE18_INPUT.json").read_bytes())
    rows = binding["rows"]
    need(len(rows) == 12 and [row["ordinal"] for row in rows] == list(range(12)), "Frozen training slice differs")
    csv_path = TRAIN / "metadata.csv"
    roster = list(csv.DictReader(exact(csv_path, binding["CSV"]).decode("utf-8").splitlines()))
    need(len(roster) == 12, "Style18 metadata count differs")
    for row, item in zip(roster, rows, strict=True):
        relative = Path(item["path"])
        need(len(relative.parts) == 2 and relative.parts[0] == "images" and relative.suffix == ".png"
             and relative.name == f"{item['ordinal']:03}.png", "Invalid Style18 media path")
        need(row["image"] == item["path"] and sha256(row["prompt"].encode()).hexdigest() == item["prompt_SHA256"],
             "Style18 metadata content differs")
        exact(TRAIN / relative, item["image"])
    dataset = UnifiedDataset(base_path=str(TRAIN), metadata_path=str(csv_path), data_file_keys=("image",),
                             main_data_operator=UnifiedDataset.default_image_operator(
                                 base_path=str(TRAIN), height=256, width=256,
                                 height_division_factor=16, width_division_factor=16))
    need(len(dataset) == 12, "Unexpected Style18 dataset size")
    return dataset, rows


def sample(dataset, rows, ordinal):
    need(type(ordinal) is int and 0 <= ordinal < len(rows), "Unexpected Style18 sample ordinal")
    data = dataset[ordinal]
    need(data["image"].mode == "RGB" and data["image"].size == (256, 256)
         and sha256(data["image"].tobytes()).hexdigest() == rows[ordinal]["RGB256_SHA256"]
         and sha256(data["prompt"].encode()).hexdigest() == rows[ordinal]["prompt_SHA256"],
         "Actual native preprocessed input differs")
    return data
