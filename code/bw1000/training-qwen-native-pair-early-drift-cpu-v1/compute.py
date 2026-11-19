"""Locate the first tensor-level difference in two completed Qwen training runs."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import torch


ROOTS = (Path("/original"), Path("/comparison"))
OUT = Path("/evidence")
CHUNK_ELEMENTS = 1 << 20


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1 << 20), b""):
            value.update(raw)
    return value.hexdigest()


def load(path, pin):
    require(path.is_file() and not path.is_symlink(), "FROZEN_TENSOR_FILE")
    before = path.stat()
    require(before.st_size == pin["bytes"] and digest(path) == pin["SHA256"], "FROZEN_TENSOR_HASH")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    require(identity(path.stat()) == identity(before), "UNCHANGED_FROZEN_TENSOR_FILE")
    return value


class Difference:
    def __init__(self):
        self.tensors = 0
        self.elements = 0
        self.changed_tensors = 0
        self.changed_elements = 0
        self.changed_scalars = 0
        self.max_abs = 0.0
        self.squared_delta = 0.0
        self.squared_original = 0.0

    def add(self, left, right):
        if isinstance(left, torch.Tensor):
            require(isinstance(right, torch.Tensor) and left.shape == right.shape
                    and left.dtype == right.dtype and left.device.type == right.device.type == "cpu",
                    "PAIRED_TENSOR_SHAPE_OR_TYPE")
            self.tensors += 1
            n = left.numel()
            self.elements += n
            left, right = left.reshape(-1), right.reshape(-1)
            modified = 0
            for start in range(0, n, CHUNK_ELEMENTS):
                a = left[start:start+CHUNK_ELEMENTS]
                b = right[start:start+CHUNK_ELEMENTS]
                modified += int(torch.count_nonzero(a != b).item())
                if a.is_floating_point():
                    a, b = a.double(), b.double()
                    require(bool(torch.isfinite(a).all() & torch.isfinite(b).all()), "NONFINITE_CAPTURE")
                    delta = a - b
                    self.max_abs = max(self.max_abs, float(delta.abs().max().item()))
                    self.squared_delta += float(torch.dot(delta, delta).item())
                    self.squared_original += float(torch.dot(a, a).item())
            self.changed_elements += modified
            self.changed_tensors += bool(modified)
        elif isinstance(left, dict):
            require(isinstance(right, dict) and left.keys() == right.keys(), "PAIRED_TREE_KEYS")
            for key in left:
                self.add(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            require(type(left) is type(right) and len(left) == len(right), "PAIRED_TREE_LENGTH")
            for a, b in zip(left, right):
                self.add(a, b)
        else:
            require(type(left) is type(right), "PAIRED_SCALAR_TYPE")
            self.changed_scalars += int(left != right)

    def result(self):
        return dict(tensors=self.tensors, elements=self.elements, changed_tensors=self.changed_tensors,
                    changed_elements=self.changed_elements, changed_scalars=self.changed_scalars,
                    max_abs=self.max_abs, abs_l2=math.sqrt(self.squared_delta),
                    rel_l2=math.sqrt(self.squared_delta/self.squared_original) if self.squared_original else None)


def differences(left, right):
    d = Difference()
    d.add(left, right)
    return d.result()


def receipt(root, expected, rank):
    folder = root / "capture" / f"rank-{rank:02d}"
    path = folder / "receipt.json"
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "PAIRED_RANK_RECEIPT_HASH")
    info = json.loads(raw)
    require(info["rank"] == rank and info["optimizer_updates"] == info["clip_calls"] == 2
            and info["micro_forwards"] == 8 and len(info["files"]) == 25, "PAIRED_RANK_RECEIPT")
    pins = {row["name"]: {key: row[key] for key in ("bytes", "SHA256")} for row in info["files"]}
    require(len(pins) == 25, "PAIRED_UNIQUE_FILE_PINS")
    return folder, pins


def pair(files, name):
    return [load(folder / name, pins[name]) for folder, pins in files]


def compare_forward(pair_files, ordinal):
    name = f"forward-{ordinal:02d}-output.pt"
    left, right = pair(pair_files, name)
    require(left["global_step"] == right["global_step"] == (ordinal-1)//4
            and left["micro_ordinal"] == right["micro_ordinal"] == ordinal, "PAIRED_FORWARD_ORDER")
    a, b = left["result"], right["result"]
    require(isinstance(a["loss"], torch.Tensor) and a["loss"].numel() == b["loss"].numel() == 1,
            "PAIRED_NATIVE_LOSS")
    return dict(loss_before=a["loss"].item(), loss_after=b["loss"].item(), result=differences(a,b),
                logits=differences(a["logits"],b["logits"]) if a.get("logits") is not None else None)


def compare_update(files, step):
    prefix = f"update-{step:02d}"
    out = {}
    for stage in ("preclip", "postclip"):
        left, right = pair(files, prefix + "-" + stage + ".pt")
        require(left["global_step"] == right["global_step"] == step-1, "PAIRED_CLIP_STEP")
        out[stage] = differences(left["gradients"], right["gradients"])
        if stage == "postclip":
            out["reported_preclip_norm"] = [float(left["reported_preclip_norm"]), float(right["reported_preclip_norm"])]
        del left, right
    for stage, key in (("before","global_step"),("after","global_step_before_increment")):
        left, right = pair(files, prefix + "-" + stage + "-optimizer.pt")
        require(left[key] == right[key] == step-1, "PAIRED_OPTIMIZER_STEP")
        out[stage] = {k: differences(left[k],right[k]) for k in ("parameters","optimizer")}
        del left, right
    return out


def main():
    epoch = float(os.environ["REFERENCE_READER_EPOCH"])
    deadline = float(os.environ["REFERENCE_READER_DEADLINE"])
    require(deadline == epoch + 870 and 0 < deadline-time.time() <= 870, "READER_EPOCH")
    signal.setitimer(signal.ITIMER_REAL, deadline-time.time())
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and not torch.cuda.is_initialized(), "CPU_ONLY")
    def no_device(*args, **kwargs):
        raise RuntimeError("NO_GPU_IN_INDEPENDENT_READER")
    torch.cuda._lazy_init = no_device
    torch.set_num_threads(2)
    binding = json.loads(Path("/fix/TERMINAL_BINDING.json").read_bytes())
    source_boundaries = []
    for root, row in zip(ROOTS, binding["originals"]):
        path = root / "outcome.json"
        require(digest(path) == row["outcome_SHA256"], "PAIRED_ORIGINAL_TERMINAL")
        actual = json.loads(path.read_bytes())
        require(actual["keeper_passed"] and actual["after"]["kfd_holder_count"] == 0, "PAIRED_ASSET_LIFECYCLE")
        source_boundaries.append(actual["native-outcome.json"]["value"]["rank_results"])
    files_by_rank = [[receipt(root, x[rank]["receipts_SHA256"]["receipt.json"], rank)
                      for root, x in zip(ROOTS,source_boundaries)] for rank in range(8)]
    for rank, files in enumerate(files_by_rank):
        a, b = (row[1] for row in files)
        for name in ("initial.pt", *(f"forward-{i:02d}-input.pt" for i in range(1,9))):
            require(a[name] == b[name], "PAIRED_MODEL_OR_RNG_INPUT_PIN")
        observed = {i for i in range(1,9) if a[f"forward-{i:02d}-output.pt"] != b[f"forward-{i:02d}-output.pt"]}
        require(observed == set(binding["different_forward_by_rank"][str(rank)]), "ORIGINAL_FORWARD_MANIFEST_MEMBERSHIP")
    selected = [(rank, i) for rank in range(8) for i in binding["different_forward_by_rank"][str(rank)]]
    require(len(selected) == 8, "ALL_MISMATCHED_FORWARD_PAIRS")
    selected.append((0,1))
    forward = [dict(rank=rank,ordinal=i,**compare_forward(files_by_rank[rank],i)) for rank,i in selected]
    updates = [dict(rank=0,step=step,**compare_update(files_by_rank[0],step)) for step in (1,2)]
    early_other_ranks = []
    for rank in (1,5):
        a, b = pair(files_by_rank[rank], "update-01-preclip.pt")
        early_other_ranks.append(dict(rank=rank,stats=differences(a["gradients"],b["gradients"])))
    require(not torch.cuda.is_initialized(), "CPU_READER_GPU_INIT")
    result = dict(original_GPU_steps=[row["gpu"][0] for row in binding["originals"]],
                  paired_same_manifest_inputs=64, actual_forward_pairs=len(forward),
                  paired_update_payload_ranks=[0], additionally_compared_preclip_step1_ranks=[1,5],
                  forward=forward, updates=updates, early_other_ranks=early_other_ranks,
                  raw_tensor_bytes_transferred=0, GPU_cards=0, no_training_replay=True)
    with (OUT/"compute.json").open("x",encoding="utf8") as stream:
        json.dump(result,stream,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
