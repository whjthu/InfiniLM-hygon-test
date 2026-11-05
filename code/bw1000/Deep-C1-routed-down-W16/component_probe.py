"""Load fixed real C1 inputs and compare the W32/W16 routed component."""

from collections import defaultdict
from hashlib import sha256
import json
import math
from statistics import mean

import numpy as np

FIXTURE_SHA = "de85e0806699b422cae9cfa1beed43c84598c684a2c0ea9236dbcea29f11a6f8"
INDEX_SHA = "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"
PASTS = (8570, 8571)
ARMS = (0, 1, 0, 1, 1, 0, 1, 0, 0, 1)
DTYPES = {"BF16": "<u2", "I64": "<i8", "F32": "<f4"}


def digest(raw):
    return sha256(raw).hexdigest()


def read_fixtures(directory):
    raw = (directory / "FIXTURE_MANIFEST.json").read_bytes()
    if digest(raw) != FIXTURE_SHA:
        raise ValueError("unexpected C1 fixture")
    by_rank = {rank: [] for rank in range(8)}
    for row in json.loads(raw)["rows"]:
        if row["layer"] != 3:
            raise ValueError("unexpected model layer")
        arrays = []
        for name in ("hidden", "route_ids", "route_weights"):
            field = row["fields"][name]
            data = (directory / field["file"]).read_bytes()
            if len(data) != field["bytes"] or digest(data) != field["SHA256"]:
                raise ValueError(f"fixture operand differs: {name}")
            arrays.append(np.frombuffer(data, DTYPES[field["dtype"]]).reshape(field["shape"]).copy())
        by_rank[row["rank"]].append((row["past"], arrays))
    if any(tuple(past for past, _ in rows) != PASTS for rows in by_rank.values()):
        raise ValueError("incomplete rank fixture")
    return by_rank


def expert_plan(model, rank_rows):
    raw = (model / "model.safetensors.index.json").read_bytes()
    if digest(raw) != INDEX_SHA:
        raise ValueError("unexpected model index")
    mapping = json.loads(raw)["weight_map"]
    experts = {int(v) for _, arrays in rank_rows for v in arrays[1].ravel()}
    if len(experts) != 12 or any(not 0 <= expert < 256 for expert in experts):
        raise ValueError("unexpected expert set")
    groups = defaultdict(list)
    for expert in sorted(experts):
        for projection in ("w1", "w3", "w2"):
            for field in ("weight", "scale"):
                key = f"layers.3.ffn.experts.{expert}.{projection}.{field}"
                path = (model / mapping[key]).resolve()
                if path.parent != model.resolve():
                    raise ValueError("checkpoint shard escapes the model directory")
                groups[path].append((key, expert, projection, field))
    return groups


def rank_slice(value, projection, field, rank):
    local = slice(rank * 256, (rank + 1) * 256)
    if projection != "w2":
        return value[local]
    return value[:, local] if field == "weight" else value


def load_bank(groups, rank, torch, safe_open):
    banks = [torch.zeros(shape, dtype=dtype) for shape, dtype in (
        ((256, 512, 4096), torch.int8), ((256, 512, 1), torch.float32),
        ((256, 4096, 256), torch.int8), ((256, 4096, 1), torch.float32),
    )]
    proofs = {}
    for path, keys in sorted(groups.items()):
        with safe_open(path, framework="pt", device="cpu") as shard:
            for key, expert, projection, field in keys:
                is_weight = field == "weight"
                dtype = torch.int8 if is_weight else torch.float32
                shape = ((4096, 2048 if is_weight else 1) if projection == "w2"
                         else (2048, 4096 if is_weight else 1))
                tensor = shard.get_tensor(key)
                if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(f"checkpoint geometry differs: {key}")
                value = rank_slice(tensor, projection, field, rank).contiguous()
                if not is_weight and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"non-finite checkpoint scale: {key}")
                if projection == "w2":
                    target = banks[2 if is_weight else 3][expert]
                else:
                    start = 0 if projection == "w1" else 256
                    target = banks[0 if is_weight else 1][expert, start:start + 256]
                target.copy_(value)
                proofs[key] = digest(value.numpy().tobytes())
    if len(proofs) != 72:
        raise ValueError("incomplete weight load")
    return banks, digest(json.dumps(proofs, sort_keys=True, separators=(",", ":")).encode())


def compare(forward, records=None):
    reference = {}
    records = [] if records is None else records
    for trial, mode in enumerate(ARMS):
        for index, past in enumerate(PASTS):
            native, output, capi = forward(index, mode)
            if (native["mode"], native["down_width"]) != (mode, (32, 16)[mode]):
                raise ValueError("wrong component mode")
            routed = native["whole_routed_return_seconds"]
            drain = native["routed_plus_outputcopy_drain_seconds"]
            if not (math.isfinite(capi) and 0 < routed <= drain <= capi):
                raise ValueError("invalid nested timing")
            if output.shape != (1, 4096) or output.dtype != np.float32 or not np.isfinite(output).all():
                raise ValueError("invalid routed output")
            if trial == 0:
                reference[past] = output.copy()
            baseline = reference[past]
            if not np.allclose(output, baseline, rtol=1e-5, atol=1e-5):
                raise ValueError("routed output differs from W32")
            records.append(dict(trial=trial, mode=mode, past=past, routed_s=routed,
                                drain_s=drain, capi_s=capi,
                                exact=output.tobytes() == baseline.tobytes(),
                                output_SHA256=digest(output.tobytes())))
    return records


def summarize(owners):
    if set(owners) != set(range(8)) or any(len(rows) != len(ARMS)*len(PASTS) for rows in owners.values()):
        raise ValueError("incomplete comparison")
    result = {}
    for past in PASTS:
        matches = [[r for r in owners[rank] if r["past"] == past] for rank in range(8)]
        if any([(r["trial"], r["mode"]) for r in rows] != list(enumerate(ARMS)) for rows in matches):
            raise ValueError("unpaired comparison")
        comparisons = []
        for start in (2, 6):
            stats = {}
            for metric in ("routed_s", "drain_s", "capi_s"):
                values = [max(rows[i][metric] for rows in matches) for i in range(start, start + 4)]
                means = [mean(v for i, v in enumerate(values) if ARMS[start+i] == mode)
                         for mode in (0, 1)]
                stats[metric] = {"W32_mean_s": means[0], "W16_mean_s": means[1],
                                 "W16_over_W32": means[1] / means[0]}
            comparisons.append(stats)
        result[str(past)] = comparisons
    return result
