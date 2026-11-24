"""Render frozen COCO development inputs with the original and trained adapters."""

from hashlib import sha256
import json
import os
from pathlib import Path

from freeze_zimage_coco_development_noise import DEVELOPMENT_COUNT, Replay, ROSTER_SHA256
from render_zimage_coco_bound_input import render_bound_input, require


PROMPTS_SHA256 = "0251ad4ad34b50ce2c8e66d41548e6b42603a8b932d39210c444ab599eebea8b"
NOISE_SHA256 = "5c4cb6303071502ebdee928dbd13df10678143dbfba1b8e38086ba832a33c4c4"
CHECKPOINT_SHA256 = "0c27c2e9c5b7359acccdffdeeecb995ab9614a7fb39cd253c712645dc70c7668"
ADAPTER_SHA256 = "98ad7562fec41994407f7fd6cba450e8a33fe9dbf00c3f22638d77d8530ce665"
CHECKPOINT_BYTES = 60_424_609
ARMS = ("before", "after")


def digest(value):
    return sha256(value).hexdigest()


def file_digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_inputs(prompt_path, noise_manifest_path, roster_path):
    raw = prompt_path.read_bytes()
    require(digest(raw) == PROMPTS_SHA256, "development prompt manifest differs")
    document = json.loads(raw)
    require(document["schema"] == "training.zimage.coco.development_prompts.v1"
            and document["roster_SHA256"] == ROSTER_SHA256 and document["cohort"] == "development",
            "development cohort identity differs")
    replay = Replay(noise_manifest_path, NOISE_SHA256, roster_path)
    roster_raw = roster_path.read_bytes()
    require(digest(roster_raw) == ROSTER_SHA256, "development roster changed")
    roster = json.loads(roster_raw)["subsets"]["development"]
    rows = document["rows"]
    require(len(rows) == len(roster) == DEVELOPMENT_COUNT
            and len({row["image_id"] for row in rows}) == DEVELOPMENT_COUNT,
            "development denominator differs")
    for ordinal, (row, expected) in enumerate(zip(rows, roster, strict=True)):
        require(row["ordinal"] == ordinal
                and all(row[key] == expected[key] for key in
                        ("image_id", "caption_id", "prompt_SHA256", "seed"))
                and digest(row["prompt"].encode("utf-8")) == row["prompt_SHA256"],
                "development row differs from the published roster")
    return rows, replay


def load_adapters(model, torch, np, binding, checkpoint_path):
    """Read the source-bound Hygon initial state and the completed COCO20 checkpoint."""
    from gencuda import hy_resume_checkpoint as common
    from gencuda import zimage_hip_replay_binding as hip
    from gencuda.zimage_native_record import validate_binding

    validate_binding(binding)
    require(binding["profile"] == hip.PROFILE and binding["platform"] == "hip",
            "expected the original Hygon COCO20 training identity")
    initial, _, rng_sha = hip.initial_state(torch, np, common, binding)
    require(rng_sha == binding["original_rng_SHA256"], "original initial RNG signature differs")
    require(checkpoint_path.stat().st_size == CHECKPOINT_BYTES
            and file_digest(checkpoint_path) == CHECKPOINT_SHA256, "step-25 checkpoint bytes differ")
    checkpoint = common.load_checkpoint(checkpoint_path)
    require(checkpoint["metadata"]["global_step"] == 25
            and checkpoint["metadata"]["next_data_cursor"] == 5, "completed checkpoint state differs")
    trained = checkpoint["state"]["trainable"]
    require(common.tree_sha256(trained) == ADAPTER_SHA256, "completed adapter values differ")
    named = {name: tensor for name, tensor in model.named_parameters() if tensor.requires_grad}
    require(set(named) == set(initial) == set(trained) and len(named) == 476
            and sum(t.numel() for t in named.values()) == 9_922_560, "trainable tensors differ")
    for name, target in named.items():
        require(target.device.type == "cuda"
                and all(state[name].shape == target.shape and state[name].dtype == target.dtype
                        for state in (initial, trained)), "adapter storage format differs")
    return named, {"before": initial, "after": trained}


def write_line(stream, row):
    stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def shard_plan(first, stop):
    require(type(first) is int and type(stop) is int and 0 <= first < stop <= DEVELOPMENT_COUNT,
            "invalid frozen development shard")
    return {"scope": "COCO development only", "cohort_pairs": DEVELOPMENT_COUNT,
            "first": first, "stop": stop, "planned_pairs": stop - first,
            "planned_outputs": 2 * (stop - first), "denoising_steps": 8, "guidance": 1.0,
            "source_checkpoint_SHA256": CHECKPOINT_SHA256}


def execute_pairs(*, pipe, torch, replay, rows, named, states, output, first, stop, deadline):
    """Render an exact contiguous shard without changing the 256-pair denominator."""
    require(len(rows) == DEVELOPMENT_COUNT, "invalid frozen development cohort")
    plan = shard_plan(first, stop)
    output.mkdir(mode=0o700)
    planned = plan["planned_outputs"]
    with (output / "plan.jsonl").open("x", encoding="utf-8") as manifest:
        write_line(manifest, plan)
    counts = {"completed": 0, "failed": 0, "complete_pairs": 0}
    with (output / "generations.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(first, stop):
            row = rows[index]
            require(row["ordinal"] == index, "shard ordinal differs")
            order = ARMS if index % 2 == 0 else ARMS[::-1]
            completed_pair = 0
            for arm in order:
                deadline()
                event = {"ordinal": index, "image_id": row["image_id"], "arm": arm, "completed": False,
                         "prompt_SHA256": row["prompt_SHA256"]}
                try:
                    with torch.no_grad():
                        for name, parameter in named.items():
                            parameter.copy_(states[arm][name])
                    image, proof = render_bound_input(pipe=pipe, torch=torch, noise_bank=replay,
                                                      entry=row, inference_steps=8, guidance=1.0)
                    torch.cuda.synchronize()
                    image = image.convert("RGB")
                    path = output / f"{index:03d}-{arm}.private.png"
                    with path.open("xb") as stream_image:
                        image.save(stream_image, format="PNG")
                        stream_image.flush()
                        os.fsync(stream_image.fileno())
                    event.update(proof, completed=True, image_relative=path.name,
                                 image_bytes=path.stat().st_size, image_SHA256=file_digest(path),
                                 RGB_SHA256=digest(image.tobytes()))
                except Exception as error:
                    counts["failed"] += 1
                    event["error_type"] = type(error).__name__
                    write_line(stream, event)
                    if isinstance(error, (RuntimeError, OSError)):
                        raise
                    continue
                counts["completed"] += 1
                completed_pair += 1
                write_line(stream, event)
            counts["complete_pairs"] += completed_pair == 2
    return {"planned_pairs": stop - first, "planned_outputs": planned,
            "missing": planned - counts["completed"] - counts["failed"], **counts}


def render(model, torch, np, *, binding, checkpoint, prompts, noise_manifest, roster, output,
           first, stop, deadline):
    """Use an already loaded and audited native model inside its bounded GPU step."""
    rows, replay = read_inputs(Path(prompts), Path(noise_manifest), Path(roster))
    deadline()
    named, states = load_adapters(model, torch, np, binding, Path(checkpoint))
    model.eval()
    return execute_pairs(pipe=model.pipe, torch=torch, replay=replay, rows=rows, named=named,
                         states=states, output=Path(output), first=first, stop=stop, deadline=deadline)
