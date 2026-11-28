"""Observe the actual Swift Arrow and DataLoaderShard continuation order without a model."""

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import SimpleNamespace

from contract import EPOCHS, INPUTS, TENSORS, TRAIN, assemble, checked, checkpoint, digest, encode, need


SWIFT = Path("/swift-runtime/site/swift")
FILES = {
    "dataset/loader.py": "bfd75ce24bf0b2ba2fe414a37afd1ce2465575269689325791481eb2bef6d9b4",
    "pipelines/train/sft.py": "024ff2ff4e6755a07c30bcc3dd081106816b31fbb30169d9e27125f0b67be7d0",
    "dataloader/shard.py": "ad05502b587d37eb4de7d7cd61f5f652eb7751403320f2b18524d21b7fbd340c",
    "trainers/mixin.py": "6b6de2e7503ce414cb837961e1355be41a56787ecca502e8570bfcfa36c15ab9",
}
ORIGINAL_CLI = "849e72c9dde3d3e4a7bd40a4c1ebc700854af944fa6fbf86317f8447f55f7c9b"
MOUNTS = Path("/llava-cpu-authority")
MEDIA = Path("/llava-train-only")


def exclusive(path, value):
    with path.open("xb") as stream:
        stream.write(encode(value))
        stream.flush()
        os.fsync(stream.fileno())


def source_rows(paths=MOUNTS, data=MEDIA, reread_media=False):
    source = checked(data / "train.jsonl", TRAIN, length=20635)
    binding = json.loads(checked(paths / "INPUTS.json", INPUTS))
    witness = json.loads(checked(paths / "TENSORS.json", TENSORS, length=17260))
    need(binding["train_sha256"] == witness["train_sha256"] == TRAIN and witness["binding_sha256"] == INPUTS
         and len(source.splitlines()) == len(binding["rows"]) == len(witness["rows"]) == 64,
         "EXACT_ORIGINAL_64_AND_4878")
    rows, lineage, transformed = [], {}, []
    for i, (line, media, seen) in enumerate(zip(source.splitlines(), binding["rows"], witness["rows"], strict=True), 1):
        value = json.loads(line)
        need(set(value) == {"messages", "images"} and value["images"] == [media["path"]]
             and media["ordinal"] == seen["ordinal"] == i and media["row_sha256"] == seen["row_sha256"] == digest(line)
             and seen["image_sha256"] == media["sha256"] and re.fullmatch(r"[0-9a-f]{64}", seen["input_sha256"])
             and re.fullmatch(r"images/CLEVR_train_[0-9]{6}\.png", media["path"]), "EXACT_SOURCE_MEDIA_AND_TENSOR_ROW")
        if reread_media:
            checked(data / media["path"], media["sha256"], length=media["bytes"], retain=False)
        resolved = dict(messages=value["messages"], images=[str(data / media["path"])])
        rows.append(resolved)
        transformed.append(encode(resolved))
        lineage[i] = dict(ordinal=i, row_sha256=seen["row_sha256"], image_sha256=seen["image_sha256"],
                          input_sha256=seen["input_sha256"])
    for field in ("row_sha256", "image_sha256", "input_sha256"):
        need(len({row[field] for row in lineage.values()}) == 64, "EXACT_UNIQUE_ORIGINAL_CLEVR_ROWS")
    return rows, lineage, b"".join(transformed)


def native_cli(dataset, processor, output):
    original = json.loads(checked(MOUNTS / "original-launch.json", ORIGINAL_CLI))
    args = original["command"][2:]
    for flag, value in (("--model", processor), ("--dataset", dataset), ("--output_dir", output)):
        need(args.count(flag) == 1, "EXACT_EXISTING_LLAVA_CLI")
        args[args.index(flag) + 1] = str(value)
    index = args.index("--external_plugins")
    need(args[index + 3] == "--callbacks" and args[index + 4] == "nvidia_baseline", "EXACT_ORIGINAL_OBSERVER_ARGUMENTS")
    del args[index:index + 5]
    need("--use_cpu" not in args, "NO_INHERITED_CPU_OVERRIDE")
    return args + ["--use_cpu", "true"]


def load_swift(dataset, processor, output):
    from copy import copy

    for rel, pin in FILES.items():
        checked(SWIFT / rel, pin, cap=256 << 10)
    sft = importlib.import_module("swift.pipelines.train.sft")
    pipeline = object.__new__(sft.SwiftSft)
    parsed = pipeline._parse_args(native_cli(dataset, processor, output))
    training = parsed.training_args
    need(tuple(parsed.dataset) == (str(dataset),) and not parsed.val_dataset and parsed.split_dataset_ratio == 0
         and parsed.torch_dtype in ("bfloat16", "bf16", importlib.import_module("torch").bfloat16)
         and training.use_cpu is True and training.bf16 is True and parsed.gradient_checkpointing is False
         and parsed.gradient_accumulation_steps == 4 and parsed.per_device_train_batch_size == 1
         and parsed.max_steps == 25 and parsed.data_seed == parsed.seed == 42
         and parsed.dataset_shuffle is parsed.train_dataloader_shuffle is True and parsed.dataloader_num_workers == 4
         and parsed.packing is parsed.padding_free is parsed.dataloader_drop_last is parsed.group_by_length is False,
         "EXACT_NATIVE_LLAVA_TRAIN_OPTIONS_EXCEPT_DIAGNOSTIC_CPU")
    result = []
    for shuffled in (False, True):
        local = copy(parsed)
        local.dataset_shuffle = shuffled
        pipeline.args = local
        train, validation = pipeline._get_dataset()
        need(train is not None and validation is None and len(train) == 64, "ONLY_64_ACTUAL_SWIFT_ARROW_TRAIN_ROWS")
        result.append(train)
    return parsed, result


def projected_pair(original, shuffled, expected, dataset):
    need(original._indices is None and shuffled._indices is not None
         and shuffled._indices.table.column_names == ["indices"], "ACTUAL_PRE_AND_POST_SHUFFLE_ARROW")
    physical = original._data.table
    other = shuffled._data.table
    need(physical.num_rows == other.num_rows == 64 and set(physical.column_names) == {"messages", "images", "dataset"}
         and physical.equals(other, check_metadata=True), "IDENTICAL_NATIVE_PHYSICAL_ARROW_TABLES")
    order = shuffled._indices.table.column("indices").to_pylist()
    need(all(type(i) is int for i in order) and sorted(order) == list(range(64)), "EXACT_NATIVE_SHUFFLE_PERMUTATION")
    plain = original.select_columns(["messages"]).with_format(None)
    formatted = shuffled.select_columns(["messages"]).with_format(None)
    for logical, i in enumerate(order):
        raw = physical.column("messages")[i].as_py()
        if type(raw) is list and all(type(cell) is str for cell in raw):
            raw = [json.loads(cell) for cell in raw]
        images = physical.column("images")[i].as_py()
        if type(images) is list and len(images) == 1 and type(images[0]) is dict:
            need(set(images[0]) == {"bytes", "path"} and images[0]["bytes"] is None, "ARROW_PATH_ONLY_PNG")
            images = [images[0]["path"]]
        need(raw == plain[i]["messages"] == formatted[logical]["messages"] == expected[i]["messages"]
             and images == expected[i]["images"] and physical.column("dataset")[i].as_py() == str(dataset),
             "ACTUAL_ALL_64_PHYSICAL_AND_LOGICAL_4878_SOURCE_VALUES")
    return order, formatted


class NativeMetadataView:
    def __init__(self, formatted, order, expected, lineage):
        self.formatted, self.order, self.expected, self.lineage = formatted, order, expected, lineage

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        need(type(index) is int and 0 <= index < len(self), "VALID_NATIVE_LOGICAL_INDEX")
        physical = self.order[index]
        need(self.formatted[index]["messages"] == self.expected[physical]["messages"],
             "NATIVE_TRAVERSAL_FETCHES_THE_EXPECTED_ORIGINAL_ROW")
        return self.lineage[physical + 1]


def cpu_rank(root, processor, scratch, rendezvous):
    import datetime

    def forbid_weight_body(event, objects):
        if event == "open" and objects and isinstance(objects[0], (str, bytes)):
            path = os.fsdecode(objects[0])
            need(not path.endswith((".safetensors", ".pt", ".pth")), "NO_CHECKPOINT_OR_MODEL_DESERIALIZATION")

    need(all(os.environ.get(key) == "1" for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE")),
         "EXPLICIT_OFFLINE_CPU_RANK")
    sys.addaudithook(forbid_weight_body)
    import torch
    from swift.trainers.mixin import DataLoaderMixin
    from swift.dataloader.shard import BatchSamplerShard, DataLoaderShard

    headers = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    need(all(os.environ.get(key) == "-1" for key in headers) and not torch.cuda.is_initialized()
         and os.environ.get("GLOO_SOCKET_IFNAME") == "lo"
         and int(os.environ["WORLD_SIZE"]) == 8 and rendezvous.parent.is_relative_to(Path("/dev/shm"))
         and rendezvous.parent.resolve(strict=True) == rendezvous.parent
         and rendezvous.parent.stat().st_uid == os.getuid()
         and stat.S_IMODE(rendezvous.parent.stat().st_mode) == 0o700,
         "EIGHT_REAL_CPU_RANKS_WITH_LOCAL_FILESTORE")
    rank = int(os.environ["RANK"])
    need(0 <= rank < 8 and rank == int(os.environ["LOCAL_RANK"]), "ACTUAL_CPU_RANK_ENVIRONMENT")
    torch.cuda._lazy_init = lambda *a, **kw: need(False, "NO_GPU_INITIALIZATION")
    dist = torch.distributed
    dist.init_process_group("gloo", init_method=rendezvous.as_uri(), world_size=8, rank=rank,
                            timeout=datetime.timedelta(seconds=100))
    result = None
    try:
        need((dist.get_rank(), dist.get_world_size()) == (rank, 8), "ACTUAL_LIVE_CPU_RANK")
        expected, lineage, raw = source_rows()
        dataset = root / "train-bound.jsonl"
        checked(dataset, digest(raw), len(raw))
        parsed, (plain, shuffled) = load_swift(dataset, processor, scratch / f"unused-{rank}")
        order, formatted = projected_pair(plain, shuffled, expected, dataset)
        meta = SimpleNamespace(dataloader_num_workers=0, dataloader_pin_memory=False, dataloader_prefetch_factor=None,
                               dataloader_persistent_workers=False, dataloader_drop_last=parsed.dataloader_drop_last,
                               train_dataloader_shuffle=parsed.train_dataloader_shuffle, data_seed=parsed.data_seed,
                               deepspeed=parsed.deepspeed, group_by_length=parsed.group_by_length, process_index=dist.get_rank())
        view = NativeMetadataView(formatted, order, expected, lineage)
        holder = SimpleNamespace(template=SimpleNamespace(sequence_parallel_size=1), train_dataset=view,
                                 _train_batch_size=1, data_collator=lambda rows: rows,
                                 accelerator=SimpleNamespace(device=torch.device("cpu")), args=meta)
        loader = DataLoaderMixin.get_train_dataloader(holder, skip_batches=0)
        sampler = loader.batch_sampler
        need(type(loader) is DataLoaderShard and type(sampler) is BatchSamplerShard and len(loader) == len(sampler) == 8
             and (sampler.rank, sampler.world_size, sampler.base_seed, sampler.total_samples)
             == (rank, 8, 42, 8), "ACTUAL_NO_SKIP_ORIGINAL_NATIVE_SAMPLER")
        epochs = []
        for epoch in EPOCHS:
            loader.set_epoch(epoch)
            need(loader.batch_sampler is sampler and sampler.curr_seed == 42 + epoch, "ACTUAL_NATIVE_EPOCH_TRANSITION")
            rows = []
            for batch in loader:
                need(type(batch) is list and len(batch) == 1 and type(batch[0]) is dict
                     and batch[0] == lineage[batch[0]["ordinal"]], "NATIVE_ONE_OBSERVED_PHYSICAL_OCCURRENCE")
                rows.append(batch[0])
            need(len(rows) == 8, "NATIVE_EIGHT_ACTUAL_CPU_BATCHES")
            epochs.append(dict(epoch=epoch, seed=sampler.curr_seed, rows=rows))
        result = dict(rank=rank, arrow_order_SHA256=digest(encode(order)), epochs=epochs,
                      metadata_workers=0, production_workers=4, no_skip_native=True,
                      dataset_SHA256=digest(raw), actual_model_forwards=0)
    finally:
        dist.destroy_process_group()
    exclusive(root / f"rank-{rank}.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "rank", "merge"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--processor", type=Path)
    p.add_argument("--scratch", type=Path)
    p.add_argument("--rendezvous", type=Path)
    args = p.parse_args()
    root = args.output.resolve(strict=True)
    need(root.stat().st_uid == os.getuid() and stat.S_IMODE(root.stat().st_mode) == 0o700,
         "PRIVATE_OWNER_OUTPUT")
    if args.mode == "prepare":
        state = checkpoint(MOUNTS / "4983.json", args.checkpoint)
        _, _, raw = source_rows(reread_media=True)
        with (root / "train-bound.jsonl").open("xb") as file:
            file.write(raw)
        exclusive(root / "PREPARED.json", dict(cursor=state, dataset_SHA256=digest(raw), cp2_members_verified=17))
    elif args.mode == "rank":
        try:
            cpu_rank(root, args.processor, args.scratch, args.rendezvous)
        except Exception as error:
            rank = int(os.environ["RANK"])
            need(0 <= rank < 8, "BOUNDED_RANK_FAILURE")
            exclusive(root / f"rank-{rank}-error.json", dict(rank=rank, error_type=type(error).__name__,
                      error_SHA256=digest(str(error).encode("utf-8", "replace")), planned_forward_slots=736))
            raise SystemExit(2) from None
    else:
        _, expected, raw = source_rows()
        prepared = json.loads((root / "PREPARED.json").read_bytes())
        reports = [json.loads((root / f"rank-{i}.json").read_bytes()) for i in range(8)]
        need(prepared["cp2_members_verified"] == 17 and prepared["dataset_SHA256"] == digest(raw)
             and len({r["arrow_order_SHA256"] for r in reports}) == 1
             and all(r["rank"] == i and r["dataset_SHA256"] == digest(raw) and r["no_skip_native"] is True
                     for i, r in enumerate(reports)), "ALL_EIGHT_NATIVE_ORDER_REPORTS")
        plan = assemble([r["epochs"] for r in reports], expected)
        need(plan["start"] == prepared["cursor"], "OBSERVED_PLAN_ORIGINAL_CP2_CURSOR")
        exclusive(root / "CP2_NATIVE_RESUME_PLAN.private.json", plan)
        print(json.dumps(dict(planned_forward_slots=736, first_update=3, start_epoch=1, intra_epoch_skip=0,
                              planned_updates=23, native_CPU_metadata_only=True,
                              plan_SHA256=digest(encode(plan))), sort_keys=True))


if __name__ == "__main__":
    main()
