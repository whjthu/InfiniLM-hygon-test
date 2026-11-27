"""Repack the original .4282 cache tensor views without loading or re-running the text model."""

import argparse
import io
import json
from pathlib import Path
import sys
import time


ORIGINAL_ARTIFACT = (4296921, "ae18b701a130899f66da7c736b913d547cea14e1c0b4d6087b780ff4ac570c2b")
ORIGINAL_WORKER = "80a7c4b3578378b4e7168f00480522a3f7cce1e535e2336730550b05e3483e3e"
ORIGINAL_START = "69b03f41adc55b08405aa50350a857eb3c2098b8fabcef7fe2ebd9dfaeb0bc80"
ORIGINAL_LOAD = "ca8ba050fff10725167d6952dd7df5626c68eaf028b65945f02d38f73bf021be"
ORIGINAL_STEP = "1103606.4282"


def recover(args):
    from hashlib import sha256

    original_source = args.original / "src"
    assert sha256((original_source / "build_cache.py").read_bytes()).hexdigest() == ORIGINAL_WORKER
    assert sha256((original_source / "native_text.py").read_bytes()).hexdigest() == (
        "8293cd57a962a2a374ea5969996844b193785f5910deabfc8cb07ba877e42731")
    sys.path.insert(0, str(original_source))
    from build_cache import NATIVE_TENSORS, check_tensors, exact_cpu_step, metadata, write_json
    from native_text import CAPTION, FRAME_RECEIPT, digest, pinned, require

    step = exact_cpu_step(args.request)
    begin = time.monotonic()
    require(args.out.parent.name == args.request and args.out.is_absolute(), "EXACT_PRIVATE_RECOVERY_SCOPE")
    args.out.mkdir(mode=0o700)
    previous = args.original / "cache-output"
    source_start = json.loads(pinned(previous / "start.json", 16384, ORIGINAL_START))
    source_load = json.loads(pinned(previous / "loaded.json", 16384, ORIGINAL_LOAD))
    require(source_start["actual_step"] == source_load["actual_step"] == ORIGINAL_STEP
            and source_start["request"] == source_load["request"] == args.original.name,
            "ORIGINAL_PRESERVED_CACHE_LINEAGE")
    original = pinned(previous / "sample.private.pt", *ORIGINAL_ARTIFACT, exact_size=ORIGINAL_ARTIFACT[0])
    import torch

    require(torch.cuda.device_count() == 0, "EXACT_RECOVERY_CPU_ONLY")
    torch.set_num_threads(2)
    sample = torch.load(io.BytesIO(original), map_location="cpu", weights_only=True)
    extra = metadata(args.render)
    require(set(sample) == set(NATIVE_TENSORS) | set(extra)
            and {name: sample[name] for name in extra} == extra, "EXACT_OLD_CACHE_METADATA")
    old_views = {name: sample[name] for name in NATIVE_TENSORS}
    before = check_tensors(old_views, torch)
    require(old_views["text_emb"].storage_offset() == 387072
            and old_views["text_emb"].untyped_storage().nbytes() == 2924544
            and old_views["text_mask"].storage_offset() == 108
            and old_views["text_mask"].untyped_storage().nbytes() == 3264,
            "ORIGINAL_TWO_TEXT_VIEWS_OF_FULL_STORAGE")
    new_values = {name: value.clone(memory_format=torch.contiguous_format) for name, value in old_views.items()}
    require(check_tensors(new_values, torch) == before and
            all(value.storage_offset() == 0 and value.untyped_storage().nbytes() == value.numel() * value.element_size()
                for value in new_values.values()), "EXACT_NEW_OWNED_STORAGE_AND_VIEW_VALUE_DIGESTS")
    output = args.out / "sample.private.pt"
    with output.open("xb") as stream:
        torch.save({**new_values, **extra}, stream)
    raw = output.read_bytes()
    require(0 < len(raw) <= 4 << 20, "COMPACT_CACHE_SIZE")
    reloaded = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    require(check_tensors({name: reloaded[name] for name in NATIVE_TENSORS}, torch) == before
            and {name: reloaded[name] for name in extra} == extra, "REPACKED_CACHE_CONTENT_READBACK")
    report = dict(request=args.request, actual_step=step, actual_GPU=0, actual_optimizer_updates=0,
                  actual_native_text_forwards=0, upstream_native_text_forwards=1,
                  recovered_from_original_step=ORIGINAL_STEP, recovery_method="storage_clone_of_verified_original_views",
                  original_serialized_cache=dict(bytes=ORIGINAL_ARTIFACT[0], SHA256=ORIGINAL_ARTIFACT[1]),
                  frame_receipt_sha256=FRAME_RECEIPT, caption_sha256=digest(CAPTION.encode()),
                  native_candidate_cache_rows=1, qualified_training_rows=0, formal_domain_delivery_increment=0,
                  native_fields=before, pt_cache=dict(bytes=len(raw), SHA256=digest(raw)),
                  model_class=source_load["model_class"], upstream_model_load_seconds=source_load["model_load_seconds"],
                  wall_seconds=time.monotonic()-begin)
    write_json(args.out / "terminal.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--render", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(recover(p.parse_args()), sort_keys=True))
