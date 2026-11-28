"""Exercise exact source and command boundaries without importing Torch."""

import argparse
import ast
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import run


def function(source, name):
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def require_rejected(callback, expected):
    try:
        callback()
    except ValueError as error:
        assert str(error) == expected, str(error)
    else:
        raise AssertionError(expected)


def main(repo):
    local = repo / "local-only/training-sdxl-style18-public-candidate-cli-v1-original"
    original = (local / "candidate-source-root" / run.TRAINER).read_bytes()
    original_backend = (local / "sdxl_training_196501.py").read_bytes()
    trainer, backend = run.trainer_delta(original), run.backend_delta(original_backend)
    compile(trainer, "exact-derived-trainer", "exec")
    compile(backend, "exact-derived-backend", "exec")
    require_rejected(lambda: run.trainer_delta(original + b"\n"), "ORIGINAL_DIFFUSERS_TRAINER_SHA")
    require_rejected(lambda: run.backend_delta(original_backend + b"\n"), "ORIGINAL_COMMON_GENCUDA_BACKEND_SHA")
    # Execute just the original backend's command builder and the exact provider's parser.
    sys.path.insert(0, str(repo / "code/bw1000/training-sdxl-style18-public-candidate-cli-v1"))
    from check_interface_once import backend_command, parser_only

    cmd = backend_command(backend, run.CONFIG)
    assert backend_command(original_backend, run.CONFIG) == cmd
    args = parser_only(trainer)(cmd[2:])
    assert (args.max_train_steps, args.checkpointing_steps, args.lr_warmup_steps,
            args.learning_rate, args.rank, args.resolution, args.train_batch_size,
            args.gradient_accumulation_steps, args.lr_scheduler, args.mixed_precision) == (
                25, 1, 0, 1e-4, 4, 256, 1, 1, "constant", "bf16")
    main_fn = function(trainer, "main")
    calls = [node for node in ast.walk(main_fn) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)]
    def one(obj, method):
        matches = [node.lineno for node in calls if (node.func.value.id, node.func.attr) == (obj, method)]
        assert len(matches) == 1, (obj, method, matches)
        return matches[0]
    assert (one("witness", "before_forward") < one("accelerator", "backward")
            < one("witness", "before_step") < one("optimizer", "step") < one("witness", "after_step")
            < one("lr_scheduler", "step") < one("accelerator", "save_state"))
    assert {arg.arg for arg in next(node for node in calls
            if node.lineno == one("witness", "before_forward")).keywords} == {
                "pixels", "clean_latent", "noise", "noisy_latent", "timestep", "sequence", "pooled", "time_ids"}
    chest = repo / "local-only/sdxl-metal-tool-chest-256-domain-source-v1"
    actual = chest / "metal_tool_chest_0001.png"
    with Image.open(BytesIO(run.read_pinned(actual, run.INPUTS["tool_chest"][0], 41917, keep=True))) as image:
        assert (image.mode, image.size) == ("RGB", (256, 256))
        image.load()
    text = json.loads((chest / "metadata.jsonl").read_bytes())["text"].encode()
    assert run.INPUTS["tool_chest"][1] in {sha256(text + ending).hexdigest() for ending in (b"", b"\n", b"\r\n")}
    with TemporaryDirectory() as work:
        root = Path(work).resolve()
        raw = root / "raw"
        raw.mkdir()
        fake = {}
        for index, role in enumerate(run.INPUTS):
            file = raw / (role + ".png")
            Image.new("RGB", (256, 256), (index, 2, 3)).save(file)
            caption = f"test-{index}".encode()
            (raw / (role + ".txt")).write_bytes(caption)
            fake[role] = sha256(file.read_bytes()).hexdigest(), sha256(caption).hexdigest()
        with patch.object(run, "INPUTS", fake):
            run.dataset(raw, root / "native-input")
            rows = [json.loads(line) for line in (root / "native-input/metadata.jsonl").read_text().splitlines()]
            assert rows == [{"file_name": role + ".png", "text": f"test-{i}"} for i, role in enumerate(fake)]
            (raw / "tool_chest.txt").write_bytes(b"changed")
            require_rejected(lambda: run.dataset(raw, root / "never-made"), "PIN_OR_IDENTITY_CHANGED")
        (root / "missing.json").write_bytes(b"{}")
        require_rejected(lambda: run.assets(root, root / "missing.json", sha256(b"{}").hexdigest()),
                         "EXACT_OFFICIAL_PIPELINE_CLOSURE")
    require_rejected(lambda: run.launch(SimpleNamespace()), "NO_SINGLE_DCU_OWNER_ADMISSION")
    print(json.dumps(dict(exact_original_sources=2, actual_provider_cli_parser=True,
                          actual_public_RGB_objects_loaded=1, local_imagefolder_fixture_rows=2,
                          model_loads=0, actual_optimizer_updates=0, GPU_calls=0), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    main(parser.parse_args().repo.resolve())
