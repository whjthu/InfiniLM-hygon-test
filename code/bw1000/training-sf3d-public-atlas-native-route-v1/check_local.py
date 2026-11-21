"""Freeze real train-only inputs and exercise the rendered framework before CUDA."""

import ast
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from render import CLI, DERIVATIVE, INPUT, REPO, checked, pin, write_overlay


BINDING = REPO / "local-only/training-sf3d-public-corset-train-atlas-v1-original/three-object-binding-private.json"
OUTPUT = REPO / "local-only/training-sf3d-public-atlas-native-route-v1-original"


def prepare_train(output):
    combined = json.loads(checked(BINDING.read_bytes(), "54ecd851d91d48359be528d93632824f1ad7d553c257ef783df95a38991ba8e7"))
    if [r["group"] for r in combined["train"]] != ["WaterBottle", "Corset"]:
        raise ValueError("wrong original training order")
    output.mkdir(mode=0o700)
    rows = []
    for origin in combined["train"]:
        row = {"group": origin["group"]}
        for target, predecessor, suffix in (("image", "input", "png"), ("target", "target", "npz")):
            source = (REPO / origin[predecessor]).resolve(strict=True)
            if not source.is_relative_to(REPO / "local-only"):
                raise ValueError("original input outside private run")
            original = origin[predecessor + "_pin"]
            raw = checked(source.read_bytes(), original["SHA256"])
            if len(raw) != original["bytes"]:
                raise ValueError("unexpected original train data size")
            name = row["group"] + "." + suffix
            (output / name).write_bytes(raw)
            row.update({target: name, target + "_pin": original})
        rows.append(row)
    plan = dict(recipe="sf3d-public-pbr-UV-atlas-v1", train=rows)
    return config_for(output, plan)


def config_for(root, plan, file_name="train-manifest.json"):
    path = root / file_name
    raw = (json.dumps(plan, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return dict(sf3d_atlas=dict(manifest=str(path), **pin(raw)),
                input_images=[str(root / item["image"]) for item in plan["train"]])


def exact_function(source, name, scope):
    node, = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    flags = __import__("__future__").annotations.compiler_flag
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec", flags=flags), scope)
    return scope[name]


class CUDAEntryReached(Exception):
    pass


def main():
    OUTPUT.mkdir(mode=0o700)
    overlay = OUTPUT / "overlay"
    evidence = write_overlay(overlay)
    train = OUTPUT / "train"
    config = prepare_train(train)
    package = ModuleType("_sf3d_atlas_candidate")
    package.__path__ = [str((overlay / INPUT).parent)]
    sys.modules[package.__name__] = package

    called = []
    from importlib import import_module
    inputs = import_module(package.__name__ + ".atlas_inputs")
    original_input = inputs.load_atlas

    def observed_input(cfg):
        loaded = original_input(cfg)
        called.append(loaded)
        return loaded

    def cuda_boundary():
        raise CUDAEntryReached

    inputs.load_atlas = observed_input
    candidate = exact_function((overlay / DERIVATIVE).read_text("utf-8"), "run_sf3d_training",
                               dict(__package__=package.__name__, os=SimpleNamespace(environ={}),
                                    require_cuda=cuda_boundary))
    routes = []

    def observe_route(module, name, *args, **kwargs):
        routes.append(module)
        if module.endswith(".sf3d_atlas_training"):
            return candidate(*args, **kwargs)
        return "original backend selected"

    cli = (overlay / CLI).read_text("utf-8")
    resume = exact_function(cli, "validate_resume_request", {})
    dispatch = exact_function(cli, "run_backend", dict(_call=observe_route,
                              validate_resume_request=resume, _source_dir=lambda root, name: OUTPUT / name))
    spec = SimpleNamespace(name="stable-fast-3d")
    original_result = dispatch("train", spec, OUTPUT, {}, OUTPUT, OUTPUT)
    assert original_result == "original backend selected" and not called
    try:
        dispatch("train", spec, OUTPUT, config, OUTPUT, OUTPUT)
    except CUDAEntryReached:
        pass
    else:
        raise AssertionError("native atlas entry did not reach the specified boundary")
    assert routes == ["gencuda.backends.sf3d_training", "gencuda.backends.sf3d_atlas_training"]
    atlas, = called
    assert atlas["labels_sha256"] == "e6f401c36d52361aa776c65a498bc70a59bd1e6b45209a95d77f043afb153e78"
    assert len(atlas["images"]) == len(atlas["targets"]) == 2
    base = "a" * 64
    domain = inputs.atlas_config_sha256(base, atlas)
    assert domain != base
    assert domain != inputs.atlas_config_sha256(base, dict(atlas, labels_sha256="b" * 64))

    plan = json.loads(Path(config["sf3d_atlas"]["manifest"]).read_bytes())
    invalid_split = config_for(train, dict(plan, development=["BoomBox"]), "bad-split.json")
    invalid_order = copy.deepcopy(config)
    invalid_order["input_images"].reverse()
    corrupted_pin = copy.deepcopy(config)
    corrupted_pin["sf3d_atlas"]["SHA256"] = "0" * 64
    for invalid in (invalid_split, invalid_order, corrupted_pin):
        try:
            dispatch("train", spec, OUTPUT, invalid, OUTPUT, OUTPUT)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid label identity crossed the CUDA boundary")
    evidence.update(actual_private_train_objects=2, actual_development_objects_read=0,
                    strict_rejections=3, actual_original_route_checks=1, actual_domain_native_cuda_boundary=1,
                    labels_sha256=atlas["labels_sha256"], train_manifest=pin(Path(config["sf3d_atlas"]["manifest"]).read_bytes()),
                    actual_target_installations=0, actual_CUDA_invocations=0, actual_optimizer_updates=0,
                    training_ready_added=0)
    with (OUTPUT / "result.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(evidence, f, indent=2)
        f.write("\n")
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
