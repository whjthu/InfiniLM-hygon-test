"""Exercise exact DiffSynth image operators with the reference torchvision PIL branch."""

import ast
from enum import Enum
import hashlib
import json
import numbers
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "local-only/training-zimage-USGS-style18-dataset-v1-original/dataset"
VISION = REPO / "local-only/ZImage-style18-vision-v023-reference-private-v1"
DIFF = REPO / "code/bw1000/training-zimage-target-native-meta-CPU-v1/upstream/diffsynth/core/data"
OUTPUT = REPO / "local-only/training-zimage-USGS-style18-vision-pil-v023-v1-original"
PINS = {
    "functional.py": (67861, "0b2fdf8e6764f4be0113acd95b58d3b14158c13fa2374ed93fbbe1f4b8730ded"),
    "functional_pil.py": (12118, "a30262323eefa9351540515b1ff78f218ecff2ce3f6ed957b271948cc8c9a276"),
    "LICENSE": (1517, "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"),
    "operators.py": (7735, "c02a73ebfc8bef63f175fe54882d0faf4224d7552ea852df75b27823a8f235a4"),
    "unified_dataset.py": (4972, "b6e0729fbd00deee8b0d03dd463df43ce7e653ecea3f75c18741d4cc03ac027f"),
    "data-binding-private.json": (14457, "2152981e81882985de57a9419a34f895440d7965cc3b277c762f15304bc51c35"),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, expected):
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("expected one canonical input file")
    size, sha = expected
    if path.stat().st_size != size:
        raise ValueError("unexpected input size")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or digest(raw) != sha:
        raise ValueError("unexpected input digest")
    return raw


def pinned(path):
    return checked(path, PINS[path.name])


def required(tree, names):
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("unexpected selected API closure")
    return selected


def execute(nodes, scope, name):
    flags = __import__("__future__").annotations.compiler_flag
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), name, "exec", flags=flags), scope)
    return scope


class NoTensor:
    def __new__(cls):
        raise AssertionError("the reference harness must stay in the PIL dispatch path")


class NeverTensorOperations:
    def __getattr__(self, name):
        raise AssertionError("unexpected Tensor operation")


def vision_pil(Image):
    # The real PIL branches are executed verbatim. Tensor work deliberately fails.
    pil = required(ast.parse(pinned(VISION / "functional_pil.py")),
                   {"_is_pil_image", "get_dimensions", "resize", "crop"})
    for node in pil:
        for decorator in node.decorator_list:
            if ast.unparse(decorator) != "torch.jit.unused":
                raise ValueError("unexpected PIL decorator")
        node.decorator_list = []
    pil_api = SimpleNamespace(**execute(pil, dict(Image=Image, accimage=None), "v0.23.0 PIL functions"))
    tree = ast.parse(pinned(VISION / "functional.py"))
    names = {"InterpolationMode", "_interpolation_modes_from_int", "get_dimensions",
             "_compute_resized_output_size", "resize", "crop", "center_crop"}
    public = required(tree, names)
    mappings = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(v, ast.Name) and v.id == "pil_modes_mapping" for v in node.targets)]
    if len(mappings) != 1:
        raise ValueError("missing original interpolation mapping")
    counts = {"resize": 0, "crop": 0}
    for method in counts:
        original = getattr(pil_api, method)

        def counted(*args, method=method, original=original, **kwargs):
            counts[method] += 1
            return original(*args, **kwargs)

        setattr(pil_api, method, counted)
    scope = dict(Enum=Enum, numbers=numbers, warnings=warnings, F_pil=pil_api,
                 F_t=NeverTensorOperations(),
                 torch=SimpleNamespace(Tensor=NoTensor,
                                       jit=SimpleNamespace(is_scripting=lambda: False, is_tracing=lambda: False)),
                 _log_api_usage_once=lambda _: None)
    execute([*public, *mappings], scope, "v0.23.0 torchvision public functions")
    return SimpleNamespace(transforms=SimpleNamespace(functional=SimpleNamespace(**scope),
                                                      InterpolationMode=scope["InterpolationMode"])), counts


def loader(Image, pandas, torchvision):
    operator_names = {"DataProcessingOperator", "DataProcessingPipeline", "LoadImage", "ImageCropAndResize",
                      "RouteByType", "ToAbsolutePath", "SequencialProcess", "LoadTorchPickle"}
    ops = required(ast.parse(pinned(DIFF / "operators.py")), operator_names)
    dataset, = required(ast.parse(pinned(DIFF / "unified_dataset.py")), {"UnifiedDataset"})
    wanted = {"__init__", "default_image_operator", "load_metadata", "__getitem__", "__len__"}
    dataset.body = required(dataset, wanted)
    dataset.bases = []
    scope = execute([*ops, dataset], dict(Image=Image, torchvision=torchvision, os=os, json=json, pandas=pandas),
                    "frozen DiffSynth CSV and image operators")
    return scope["UnifiedDataset"]


def main():
    import PIL
    from PIL import Image
    import pandas

    OUTPUT.mkdir(exist_ok=False)
    pinned(VISION / "LICENSE")
    binding = json.loads(pinned(DATA / "data-binding-private.json"))
    torchvision, counts = vision_pil(Image)
    constructor = loader(Image, pandas, torchvision)
    report, rows = {}, []
    for split, planned in (("train", 12), ("dev", 6)):
        root = DATA / split
        csv = root / "metadata.csv"
        csv_pin = binding["metadata"][split]
        checked(csv, (csv_pin["bytes"], csv_pin["SHA256"]))
        originals = [entry for entry in binding["images"] if entry["split"] == split]
        for entry in originals:
            image = DATA / entry["dataset_relative"]
            if not image.is_relative_to(root / "images"):
                raise ValueError("image is outside its frozen split")
            source = entry["source"]
            checked(image, (source["bytes"], source["SHA256"]))
        image_op = constructor.default_image_operator(base_path=str(root), height=256, width=256)
        dataset = constructor(base_path=str(root), metadata_path=str(csv), repeat=1,
                              data_file_keys=("image",), main_data_operator=image_op)
        ordered = [Path(entry["dataset_relative"]).relative_to(split).as_posix() for entry in originals]
        if len(dataset) != planned or [row["image"] for row in dataset.data] != ordered:
            raise ValueError("actual CSV order or denominator differs")
        for index in range(planned):
            entry = dataset[index]
            image = entry["image"]
            if image.mode != "RGB" or image.size != (256, 256):
                raise ValueError("actual DiffSynth output shape or mode differs")
            prompt = digest(entry["prompt"].encode("utf-8"))
            if prompt != binding["caption_SHA256"]:
                raise ValueError("actual prompt identity differs")
            rows.append(dict(split=split, ordinal=index, original_image=originals[index]["source"]["SHA256"],
                             output_RGB=digest(image.tobytes()), prompt=prompt))
        report[split] = {"actual_rows": planned, "actual_output_size": [256, 256], "metadata_SHA256": csv_pin["SHA256"]}
    if counts != {"resize": 18, "crop": 18} or len({row["output_RGB"] for row in rows}) != 18:
        raise ValueError("actual operation denominator or exact-pixel uniqueness differs")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise ValueError("unexpected full Torch runtime")
    payload = (json.dumps(rows, sort_keys=True) + "\n").encode("utf-8")
    with (OUTPUT / "pixels-and-order-private.json").open("xb") as f:
        f.write(payload)
    result = dict(state="REAL_DIFFSYNTH_CSV_AND_REFERENCE_V023_PIL_IMAGE_BRANCH_CHECKED",
                  evaluated_splits=report, actual_PIL_calls=counts, combined_order_and_pixels_SHA256=digest(payload),
                  real_Torch_imports=0, reference_vision_version="v0.23.0", local_Pillow=PIL.__version__,
                  local_pandas=pandas.__version__, target_torchvision_identity_verified=False,
                  actual_model_calls=0, actual_gpu_steps=0, actual_optimizer_updates=0,
                  training_ready_increment=0, pins={name: {"bytes": size, "SHA256": sha}
                                                   for name, (size, sha) in PINS.items()})
    with (OUTPUT / "result.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
