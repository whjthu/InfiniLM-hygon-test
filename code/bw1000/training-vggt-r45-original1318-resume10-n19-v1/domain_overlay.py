"""Replace only the canonical VGGT data collector for a fixed two-pair training run."""
import argparse
import ast
import hashlib
from pathlib import Path

try:
    from .data_contract import ADAPTER_SHA256, pinned, verify_bundle
except ImportError:
    from data_contract import ADAPTER_SHA256, pinned, verify_bundle

BACKEND_SHA256 = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"
SETTINGS = dict(seed=20260911, minimum_sequence_frames=24, image_size=224, frames_per_step=2,
                cached_batches=8, learning_rate=2e-5, weight_decay=0.01, max_grad_norm=1.0,
                max_steps=20, warmup_steps=5, checkpoint_steps=[10, 20, 25],
                checkpoint_cpu_threads_16=True, validate_recovery=False, training_scope="camera_head",
                cache_frozen_aggregator_features=False, cards=[0], precision="bfloat16")
VARIABLE_KEYS = {"dataset_root", "annotation_root", "dataset_manifest", "dataset_sha", "resume_from_checkpoint"}


def check_config(config):
    if set(config) - set(SETTINGS) - VARIABLE_KEYS:
        raise ValueError("Unexpected domain training configuration")
    for key, expected in SETTINGS.items():
        if type(config.get(key)) is not type(expected) or config[key] != expected:
            raise ValueError("Domain training schedule or optimizer changed: " + key)
        if isinstance(expected,list) and any(type(x) is not type(y) for x,y in zip(config[key],expected)):
            raise ValueError("Domain training list type changed: " + key)
    root = Path(config["dataset_root"])
    if not root.is_absolute() or config["annotation_root"] != str(root) or config["dataset_manifest"] != str(root / "bundle.json"):
        raise ValueError("Expected the same shared bundle path in every process")
    return root


def batches(torch, config, tensor_batch, valid_depth):
    from .tum_vggt_pair import prepare_pair, read_pinned
    import json

    root = check_config(config)
    value = verify_bundle(root, config["dataset_sha"])
    result, evidence = [], []
    shapes = dict(images=(1,2,3,224,224), depths=(1,2,224,224), extrinsics=(1,2,3,4),
                  intrinsics=(1,2,3,3), point_masks=(1,2,224,224))
    for item in value["pairs"]:
        pair = json.loads(read_pinned(root / item["manifest"], item["manifest_SHA256"]))
        raw, receipt = prepare_pair(root / item["root"], pair)
        batch = tensor_batch(torch, raw)
        if any(tuple(batch[name].shape) != shape for name,shape in shapes.items()) or not valid_depth(batch["point_masks"]):
            raise ValueError("Domain input shape or geometry changed")
        result.append(batch)
        evidence.append(receipt)
    return result, evidence


def render(raw):
    if hashlib.sha256(raw).hexdigest() != BACKEND_SHA256:
        raise ValueError("Canonical a68952cb VGGT backend changed")
    source = raw.decode("utf-8")
    edits = {
        "    from data.datasets.co3d import Co3dDataset\n": "    from ..domain_overlay import batches as domain_batches\n",
        '        "sequence_count": len(dataset.sequence_list),\n':
            '        "sequence_count": len({batch["seq_name"] for batch in batches}),\n'
            '        "domain_geometry": domain_geometry,\n'
            '        "formal_CO3D_eligible": False,\n'
            '        "formal_performance_eligible": False,\n',
    }
    for old,new in edits.items():
        if source.count(old) != 1:
            raise ValueError("Canonical VGGT data/result boundary differs")
        source = source.replace(old,new)
    start = source.index("    dataset = Co3dDataset(\n")
    end = source.index("    data_seconds = perf_counter() - data_started\n")
    source = (source[:start] +
              "    batches, domain_geometry = domain_batches(torch, config, _tensor_batch, valid_depth_batch)\n"
              "    rejected_batches = 0\n" + source[end:])
    before, after = ast.parse(raw), ast.parse(source)
    protected = {node.name:ast.dump(node,include_attributes=False) for node in before.body
                 if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name != "run_vggt_training"}
    actual = {node.name:ast.dump(node,include_attributes=False) for node in after.body
              if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name != "run_vggt_training"}
    if protected != actual:
        raise ValueError("Canonical model/checkpoint/helper definitions changed")
    return source.encode("utf-8")


def stage(original, adapter, output):
    generated = render(original.read_bytes())
    raw_adapter = pinned(adapter, ADAPTER_SHA256)
    if output.exists():
        raise FileExistsError(output)
    members = {"backends/vggt_training.py":generated, "tum_vggt_pair.py":raw_adapter,
               "domain_overlay.py":Path(__file__).read_bytes(),
               "data_contract.py":Path(__file__).with_name("data_contract.py").read_bytes()}
    output.mkdir()
    for name,raw in members.items():
        path = output / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return {name:hashlib.sha256(raw).hexdigest() for name,raw in members.items()}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend",type=Path)
    parser.add_argument("adapter",type=Path)
    parser.add_argument("new_gencuda_delta",type=Path)
    args = parser.parse_args()
    print(json.dumps(stage(args.backend,args.adapter,args.new_gencuda_delta),sort_keys=True))
