"""Freeze the registered replay from the retained Hygon model, image and first-update files."""

import hashlib
import json
from pathlib import Path
import sys


PRIOR = Path("/prior-approved")
NATIVE = Path("/native-source")
DATASET = Path("/coco20-inputs")
MODEL = Path("/sealed-runtime")
REFERENCE = Path("/original-first")
FRAMEWORK = Path("/common-framework")
PROVEN = {
    "RUNTIME_EVIDENCE.json": "23abd881be38c6d34695eeac68fc2927f89d7500ef0e333c63b41fe000a70ff1",
    "COCO20_ORIGINAL.json": "070013087a083bdec7defafaef92e781579e4ea491665536c48abe2336a21f55",
    "worker-result.json": "ee8005898010ceb6865d3679989a1109ced57aaca4cb48c38210d2599e8abc23",
    "FUSION_IDENTITY.private.json": "24b683a6a3fabc5b8ca0400c25c6b701f05aea43b54bf62d1158c64e4c123dc8",
}
ORIGINAL_INPUT_SHA256 = "09b187d1d9c57d2413226cb252e8e6313e35b3e166c83fd0ba1ba0f2165aece1"
ORIGINAL_RNG_SHA256 = "bb171816fbab74cdff76a876a0a1310f73569d3384366bcf2e55eb1e9f5a8aaf"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_pin(path, expected):
    if not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError("Missing or aliased original file: " + path.name)
    actual = path.stat()
    if actual.st_size != expected["bytes"]:
        raise ValueError("Original file extent differs: " + path.name)
    return dict(expected, path=str(path), mtime_ns=actual.st_mtime_ns,
                inode=actual.st_ino, device=actual.st_dev)


def main(output):
    sys.path.insert(0, str(FRAMEWORK / "generation/src"))
    import numpy
    import torch
    from gencuda import hy_resume_checkpoint as common
    from gencuda import zimage_hip_replay_binding as hip
    from gencuda.zimage_native_record import canonical, validate_binding

    if torch.cuda.is_available():
        raise ValueError("Binding preflight requires zero GRES")
    old = {}
    for name, expected in PROVEN.items():
        path = PRIOR / name
        old[name] = json.loads(hip.read_pin(path, {"bytes": path.stat().st_size, "SHA256": expected}, 1 << 20))
    proof = old["worker-result.json"]
    if proof["step"] != "1103606.1737" or proof["passed"] is not True or not proof["domain_update"]["original_initial_adapter_exact"]:
        raise ValueError("Original full-model validation differs")
    model_files = {}
    asset_paths = {}
    for row in old["RUNTIME_EVIDENCE.json"]["independent_runtime"]["files"]:
        name = row["path"]
        pin = {k: row[k] for k in ("bytes", "SHA256")}
        model_files[name] = file_pin(MODEL / name, pin)
        asset_paths[int(row["asset_id"])] = str(MODEL / name)
    model_paths = [[asset_paths[i] for i in (14, 15, 16)], [asset_paths[i] for i in (5, 6, 7)], asset_paths[19]]
    preset_path = asset_paths[20]
    preset_name = Path(preset_path).relative_to(MODEL).as_posix()
    data = old["COCO20_ORIGINAL.json"]
    images = {}
    for name, pin in data["images"].items():
        candidate = DATASET / name
        images[name] = file_pin(candidate, pin)
        hip.read_pin(candidate, pin, 16 << 20)
    metadata = DATASET / "metadata-zimage-20.csv"
    hip.read_pin(metadata, data["metadata"], 1 << 20)
    source_files = {}
    tree = hashlib.sha256()
    for path in sorted((p for p in NATIVE.rglob("*") if not p.is_dir()),
                       key=lambda p: p.relative_to(NATIVE).as_posix()):
        if path.resolve() != path or path.stat().st_size > 16 << 20:
            raise ValueError("Unexpected native source namespace")
        name = path.relative_to(NATIVE).as_posix()
        content = path.read_bytes()
        for part in (name.encode(), content):
            tree.update(len(part).to_bytes(8, "big"))
            tree.update(part)
        source_files[name] = {"bytes": len(content), "SHA256": digest(content)}
    if tree.hexdigest() != hip.SOURCE_TREE_SHA256:
        raise ValueError("Retained DiffSynth source changed")
    progress = REFERENCE / "progress.json"
    progress_raw = hip.read_pin(progress, dict(bytes=1873, SHA256=hip.PROGRESS_SHA256), 64 << 10)
    initial_pins = {pin["name"]: pin for pin in json.loads(progress_raw)["files"]}
    captures = {}
    for key, name in (("trainable", "initial.pt"), ("rng", "input.pt")):
        captures[key] = file_pin(REFERENCE / name, initial_pins[name])
    config = dict(model="z-image-turbo",seed=20260911,cards="0",precision="bfloat16",lora_rank=4,
                  lora_target_modules="to_q,to_k,to_v,to_out.0,w1,w2,w3",learning_rate=1e-4,weight_decay=.01,
                  max_steps=25,warmup_steps=5,max_grad_norm=1.,gradient_checkpointing=True,height=256,width=256,
                  checkpoint_steps=[10,20,25],validate_recovery=False,validate_turbo_regression=True,
                  dataset_root=str(DATASET),dataset_metadata=str(metadata))
    fused_path = PRIOR / "FUSION_IDENTITY.private.json"
    fused = dict(path=str(fused_path),bytes=fused_path.stat().st_size,SHA256=hip.FUSION_SHA256)
    binding = dict(case="T07.sft",profile=hip.PROFILE,platform="hip",logical_device_ids=[0],
                   model_root=str(MODEL),model_paths=model_paths,tokenizer_path=str(MODEL / "base/tokenizer"),
                   preset_path=preset_path,model=dict(files=model_files,model_files=len(model_files),
                                                     preset_files=1,preset_stat=model_files[preset_name]),
                   source_root=str(NATIVE),source_files=source_files,source_tree_SHA256=tree.hexdigest(),
                   data=dict(root=str(DATASET),row_order=data["row_order"],images=images,
                             small_files={metadata.name:data["metadata"]}),
                   initial=dict(artifacts=captures,trainable_tensors=476,trainable_elements=9922560),
                   config=config,config_canonical_SHA256=canonical(config),fused_base=fused,
                   execution_identity=dict(source_tree_SHA256=tree.hexdigest(),original_step="1087656.1153",
                                           retained_model_verified_step=proof["step"]),
                   versions=proof["versions"],GPU_arch=proof["architecture"],HIP_version=proof["hip"],
                   GPU_total_bytes=proof["total_device_memory"],initial_minimum_free_bytes=36 << 30,
                   original_rng_SHA256=ORIGINAL_RNG_SHA256,original_input_SHA256=ORIGINAL_INPUT_SHA256,
                   planned_continuous_steps=25,planned_resume_global_steps=[21,22,23,24,25])
    validate_binding(binding)
    torch.set_num_threads(4)
    initial, _, observed_rng = hip.initial_state(torch,numpy,common,binding)
    if observed_rng != ORIGINAL_RNG_SHA256 or len(initial) != 476 or torch.cuda.is_initialized():
        raise ValueError("Independent CPU initial state or isolation differs")
    payload = (json.dumps(binding,sort_keys=True) + "\n").encode()
    if len(payload) > 1 << 20:
        raise ValueError("Registered binding exceeds CLI input limit")
    with output.open("xb") as stream:
        stream.write(payload)
    print(json.dumps({"binding_SHA256":digest(payload),"binding_bytes":len(payload),
                      "source_files":len(source_files),"model_files":len(model_files),
                      "original_images":len(images),"initial_tensors":len(initial),
                      "original_rng_SHA256":observed_rng,"actual_optimizer_updates":0},sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
