"""Render the bounded original DTK compile workflow for this standalone object."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TEMPLATE = REPO / "code/bw1000/Deep-C4-causal-consumer-r483/inverse-output-rope-CPU-v1/compile_once.py"
TEMPLATE_PIN = "763c53389ab8cfddb215c8a5f3dd53b44821179e7e636c95065854b1f438c182"
LABEL = "GLM474-fresh-kv-pack-gfx936-CPU-v1"
EXPORTS = {"glm_r474_fresh_kv_pack_t64", "glm_r474_fresh_kv_pack_t128"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def render(raw):
    if sha(raw) != TEMPLATE_PIN:
        raise ValueError("bounded compiler template SHA changed")
    edits = (
        (b"Deep-inverse-output-rope-gfx936-CPU-v1", LABEL.encode(), 1),
        (b"deep_inverse_rope_h1_v1", b"glm_r474_fresh_kv_pack_t64", 1),
        (b"deep_inverse_rope_h2_v1", b"glm_r474_fresh_kv_pack_t128", 1),
        (b'"deep_inverse_rope_"', b'"glm_r474_fresh_kv_pack_"', 1),
        (b"inverse_output_rope.", b"fresh_kv_pack.", 5),
    )
    for old, new, count in edits:
        if raw.count(old) != count:
            raise ValueError("compiler template transformation boundary")
        raw = raw.replace(old, new)
    return raw


def stage(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    tool = render(TEMPLATE.read_bytes())
    plan_bytes = (HERE / "PLAN.json").read_bytes()
    plan = json.loads(plan_bytes)
    leaf = (HERE / "fresh_kv_pack.cu").read_bytes()
    if (plan["label"] != LABEL or plan["source_SHA256"] != sha(leaf)
            or plan["compile_entry_SHA256"] != sha(tool)
            or {entry["export"] for entry in plan["variants"]} != EXPORTS):
        raise ValueError("frozen compile identity mismatch")
    files = {"fresh_kv_pack.cu": leaf, "PLAN.json": plan_bytes, "compile_once.py": tool}
    manifest = json.dumps({"files_SHA256": {name: sha(raw) for name, raw in files.items()}},
                          sort_keys=True, indent=2).encode("utf8") + b"\n"
    with TemporaryDirectory(prefix="glm-fresh-kv-object-", dir=output.parent) as temp:
        staged = Path(temp) / "compiled-source"
        staged.mkdir(mode=0o700)
        for name, raw in {**files, "SOURCE_FILES.json": manifest}.items():
            with (staged / name).open("xb") as handle:
                handle.write(raw)
        staged.rename(output)
    return {"manifest_SHA256": sha(manifest), "compiler": sha(tool), "planned_device_TUs": 1,
            "actual_target_compiles": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_output", type=Path)
    print(json.dumps(stage(parser.parse_args().new_output), indent=2, allow_nan=False))
