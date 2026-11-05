"""Bind the private GLM continuation entry to the original 910371.835 build."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from assemble_private_worker import pinned, replace_once
from derive_continuation_resident import derive_resident

PARENT_SHA = "64311f41c945fc5aa69bad9e07bce4f7ef1a44c576624193a87b5e14b4462c36"
NATIVE_SHA = "35c08ce45814178e3692c55eb13dd5f074643993be1fbe911ecf91a2c200a0cb"
RECEIPT_SHA = "6f6ff56db562391fc8003eb9fe0faab945cbded4eeff22def31c53c519be62dd"
NATIVE_DIR = "/refactor/candidates/GLM952-block-prefix-native-v1"
NATIVE_PATH = NATIVE_DIR + "/libGLM952_block_prefix_v1.so"
HERE = Path(__file__).resolve().parent

NATIVE_GUARD = f'''
"""Bind the staged-prefix DSO to the original successful CPU build receipt."""
from hashlib import sha256
import json
from pathlib import Path

NATIVE_SHA256 = {NATIVE_SHA!r}
NATIVE_PATH = {NATIVE_PATH!r}
RECEIPT_PATH = {NATIVE_DIR + "/CPU-build.json"!r}
RECEIPT_SHA = {RECEIPT_SHA!r}
ADDED_CAPI = {{
    "modern_glm_block_stage_v1", "modern_glm_block_prepare_v1", "modern_glm_block_publish_v1",
}}


def verify_config(config):
    if (config["Native_SHA256"], config["Native_library"]) != (NATIVE_SHA256, NATIVE_PATH):
        raise ValueError("GLM continuation Native identity differs")


def verify(config):
    verify_config(config)
    raw = Path(RECEIPT_PATH).read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("GLM continuation CPU build receipt differs")
    receipt = json.loads(raw)
    if (
        receipt["actual_CPU"] != "910371.835"
        or receipt["completed"] is not True
        or receipt["GPU"] is not False
        or receipt["Model"] is not False
        or receipt["dlopen"] is not False
        or receipt["compile_count"] != 8
        or receipt["retained_objects"] != 21
        or receipt["dynamic_metadata_equal"] is not True
        or set(receipt["added_CAPI"]) != ADDED_CAPI
        or (receipt["library_SHA256"], receipt["library"]) != (NATIVE_SHA256, NATIVE_PATH)
    ):
        raise ValueError("GLM continuation CPU build receipt scope differs")
    if sha256(Path(NATIVE_PATH).read_bytes()).hexdigest() != NATIVE_SHA256:
        raise ValueError("GLM continuation Native bytes differ")
    return dict(library_SHA256=NATIVE_SHA256, library_actual_path=NATIVE_PATH,
                cpu_receipt_SHA256=RECEIPT_SHA, cpu_step=receipt["actual_CPU"])
'''.lstrip()


def derive(raw, pairs):
    source = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        source = replace_once(source, old, new)
    compile(source, "<derived-prefix-continuation>", "exec")
    return source.encode("utf-8")


def assemble(parent, out, plan_path):
    parent, out = parent.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(parent):
        raise ValueError("private continuation output must be new and outside its parent")
    manifest = json.loads(pinned(parent / "SOURCE_MANIFEST.json", PARENT_SHA))
    files = {}
    for name, digest in manifest["files_SHA256"].items():
        src = (parent / name).resolve()
        if not src.is_relative_to(parent):
            raise ValueError("private GLM source member escapes provider")
        files[name] = pinned(src, digest)

    prefix = "public_glm_native/"
    admission = prefix + "model_admission_r404.py"
    files[admission] = derive(files[admission], (
        ("LIBRARY='/refactor/candidates/GLM-V-on-demand-Model-integration-r403-CPU/libGLM_V_on_demand_Model_r403.so'",
         "LIBRARY=" + repr(NATIVE_PATH)),
        ("NATIVE_SHA='9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'",
         "NATIVE_SHA=" + repr(NATIVE_SHA)),
    ))
    for name in ("accepted_prefix_commit.py", "accepted_prefix_owner.py",
                 "accepted_prefix_continuation.py", "freeze_continuation_inputs.py"):
        files[prefix + name] = (HERE / name).read_bytes()
    files[prefix + "prefix_native_guard.py"] = NATIVE_GUARD.encode("utf-8")
    files[prefix + "resident_continuation.py"] = derive_resident(parent / (prefix + "resident_compare.py"))
    worker = files[prefix + "diagnostic_worker.py"]
    files[prefix + "diagnostic_worker.py"] = derive(worker, (
        ("from .native_chain_r186 import verify as verify_native_chain",
         "from .prefix_native_guard import verify as verify_native_chain"),
        ("from .resident_compare import run_resident_compare",
         "from .resident_continuation import run_resident_prefix_continuation"),
        ('config.get("block_compare_execution_authorized")',
         'config.get("prefix_continuation_execution_authorized")'),
        ("proof = verify_native_chain()", "proof = verify_native_chain(config)"),
        ("result = run_resident_compare(", "result = run_resident_prefix_continuation("),
        ("run=run, configure_native_environment=configure_native_environment,",
         'run=run, configure_native_environment=configure_native_environment,\n'
         '            verified_native_sha=proof["library_SHA256"],'),
    ))
    stage = files[prefix + "stage_compare.py"]
    files[prefix + "stage_compare.py"] = derive(stage, (
        ("from .resident_compare import NATIVE_SHA256, PLAN_SHA256",
         "from .freeze_continuation_inputs import freeze, PARENT\n"
         "from .prefix_native_guard import NATIVE_SHA256, RECEIPT_SHA, verify, verify_config\n"
         "from .resident_continuation import PLAN_SHA256"),
        ("    validate_deployment_paths(config)\n",
         "    validate_deployment_paths(config)\n    verify_config(config)\n"),
        ('"scope": "GLM952 C1 block logits comparison"',
         '"scope": "GLM952 staged-prefix continuation", "Native_CPU_receipt_SHA256": RECEIPT_SHA'),
        ('"forwards_per_owner": 72', '"forwards_per_owner": 55'),
        ('config.get("block_compare_execution_authorized")',
         'config.get("prefix_continuation_execution_authorized")'),
        ('    read_pinned(config["block_compare_private_plan"], PLAN_SHA256)\n    run_stage(config)',
         '    verify(config)\n'
         '    original = Path(config["block_compare_private_plan"]).read_bytes()\n'
         '    if hashlib.sha256(original).hexdigest() != PARENT or (\n'
         '            hashlib.sha256(freeze(original)).hexdigest() != PLAN_SHA256):\n'
         '        raise ValueError("private GLM continuation plan differs")\n'
         '    run_stage(config)'),
    ))
    config = json.loads(files["config.template.json"])
    config.update(Native_library=NATIVE_PATH, Native_SHA256=NATIVE_SHA,
                  prefix_continuation_execution_authorized=False,
                  block_compare_private_plan=plan_path,
                  nodes=["a02r3n15", "a02r3n16"],
                  runtime_directory="/refactor/results/GLM952-prefix-continuation")
    files["config.template.json"] = (json.dumps(config, indent=2) + "\n").encode()
    normalized_modules = (
        "accepted_prefix_commit.py", "accepted_prefix_continuation.py",
        "accepted_prefix_owner.py", "freeze_continuation_inputs.py",
        "owner_compare.py", "resident_compare.py",
    )
    for name in normalized_modules:
        member = prefix + name
        files[member] = files[member].replace(b"\r\n", b"\n").rstrip() + b"\n"
    for name, raw in files.items():
        if name.startswith(prefix) and name.endswith(".py"):
            compile(raw, name, "exec")
    result = dict(parent_source_manifest_SHA256=PARENT_SHA,
                  Native_CPU_receipt_SHA256=RECEIPT_SHA, Native_SHA256=NATIVE_SHA,
                  GPU_GO=False, Model_GO=False, diagnostic_only=True,
                  files_SHA256={name: sha256(raw).hexdigest() for name, raw in sorted(files.items())})
    files["SOURCE_MANIFEST.json"] = (json.dumps(result, indent=2) + "\n").encode()
    out.mkdir()
    for name, raw in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return {name: sha256(files[name]).hexdigest()
            for name in ("SOURCE_MANIFEST.json", "config.template.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("plan_path")
    args = parser.parse_args()
    print(json.dumps(assemble(args.parent, args.out, args.plan_path)))
