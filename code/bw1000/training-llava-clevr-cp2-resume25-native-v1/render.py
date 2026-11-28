"""Generate the continuation entry without modifying any first-two or CPU sources."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIRST2 = HERE.with_name("training-llava-clevr-first2-native-v1")
CPU = HERE.with_name("training-llava-clevr-cp2-resume25-v1")
PROVENANCE = {
    FIRST2 / "train.py": "11ef4797bcf6a4d9e19f7e45aa078dd4cd2ed7e52132358780a28151526c6fda",
    FIRST2 / "callback.py": "9b3ca2261ebaa8b2c7cf1b1e54ab61e8fbff41dd39065fb7447b1203ae0d1430",
    CPU / "contract.py": "9298a923dbfe340c789b7882b8cfb592e47457868c35dba7d73dae7fa544d83b",
}


def digest(raw):
    return sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def exactly(text, before, after):
    if text.count(before) != 1:
        raise ValueError("EXACT_FIRST2_SOURCE_ANCHOR")
    return text.replace(before, after)


def render():
    originals = {path.name: path.read_bytes() for path in PROVENANCE}
    if any(digest(originals[path.name]) != expected for path, expected in PROVENANCE.items()):
        raise ValueError("EXACT_ORIGINAL_4983_SOURCE_LINEAGE")
    text = originals["train.py"].decode()
    text = exactly(text, 'CALLBACK = "llava_clevr_first2_native_v1"', 'import resume_config\n\nCALLBACK = resume_config.CALLBACK')
    text = exactly(text, "    _, raw = inputs(probe, fingerprint_pin)",
                   "    resume_config.bind(root, request)\n    _, raw = inputs(probe, fingerprint_pin)")
    text = exactly(text, '    sys.argv = [str(entry), *cli(original, model, dataset, root / "train", HERE / "callback.py")]',
                   '    arguments = cli(original, model, dataset, root / "train", HERE / "callback.py")\n'
                   '    sys.argv = [str(entry), *resume_config.resume_argv(arguments, CALLBACK)]')
    callback = originals["callback.py"].decode()
    wanted = {"cpu_tensor", "input_sha", "geometry", "snapshot"}
    helpers = [ast.get_source_segment(callback, node) for node in ast.parse(callback).body
               if type(node) is ast.FunctionDef and node.name in wanted]
    if len(helpers) != len(wanted):
        raise ValueError("EXACT_4983_INPUT_SNAPSHOT_HELPERS")
    observer = "import numpy as np\nimport torch\nfrom vlm.runtime.first_update_audit import fingerprint\nfrom train import need\n\n"
    observer += "\n\n".join(helpers) + "\n"
    files = dict((name, (HERE / name).read_bytes()) for name in ("callback.py", "restore.py", "resume_config.py", "verify.py"))
    files.update({"train.py": text.encode(), "first2_observe.py": observer.encode(), "contract.py": originals["contract.py"]})
    for name, raw in files.items():
        compile(raw, name, "exec")
    files["SOURCE_FILES.json"] = encoded(dict(
        state="CPU_ORDER_AND_FRESH_OWNER_REQUIRED",
        files={name: dict(bytes=len(raw), SHA256=digest(raw)) for name, raw in files.items()},
    ))
    return files


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    files = render()
    args.output.mkdir(mode=0o700)
    for name, raw in files.items():
        with (args.output / name).open("xb") as out:
            out.write(raw)
    print(digest(files["SOURCE_FILES.json"]))


if __name__ == "__main__":
    main()
