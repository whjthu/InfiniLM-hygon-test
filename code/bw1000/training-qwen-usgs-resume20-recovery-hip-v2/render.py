"""Render the five-step controller from the published paired controller."""

import argparse
import ast
import json
from pathlib import Path

import recovery_binding as binding


BASE = Path(__file__).resolve().parents[1] / binding.ORIGINAL_LABEL
CONTROLLER_SHA256 = "9817a55d15261570388dae669a36d9452e5f7164bb57a28d8a1cca84883e203f"
RECEIVER_SHA256 = "48e5e2a1d25d23d525f07627eccb597a2c73ab8e716ddafe5caf072dfc9744b3"
FRAMEWORK_SHA256 = "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41"


def change(raw, before, after, count=1):
    if raw.count(before) != count:
        raise ValueError(f"paired source anchor differs: {before.splitlines()[0]}")
    return raw.replace(before, after)


def replace_function(source, name, body):
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function: {name}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + body + "".join(lines[node.end_lineno:])


def controller(source):
    src = source.decode()
    src = change(src, 'import mid_epoch_recovery', 'import mid_epoch_recovery\nimport recovery_binding')
    src = change(src, f'LABEL = "{binding.ORIGINAL_LABEL}"', 'LABEL = recovery_binding.LABEL\nORIGINAL = recovery_binding.ORIGINAL')
    src = replace_function(src, "prepare", '''def prepare():
    cfg = read(RUN / "intent.json")
    require(cfg["node"] == NODE and cfg["parent"] == PARENT and cfg["arm"] == "resumed",
            "wrong recovery resource intent")
    require(accounting(recovery_binding.ORIGINAL_LABEL + "-gpu")[:4] ==
            ["1103606.501", recovery_binding.ORIGINAL_LABEL + "-gpu", "FAILED", "1:0"]
            and accounting(recovery_binding.ORIGINAL_LABEL + "-post")[:4] ==
            ["1103606.508", recovery_binding.ORIGINAL_LABEL + "-post", "COMPLETED", "0:0"],
            "original continuous and cleanup step identity differs")
    from recovery_binding import original_inputs, WITNESS
    plan, config, pins, witness = original_inputs(ORIGINAL, RUN)
    manifest = BASE / "source/FRAMEWORK_FILES.json"
    require(sha(manifest) == "e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41",
            "common framework manifest differs")
    files = read(manifest)["files"]
    require(len(files) == 393 and WITNESS in files, "framework file denominator differs")
    old_updates = mid_epoch_recovery.candidate_sources(BASE / "framework")
    FRAMEWORK.mkdir(mode=0o700, exist_ok=False)
    for name, pin in files.items():
        original = ORIGINAL / "framework" / name
        expected = old_updates.get(name)
        asset(original, len(expected) if expected is not None else pin["bytes"],
              hashlib.sha256(expected).hexdigest() if expected is not None else pin["SHA256"])
        require(original.resolve(strict=True) == original and stat.S_IMODE(original.stat().st_mode) == pin["mode"],
                "original framework mode or path differs")
        path = FRAMEWORK / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == WITNESS:
            path.write_bytes(witness)
            path.chmod(pin["mode"])
            require(sha(path) == hashlib.sha256(witness).hexdigest(), "patched shared witness changed")
        else:
            shutil.copy2(original, path)
            require(sha(path) == sha(original), "reused framework member changed")
    for name, pin in read(RUN / "src/MODEL_ASSETS.json")["model"].items():
        asset(MODEL / name, pin["bytes"], pin["SHA256"])
    require(shutil.disk_usage(RUN).free >= 4 * 1024**3, "insufficient recovery checkpoint space")
    save(RUN / "checkpoint20-files.json", pins)
    save(RUN / "resumed-plan.json", plan)
    save(RUN / "witness-config.json", config)
''')
    src = change(src, 'checkpoint=RUN/"continuous/train/checkpoint-20"', 'checkpoint=ORIGINAL/recovery_binding.CHECKPOINT')
    src = change(src, 'inventory=read(RUN/"continuous-checkpoint-20-files.json")',
                 'inventory=read(ORIGINAL/"continuous-checkpoint-20-files.json")')
    src = change(src, '        save(RUN/"checkpoint20-files.json",pins)\n',
                 '        require(pins == read(RUN/"checkpoint20-files.json"), "preflight checkpoint pins changed")\n')
    src = change(src, '''    for arm in ("continuous","resumed"):
        require(time.time()<deadline-30,"insufficient shared native budget for next arm")
        if run_arm(arm,deadline)!=0:
            return 1
        for number in ((10,20,25) if arm=="continuous" else (25,)):
            require(time.time()<deadline,"shared native epoch exhausted before checkpoint inventory")
            checkpoint_inventory(arm,number)
''', '''    require(time.time()<deadline-30, "insufficient recovery budget")
    if run_arm("resumed",deadline)!=0:
        return 1
    require(time.time()<deadline, "recovery epoch exhausted before checkpoint inventory")
    checkpoint_inventory("resumed",25)
''')
    src = change(src, '    comparison=compare(RUN)\n',
                 '    comparison=compare(ORIGINAL,RUN)\n    recovery_binding.verify_continuous_hashes(comparison)\n')
    src = change(src, '(RUN, RUN, "rw"), (BASE / "source", BASE / "source", "ro"), (MODEL, MODEL, "ro"),',
                 '''(RUN, RUN, "rw"), (BASE / "source", BASE / "source", "ro"), (MODEL, MODEL, "ro"),
                *((ORIGINAL / name, ORIGINAL / name, "ro") for name in
                  ("view", "continuous/train/checkpoint-20", "continuous/train/treasury-witness",
                   "continuous-checkpoint-20-files.json")),''')
    src = change(src, '''and all((RUN/a/"native-outcome.json").exists() and read(RUN/a/"native-outcome.json")["passed"] for a in ("continuous","resumed"))''',
                 '''and (RUN/"resumed/native-outcome.json").exists() and read(RUN/"resumed/native-outcome.json")["passed"]''')
    src = change(src, 'planned_optimizer_invocations=30,planned_updates_by_arm={"continuous":25,"resumed":5}',
                 'planned_optimizer_invocations=5,planned_updates_by_arm={"resumed":5},reused_continuous_updates=25')
    compile(src, "controller.py", "exec")
    return src.encode()


def receiver(source):
    src = source.decode()
    parameters = {
        binding.ORIGINAL_LABEL: binding.LABEL,
        "f572fda6784140cbbdcae7bb91a916b9": binding.REQUEST,
        "(REQUEST,LABEL,8,30)": "(REQUEST,LABEL,8,5)",
        'arm="continuous",planned_optimizer_updates=30': 'arm="resumed",planned_optimizer_updates=5',
        'planned_updates_by_arm={"continuous":25,"resumed":5}': 'planned_updates_by_arm={"resumed":5}',
        'planned_updates=30': 'planned_updates=5',
    }
    for old, new in parameters.items():
        src = change(src, old, new)
    compile(src, "receive.py", "exec")
    return src.encode()


def runtime(root, original=BASE):
    original_manifest = json.loads((original / "SOURCE_FILES.json").read_bytes())["files"]
    names = ("compare_recovery.py", "mid_epoch_recovery.py", "qwen_domain_recipe.py",
             "qwen_treasury_witness.py", "owned_processes.py", "snapshot.py",
             "data_view.py", "slurm_accounting.py", "MODEL_ASSETS.json")
    files = {name: binding.pinned_bytes(original.absolute() / name, original_manifest[name]["SHA256"], 128 * 1024)
             for name in names}
    c = binding.pinned_bytes(original.absolute() / "controller.py", CONTROLLER_SHA256, 128 * 1024)
    r = binding.pinned_bytes(original.absolute() / "receive.py", RECEIVER_SHA256, 128 * 1024)
    files.update({"controller.py": controller(c), "receive.py": receiver(r),
                  "recovery_binding.py": (root / "recovery_binding.py").read_bytes(),
                  "CLAIM.json": (root / "CLAIM.json").read_bytes()})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    files = runtime(Path(__file__).resolve().parent)
    members = {name: {"bytes": len(raw), "SHA256": binding.digest(raw)}
               for name, raw in sorted(files.items())}
    args.output.mkdir(exist_ok=False)
    for name, raw in files.items():
        (args.output / name).write_bytes(raw)
    manifest = (json.dumps({"files": members}, indent=2, sort_keys=True) + "\n").encode()
    (args.output / "SOURCE_FILES.json").write_bytes(manifest)
    print(json.dumps(dict(manifest_SHA256=binding.digest(manifest), members=len(members))))


if __name__ == "__main__":
    main()
