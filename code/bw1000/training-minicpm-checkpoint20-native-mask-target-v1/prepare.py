"""Render the four-arm native-mask controller from pinned deployed source."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGET = HERE.name
REQUEST = "3b71a062261547af8b997684603c9e6b"
PRIOR = HERE.with_name("training-minicpm-checkpoint20-sdpa-repeatability-target-v1")
PRIOR_PREPARE = "6535f7e5316df26757fa7bac630cba5244c24a3b04828aaa4b14ee59daf3c87b"
PRIOR_RENDER = "5bf1503b7af5863de7b3c8b5dc330d0e6f5bc1a8d0a20abab7988bf0cf8792b2"
TAP = HERE.with_name("training-minicpm-checkpoint20-native-mask-tap-v1")
TAP_MANIFEST = "bb958443d5c7f084c38c860e28fcf7a51b51de9fa4df4e813cd2f423ccacd285"
ORIGINAL = "results/bw1000/2026-11-24/training-minicpm-sdpa-original2303-terminal-independent.json"
ORIGINAL_SHA = "2c2a33cae30da15984a6cb5255a232f7447774e4fa4a00a7e26bba7523527c0e"
COMPANIONS = "'native_plugin.py', 'tap.py', 'mask_rows.py', 'mask_compare.py', "
SUMMARY = "native-mask-repeatability-comparison.json"


def digest(raw):
    return sha256(raw).hexdigest()


def exact(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("published dependency changed: " + path.name)
    return raw


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("original rendering anchor differs: " + old[:120])
    return text.replace(old, new, 1)


def contract(source):
    for old, new in (
        (PRIOR.name, TARGET), ("cbbd25487dca4090a52c923bbfb7f60b", REQUEST), ("29953", "29957"),
        ("{'a1': 460, 'b1': 840, 'b2': 1220, 'a2': 1600}", "{'u1': 460, 'c1': 840, 'c2': 1220, 'u2': 1600}"),
        ("MINICPM_SDPA_REPEATABILITY_ADMISSION_V1", "MINICPM_NATIVE_MASK_ADMISSION_V1"),
        ("     and doc['root_GO'] is True",
         "     and doc['original_native_route_comparison_step'] == '1103606.2303'\n"
         f"         and doc['original_native_route_comparison_SHA256'] == {ORIGINAL_SHA!r}\n"
         "         and doc['mask_storage_cap_per_capture_rank_bytes'] == 536870912\n"
         "         and doc['root_GO'] is True"),
        ("'native_route_observer.py','route_control.py'", (COMPANIONS + "'route_control.py'").replace(", ", ",")),
        ("'route_control.py', 'native_route_observer.py',", "'route_control.py', " + COMPANIONS),
        ("target-worker/native_route_observer.py", "target-worker/native_plugin.py"),
        ("    result['environment_overrides']['MINICPM_ROUTE_ARM'] = arm",
         "    result['environment_overrides']['MINICPM_NATIVE_MASK_ARM'] = arm\n"
         "    result['environment_overrides']['MINICPM_NATIVE_MASK_STORAGE_LIMIT'] = '536870912'"),
    ):
        source = replace(source, old, new)
    return source


def host(source):
    source = replace(source, "one four-arm SDPA repeatability run", "one four-arm native dropout mask diagnostic")
    source = replace(source, "    require_original(baseline, route)\n", f"""    require_original(baseline, route)
    latest = json.loads(read(repo / {ORIGINAL!r}, {ORIGINAL_SHA!r}))
    steps = latest['terminal_and_zero_GRES_post']
    pairs = latest['paired_loss_and_rng']
    need(steps['gpu'][0] == '1103606.2303' and steps['gpu'][2:4] == ['COMPLETED', '0:0']
         and steps['post'][0] == '1103606.2304' and latest['exact_post_lease_and_ownership_verified'] is True
         and latest['comparison']['verified_forwards'] == 32 and latest['actual_optimizer_updates'] == 0
         and all(pairs[name]['exact_final_rng_ranks'] == list(range(8)) and len(pairs[name]['exact_loss_ranks']) == 3
                 for name in ('flash', 'math')), 'original SDPA negative control trigger differs')
""")
    source = replace(source, "'native_route_observer.py', 'route_control.py'", COMPANIONS + "'route_control.py'")
    source = replace(source, "sdpa-repeatability-comparison.json", SUMMARY)
    return source.replace("SDPA repeatability namespace", "native mask namespace")


def worker(source):
    source = replace(source, "MINICPM_ROUTE_ARM", "MINICPM_NATIVE_MASK_ARM")
    source = replace(source, "        from route_control import comparison", "        from mask_compare import comparison")
    return replace(source, "sdpa-repeatability-comparison.json", SUMMARY)


def plugin(source):
    return replace(source, "                    value = dict(value, native_mask_observation=", """                    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
                    owner = dict(arm=arm, declared_policy="NATIVE_AUTO", pid=os.getpid(),
                                 startticks=int(fields[19]), step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"],
                                 boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
                    value = dict(value, route_owner=owner, native_mask_observation=""")


def light_mask_rows(tap):
    tree = ast.parse(tap)
    comparison = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compare")
    return "from route_control import require\n\nEXPECTED_DROPOUTS = 186\n\n" + ast.unparse(comparison) + "\n"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def build(destination):
    prepare_path = PRIOR / "prepare.py"
    builder = ModuleType("frozen_original_sdpa_builder")
    builder.__file__ = str(prepare_path)
    exec(compile(exact(prepare_path, PRIOR_PREPARE), str(prepare_path), "exec"), vars(builder))
    old, pinned = builder.build(destination)
    if pinned != PRIOR_RENDER:
        raise ValueError("original rendered controller changed")
    manifest = json.loads(exact(old / "SOURCE_FILES.json", PRIOR_RENDER))
    prior = {n: exact(old / n, p["SHA256"]) for n, p in manifest["files"].items()}
    tap_pins = json.loads(exact(TAP / "SOURCE_FILES.json", TAP_MANIFEST))["files"]
    tap, native = (exact(TAP / n, tap_pins[n]["SHA256"]).decode() for n in ("tap.py", "native_plugin.py"))
    candidate = {name: prior[name] for name in ("current_admission.py", "current_ledger.json",
                                                  "current_membership.py", "route_control.py")}
    for name, transform in (("bracket_contract.py", contract), ("host_once.py", host), ("worker.py", worker)):
        candidate[name] = transform(prior[name].decode()).encode()
    candidate.update({
        "tap.py": tap.encode(), "native_plugin.py": plugin(native).encode(),
        "mask_rows.py": light_mask_rows(tap).encode(), "mask_compare.py": (HERE / "mask_compare.py").read_bytes(),
        "CLAIM.json": (HERE / "CLAIM.json").read_bytes(),
        "read_completed.py": replace(prior["read_completed.py"].decode(),
            "from route_control import ARMS, comparison, require",
            "from mask_compare import ARMS, comparison\nfrom route_control import require").encode(),
    })
    proof = exact(REPO / ORIGINAL, ORIGINAL_SHA)
    original_output = destination / ORIGINAL
    original_output.parent.mkdir(parents=True, exist_ok=True)
    with original_output.open("xb") as output:
        output.write(proof)
    for name, raw in candidate.items():
        if name.endswith(".py"):
            ast.parse(raw, filename=name, feature_version=(3, 10))
    folder = old.with_name(TARGET)
    old.rename(folder)
    # The prior rendered directory was generated solely inside this new destination.
    (folder / "native_route_observer.py").unlink()
    output = {"schema": "bw1000.training.minicpm.native-mask.rendered-target.v1", "request": REQUEST,
              "prior_rendered_manifest_SHA256": PRIOR_RENDER, "tap_source_manifest_SHA256": TAP_MANIFEST,
              "original_independent_terminal_SHA256": ORIGINAL_SHA,
              "files": {name: {"bytes": len(raw), "SHA256": digest(raw)} for name, raw in candidate.items()}}
    for name, raw in {**candidate, "SOURCE_FILES.json": encoded(output)}.items():
        (folder / name).write_bytes(raw)
    return folder, digest(encoded(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    path, digest = build(parser.parse_args().destination.absolute())
    print(json.dumps(dict(target_source=str(path), source_SHA256=digest, request=REQUEST)))


if __name__ == "__main__":
    main()
