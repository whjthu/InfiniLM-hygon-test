"""Render the one-shot four-arm controller from exact published originals."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = HERE.with_name("training-minicpm-checkpoint20-sdpa-repeatability-v1")
OLD_NAME = "training-minicpm-checkpoint20-rng-bracket-target-v1"
TARGET = HERE.name
REQUEST = "cbbd25487dca4090a52c923bbfb7f60b"
PRIOR = {
    "bracket_contract.py": "8c09c94f5d7efe5fc85ac4a52e3bf0b8c553fbc700aebe4e5e12324fdb26a0ac",
    "host_once.py": "68c50e546e3f9d6f83be0595d045006a8e69a1c4831f968d0bfd839994c02173",
    "worker.py": "cd7296dfc83e64b7d7d244b7d05424764c688ce62dbb548ee55dc46e7a34c4ec",
}
NEW_WORKERS = ("native_route_observer.py", "route_control.py", "read_completed.py")
ARCHIVE_PIN = "fca3ddc876ec7d3e0db17bed47589fb42962d9540d15200277150dfdc2651452"
ROUTE_MF = "d5a8da956c1c57cd37ce59b08c27b917dd4ee5eaff876be59b41506fe993e8a3"
ORIGINAL_BRACKET = "results/bw1000/2026-11-22/MiniCPM-checkpoint20-RNG-bracket-original1194.json"
ORIGINAL_BRACKET_PIN = "0dc5a3bbee98d62deb2a686f99623e985491a13f70490b5581120a46afc60f38"
NEW_CURRENT = {
    "current_membership.py": (REPO / "tools/read_training_slurm_membership.py",
                              "8d071a48cad3a36ae0f917a2b5c7714abecc749cca55c8a38d1942fb2cc679be"),
    "current_ledger.json": (REPO / "coordination/training/training80-replacement1194900-node-task-map-v1.json",
                             "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d"),
}


def digest(raw):
    return sha256(raw).hexdigest()


def exact(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("published dependency changed: " + path.name)
    return raw


def replace(text, before, after):
    if text.count(before) != 1:
        raise ValueError("original source anchor differs: " + before[:100])
    return text.replace(before, after, 1)


def contract(source):
    anchors = (
        ("one three-arm", "one four-arm"),
        (OLD_NAME, TARGET),
        ("725c35a7012044e1a68c9bed22edcc75", REQUEST),
        ("29949", "29953"),
        ("AB = {'a1': 460, 'b': 840, 'a2': 1220}", "AB = {'a1': 460, 'b1': 840, 'b2': 1220, 'a2': 1600}"),
        ("1260, 1360, 1440, 1800", "1640, 1740, 1860, 2220"),
        ("PLANNED = 24", "PLANNED = 32"),
        ("MINICPM_RNG_BRACKET_ADMISSION_V1", "MINICPM_SDPA_REPEATABILITY_ADMISSION_V1"),
    )
    # Runtime member names are determined by this assembly and checked by both parent and worker.
    for before, after in anchors:
        source = replace(source, before, after)
    source = replace(source, next(line for line in source.splitlines() if line.startswith("TARGET_FILES = ")),
                     "TARGET_FILES = {'CLAIM.json','bracket_contract.py','host_once.py','worker.py',"
                     "'current_admission.py','current_membership.py','current_ledger.json',"
                     "'native_route_observer.py','route_control.py','read_completed.py'}")
    source = replace(source, "'compare_bracket.py', 'rng_treatment.py', 'native_rng_observer.py', 'original1189-route.json',",
                     "'read_completed.py', 'route_control.py', 'native_route_observer.py', 'original1189-route.json',")
    source = replace(source, "target-worker/native_rng_observer.py", "target-worker/native_route_observer.py")
    source = replace(source, "MINICPM_RNG_TREATMENT", "MINICPM_ROUTE_ARM")
    return replace(source, """'first-forward-state.json', 'first-forward-ready.json', 'first-forward-stop.json',
                                  'rng-treatment.json', 'rng-state-audit.json'""",
                   """'first-forward-state.json', 'first-forward-ready.json', 'first-forward-stop.json'""")


def worker(source):
    source = replace(source, "MINICPM_RNG_TREATMENT", "MINICPM_ROUTE_ARM")
    source = replace(source, "        from compare_bracket import consume as compare\n",
                     "        from read_completed import read_arms\n        from route_control import comparison\n")
    source = replace(source, "        return compare(root, step, HERE / 'original1189-route.json', records)",
                     "        fresh, sources, errors = read_arms(root, step)\n"
                     "        result = comparison(fresh, step, records)\n"
                     "        return dict(result, native_state_receipts_SHA256=sources, missing_or_invalid_receipt_files=errors)")
    return replace(source, "rng-bracket-comparison.json", "sdpa-repeatability-comparison.json")


def host(source):
    source = replace(source, "one new RNG bracket", "one four-arm SDPA repeatability run")
    source = replace(source, "REPO = HERE.parents[2]", "REPO = HERE.parents[2]\n"
                     "import current_admission\n\n"
                     f"BASELINE = {ORIGINAL_BRACKET!r}\n"
                     f"BASELINE_SHA = {ORIGINAL_BRACKET_PIN!r}")
    source = replace(source, "    require_original(baseline, route)\n", """    require_original(baseline, route)
    bracket = json.loads(read(repo / BASELINE, BASELINE_SHA))
    rows = bracket['forwards']
    need(bracket['parent'] == PARENT and bracket['node'] == NODE
         and bracket['actual_optimizer_updates'] == 0 and rows['completed'] == rows['planned'] == 24
         and bracket['interpretation'] == 'CONTROL_REPEAT_DIFFERS'
         and any(row[:4] == ['1103606.1194', 'gpu', 'COMPLETED', '0:0']
                 for row in bracket['steps']),
         'original completed SDPA repeatability trigger differs')
""")
    source = replace(source, "    rng_source = frozen(repo / SOURCE, SOURCE_SHA)\n", "")
    source = replace(source, """    for name in ('rng_treatment.py', 'native_rng_observer.py', 'compare_bracket.py'):
        bundle[name] = rng_source[name]
""", "")
    source = replace(source, "    old.RUN = Path(str(RUN))",
                     "    old.RUN = Path(str(RUN))\n"
                     "    old.original_membership = lambda: current_admission.check(os.getuid())")
    source = replace(source, "rng-bracket-comparison.json", "sdpa-repeatability-comparison.json")
    source = replace(source, "RNG bracket namespace or Slurm label already exists",
                     "SDPA repeatability namespace or Slurm label already exists")
    source = replace(source, "summary['verified_individual_forwards']", "summary['verified_forwards']")
    source = replace(source, "for name in ('bracket_contract.py', 'worker.py')",
                     "for name in ('bracket_contract.py', 'worker.py', 'native_route_observer.py', 'route_control.py', 'read_completed.py')")
    source = replace(source, "    group.add_argument('--preflight', action='store_true')",
                     "    group.add_argument('--asset-preflight', action='store_true')\n"
                     "    group.add_argument('--preflight', action='store_true')")
    source = replace(source, """    need(args.admission is not None and args.admission_sha256 and args.source_sha256, 'missing exact source and adoption pins')
    doc = json.loads(read(args.admission, args.admission_sha256))
    admission(doc)
""", """    need(args.source_sha256, 'missing exact rendered source pin')
    if args.asset_preflight:
        need(args.admission is None and args.admission_sha256 is None, 'asset preflight takes only the source pin')
        doc = {}
    else:
        need(args.admission is not None and args.admission_sha256, 'missing exact adoption pin')
        doc = json.loads(read(args.admission, args.admission_sha256))
        admission(doc)
""")
    source = replace(source, """    for arm in AB:
        render(previous, native, arm)
    bind(previous, old)
""", """    for arm in AB:
        render(previous, native, arm)
    if args.asset_preflight:
        labels = {phase: old.accounting(LABEL + '-' + phase) for phase in PHASES}
        need(not Path(str(RUN)).exists() and all(row is None for row in labels.values()),
             'SDPA repeatability namespace or Slurm label already exists')
        print(json.dumps(dict(request=REQUEST, planned_forwards=PLANNED, checkpoint_members=14,
                             installed_worker_files=len(bundle), actual_GPU_steps=0,
                             source_manifest_SHA256=args.source_sha256, original_asset_gate='PASSED',
                             device_admission_checked=False)))
        return 0
    bind(previous, old)
""")
    return source


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def build(destination):
    helper = REPO / "tools/prepare_minicpm_archived_recovery_host.py"
    archive = ModuleType("frozen_minicpm_archive")
    exec(compile(exact(helper, ARCHIVE_PIN), str(helper), "exec"), vars(archive))
    source_manifest = json.loads(exact(SOURCE / "SOURCE_FILES.json", ROUTE_MF))
    additions = {n: exact(SOURCE / n, source_manifest["files"][n]["SHA256"]) for n in NEW_WORKERS}
    additions.update({name: exact(path, pin) for name, (path, pin) in NEW_CURRENT.items()})
    additions.update({n: (HERE / n).read_bytes() for n in ("current_admission.py", "CLAIM.json")})
    if json.loads(additions["CLAIM.json"])["original_rng_bracket_public_SHA256"] != ORIGINAL_BRACKET_PIN:
        raise ValueError("original bracket pin differs from claim")
    bracket = exact(REPO / ORIGINAL_BRACKET, ORIGINAL_BRACKET_PIN)
    archive_meta = archive.materialize(REPO, destination)
    with (destination / ORIGINAL_BRACKET).open("xb") as output:
        output.write(bracket)
    parent = destination / "code/bw1000" / OLD_NAME
    for name, transform in (("bracket_contract.py", contract), ("worker.py", worker), ("host_once.py", host)):
        additions[name] = transform(exact(parent / name, PRIOR[name]).decode()).encode()
    for name, raw in additions.items():
        if name.endswith(".py"):
            ast.parse(raw, filename=name, feature_version=(3, 10))
    directory = destination / "code/bw1000" / TARGET
    directory.mkdir(mode=0o700)
    manifest = {"schema": "bw1000.training.minicpm.sdpa-repeatability.rendered-target.v1",
                "request": REQUEST, "archived_source_count": archive_meta["exact_files"],
                "archived_member_set_SHA256": archive_meta["member_set_SHA256"], "original_route_manifest_SHA256": ROUTE_MF,
                "original_rng_bracket_public_SHA256": ORIGINAL_BRACKET_PIN,
                "files": {name: {"bytes": len(raw), "SHA256": digest(raw)} for name, raw in additions.items()}}
    for name, raw in {**additions, "SOURCE_FILES.json": encoded(manifest)}.items():
        with (directory / name).open("xb") as output:
            output.write(raw)
    return directory, digest(encoded(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    path, pin = build(parser.parse_args().destination.absolute())
    print(json.dumps(dict(target_source=str(path), source_SHA256=pin, request=REQUEST)))


if __name__ == "__main__":
    main()
