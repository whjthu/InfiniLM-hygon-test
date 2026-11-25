"""Render the published .1506 paired flow for the exact .1318 head."""

import ast
import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import sys
import zlib

import authority as auth


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PILOT = "code/bw1000/training-vggt-tum-two-scene-bidir-dev24-pair-v2/launch.py"
PILOT_SHA = "5a4f895c109f19a92de737a9b01ee37650dc4e080a6f24b620f43be6c35aeb47"
NEAR = "code/bw1000/training-vggt-tum-near7-head-dev24-pair-v1/launch.py"
AUDIT = "code/bw1000/training-vggt-original1318-dev24-audit-v1/audit.py"
AUDIT_SHA = "32f2eb1c523487654a3ce69e1dc0e8a01025119d4ab91cf8179105c883ac2173"
MEMBERSHIP = "tools/read_training_slurm_membership.py"
CURRENT_MEMBERSHIP_SHA = "9b3b0ca57c7ade5c28874faf46aa6925178ec0536f666842a085a76825d06de7"
HISTORICAL_MEMBERSHIP_SHA = "8d071a48cad3a36ae0f917a2b5c7714abecc749cca55c8a38d1942fb2cc679be"
CANONICAL = "code/bw1000/training-domain-pilot-v1/score_vggt_tum_pair.py"
CANONICAL_SHA = "6454c9a5f10459e5f2eb8b268918d1303d3f04c2038a12cca22c20ac87b58e22"
CONTROLLER = "tools/control_vggt_tum_cross_camera_dev.py"
NEAR_SHA = "6610a50a05d2079debf1f94f77939994b4fd1136c85193f89c55ab54741d2cfa"
SOURCE_ROUNDOFF = HERE.parent / "training-vggt-tum-two-scene-bidir-dev24-pair-v2/read_vggt_dev24_roundoff_audit.py"
ROUNDOFF_SHA = "4a0fa432a61734ac21c102c274a1028217d6a212e9ba8b19acfd83c054f43d97"
ACTIVE_LEDGER = "coordination/training/training80-replacement1194900-node-task-map-v1.json"
ACTIVE_LEDGER_SHA = "11d4cc4798f6533dc4bcf31d82dc2c206035aa498ab3842878a7cb56d688f71d"
PLAN = {"GPU": 1, "CPU": 4, "memory_MiB": 65536, "gpu_seconds": 1200, "keeper_seconds": 1500, "whole_seconds": 1800,
        "inputs": 22, "planned_windows": 24, "prediction_events": 44, "scoring_slots": 48,
        "missing_slots": 4, "optimizer_updates": 0, "formal_CO3D": False}
SHA = auth.SHA


def read_local(relative, pin=None):
    return auth.read(REPO / relative, 2 * 1024**2, pin)


def original_module(relative, pin, extra=None):
    raw = read_local(relative, pin)
    body = ast.parse(raw)
    if extra == "scorer":
        funcs = {stmt.name: stmt for stmt in body.body if isinstance(stmt, ast.FunctionDef)}
        scope = funcs["controller_sources"].body
        first, = (i for i, node in enumerate(scope) if isinstance(node, ast.Assign)
                  and ast.unparse(node.targets[0]) == "scorer_path")
        last, = (i for i, node in enumerate(scope) if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == "sources[scorer_path]")
        ns = {"sources": {auth.SCORER: read_local(auth.SCORER, "a42abb16e390417416aa3d6bfb23ecc5333995d1e1fa018964ed5391a782e620")},
              "need": auth.need}
        exec(compile(ast.Module(body=[funcs["replace_unique"], *scope[first:last+1]], type_ignores=[]),
                     relative, "exec"), ns)
        return ns["sources"][auth.SCORER]
    sys.path.insert(0, str(REPO / "tools"))
    try:
        return runpy.run_path(str(REPO / relative), run_name="pinned_vggt_original_module")
    finally:
        sys.path.pop(0)


def execution_identity(raw):
    selected = {"canonical_scorer", "load_input", "read_cases", "summarize", "measure", "load_pinned_state", "evaluate"}
    tree = ast.parse(raw)
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    auth.need(len(funcs) == len(selected), "EXACT_SCORER_EXECUTION_FUNCTIONS")
    evaluate = next(node for node in funcs if node.name == "evaluate")
    aliases, = (node for node in evaluate.body if isinstance(node, ast.Assign)
                and ast.unparse(node.targets[0]) == "roots")
    auth.need(isinstance(aliases.value, ast.Dict) and len(aliases.value.keys) == 1, "EXACT_TWO_ARMS_ONLY")
    aliases.value = ast.parse("{'candidate': args.candidate}", mode="eval").body
    # Only source-local CLI/mount aliases and arm names are normalized. Scoring code remains byte-pinned independently.
    constants = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in ("WINDOWS", "TORCH_VERSION", "HIP_VERSION", "BACKEND_SHA",
                                                       "PROVIDERS", "SCORER_SHA") for target in node.targets)]
    auth.need(len(constants) == 6, "EXACT_SCORER_PROVIDER_CONSTANTS")
    return auth.sha(ast.dump(ast.Module(body=[*constants, *funcs], type_ignores=[]), include_attributes=False).encode())


def scorers():
    old = {number: original_module(path, pin, "scorer") for number, path, pin in
           (("1038", NEAR, NEAR_SHA), ("1506", PILOT, PILOT_SHA))}
    freezer = original_module(AUDIT, AUDIT_SHA)
    candidate = freezer["candidate_source"](freezer["public"](), old["1506"])
    actual = auth.sha(candidate)
    auth.need(actual == "953ca764a83dfa607f5f0f8396d5f3c975fdc2711b9be1ba21f4ed8210cf0220", "FROZEN_1318_SCORER")
    prior = {name: {"full_scorer_SHA256": auth.sha(raw), "execution_SHA256": execution_identity(raw),
                    "canonical_SHA256": CANONICAL_SHA, "base": auth.BASE} for name, raw in old.items()}
    return candidate, prior


def replace(source, old, new):
    auth.need(source.count(old) == 1, "ORIGINAL_CONTROLLER_SOURCE_ANCHOR_CHANGED")
    return source.replace(old, new, 1)


def load_scope(scope):
    auth.need(set(scope) == {"request", "parent", "node", "only_exclusive_node", "approved_operational_scope"}
              and re.fullmatch(r"[a-f0-9]{32}", scope["request"])
              and scope["request"] not in {auth.CPU_REQUEST, auth.CANDIDATE_TRAIN_REQUEST}
              and scope["parent"] in ("1103606", "1194900") and re.fullmatch(r"f[0-9]{2}r[0-9]+n[0-9]{2}", scope["node"])
              and scope["only_exclusive_node"] is True and scope["approved_operational_scope"] is True,
              "EXPLICIT_ONE_EXCLUSIVE_APPROVED_PLACEMENT")


def render(scope, read=lambda _, name, pin=None: read_local(name, pin)):
    load_scope(scope)
    parent, node, req = [scope[name] for name in ("parent", "node", "request")]
    ledger = json.loads(read("source-read-only", ACTIVE_LEDGER, ACTIVE_LEDGER_SHA))
    roles = [name for name, group in ledger["groups"].items() if node in group["allocations"][parent]]
    auth.need(len(roles) == 1, "EXACT_PARENT_NODE_ROLE_IN_PUBLISHED_LEDGER")
    label = "vg-1318-dev24-" + req
    original = original_module(PILOT, PILOT_SHA)
    build = original["controller_sources"]
    build.__globals__.update(PARENT=parent, NODE=node, LABEL=label)
    def sourced(publication, name, wanted=None):
        if name == MEMBERSHIP:
            auth.need(wanted == HISTORICAL_MEMBERSHIP_SHA, "READBACK_MIGRATION_EXACT_ORIGINAL_ANCHOR")
            wanted = CURRENT_MEMBERSHIP_SHA
        return read(publication, name, wanted)
    sources = build("source-read-only", sourced)
    candidate, prior = scorers()
    source = sources[CONTROLLER]
    for old, new in (
        (b"training-vggt-tum-bidirectional30-v1", auth.TRAIN_DIR.encode()),
        (b"str(root / 'inputs/inputs-private.json')", b"'/dev24/inputs-private.json'"),
        (b"             (RESULTS / 'training-vggt-native-provider-HIP-CPU-v1/upstream', '/vggt-source:ro'),",
         b"             (RESULTS / 'training-vggt-tum-cross-camera-dev24-v1-staged/inputs', '/dev24:ro'),\n"
         b"             (RESULTS / 'training-vggt-native-provider-HIP-CPU-v1/upstream', '/vggt-source:ro'),"),
        (b"epoch=time.time(), max_seconds=1500, parent=PARENT, node=NODE",
         b"epoch=staged['single_epoch'], deadline=staged['whole_deadline'], max_seconds=1500, parent=PARENT, node=NODE"),
        (b"    deadline = time.monotonic() + 1500",
         b"    remaining = min(staged['single_epoch'] + 1500, staged['whole_deadline']) - time.time()\n"
         b"    require(0 < remaining <= 1500, 'expired single launch epoch')\n    deadline = time.monotonic() + remaining"),
        (b"    main()\n\n\ndef gpu_command",
         b"    from vggt_1318_hip_provenance import hip_dso\n    observed_so = hip_dso()\n"
         b"    write_json(root / 'hip-provider.json', observed_so)\n    main()\n"
         b"    require(hip_dso() == observed_so, 'in-process HIP shared object changed')\n\n\ndef gpu_command"),
        (b"    'tools/control_vggt_tum_cross_camera_dev.py', 'tools/stage_vggt_tum_cross_camera_dev.py',",
         b"    'tools/control_vggt_tum_cross_camera_dev.py', 'tools/stage_vggt_tum_cross_camera_dev.py',\n"
         b"    'tools/vggt_1318_hip_provenance.py',"),
        (b"['T12'][NODE]['state']", f"[{roles[0]!r}][NODE]['state']".encode()),
    ):
        source = replace(source, old, new)
    sources[CONTROLLER] = source
    sources[auth.SCORER] = candidate
    sources["tools/vggt_1318_hip_provenance.py"] = (HERE / "hip_provenance.py").read_bytes()
    sources.pop(auth.LEGACY_INDEX)
    index = {name: auth.sha(raw) for name, raw in sources.items()}
    sources[auth.LEGACY_INDEX] = auth.encode(index)
    for name, raw in sources.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    return sources, prior, label


def frozen(scope, publication, epoch, raw_3090, *, read=lambda _, name, pin=None: read_local(name, pin), preflight=False):
    auth.need(re.fullmatch(r"[0-9a-f]{40}", publication), "EXACT_SOURCE_PUBLICATION")
    public = read(publication, auth.PUBLIC_3090, auth.PUBLIC_3090_SHA)
    auth.cpu_evidence(public, raw_3090)
    sources, past, label = render(scope, read)
    auxiliary = {"authority.py": (HERE / "authority.py").read_bytes()}
    body = (b"INPUT=" + repr(dict(scope=scope, publication=publication, epoch=epoch, deadline=epoch+1800,
                                  independent_preflight_only=preflight,
                                  label=label, source_index_SHA256=auth.sha(sources[auth.LEGACY_INDEX]),
                                  candidate_execution_sha=execution_identity(sources[auth.SCORER]), prior=past,
                                  sources=sources, auxiliary=auxiliary, cpu_terminal=raw_3090, public_cpu=public,
                                  training=read(publication, auth.TRAIN, auth.TRAIN_SHA))).encode("ascii") + b"\n"
            + (HERE / "receive.py").read_bytes())
    if preflight:
        compile(body, "<independent target provenance only>", "exec")
        return body, None
    original = original_module(PILOT, PILOT_SHA)
    params = dict(request=scope["request"], sha=auth.sha(body), body=base64.b64encode(zlib.compress(body)).decode("ascii"),
                  directory=str(auth.PROJECT / "incoming" / (label + "-bootstrap")))
    program = ("PAYLOAD=" + repr(params) + "\n").encode("ascii") + original["BOOTSTRAP"].encode("ascii")
    auth.need(len(program) <= 131072, "ONE_FINITE_SOURCE_ONLY_OWNER_PAYLOAD")
    compile(program, "<paired source-only bootstrap>", "exec")
    return body, program


def ship_once(scope, publication, snapshot, raw_3090, transport, check_publication, *,
              independent_readback_raw, independent_readback_sha):
    """Future entry point. Retains the original SQLite owner, cutoff and one held-transport semantics."""
    load_scope(scope)
    auth.need(0 < len(independent_readback_raw) <= 16384 and auth.sha(independent_readback_raw) == independent_readback_sha,
              "PINNED_INDEPENDENT_TARGET_IDENTITY")
    readback = json.loads(independent_readback_raw)
    auth.need(readback["state"] == "EXACT_FROZEN_PAIR_IDENTITY_READBACK" and readback["approved_scope"] == scope
              and readback["head_file_SHA256"] == auth.HEAD_FILE_SHA and readback["head_module_SHA256"] == auth.HEAD_MODULE_SHA
              and readback["inputs_SHA256"] == auth.INPUT_SHA and readback["base"] == auth.BASE
              and readback["source_actual_CPU_step"] == auth.CPU_STEP and readback["new_GPU"] == 0,
              "INDEPENDENT_TARGET_IDENTITY_BEFORE_QUEUE")
    auth.need(callable(transport) and callable(check_publication), "HELD_OPERATOR_TRANSPORT_AND_PUBLICATION_REQUIRED")
    original = original_module(PILOT, PILOT_SHA)
    func = original["ship"]
    label = "vg-1318-dev24-" + scope["request"]
    own_files = [str(file.relative_to(REPO).as_posix()) for file in sorted(HERE.iterdir())
                 if file.suffix in (".py", ".json")]
    legacy_pins = [PILOT, NEAR, AUDIT, auth.PUBLIC_3090, auth.TRAIN, auth.LEGACY_INDEX, MEMBERSHIP, ACTIVE_LEDGER,
                   SOURCE_ROUNDOFF.relative_to(REPO).as_posix()]
    legacy_pins.extend(json.loads(read_local(auth.LEGACY_INDEX)).keys())

    def actual_publication(repo, commit, paths):
        check_publication(repo, commit, sorted(set(own_files + legacy_pins)))

    def verified_frozen(repo, commit, *, epoch, deadline, **_):
        body, program = frozen(scope, commit, epoch, raw_3090)
        claim = {"case": "T12.domain-vggt-1318-dev24-paired", "request": scope["request"]}
        return claim, auth.sha(auth.encode(dict(scope=scope, spec=PLAN, proof_SHA256=independent_readback_sha))), auth.sha(body), program

    func.__globals__.update(REQUEST=scope["request"], PARENT=scope["parent"], NODE=scope["node"],
                            HERE=HERE, BASE=HERE.relative_to(REPO),
                            LABEL=label, DIRECTORY=Path("local-only") / (label + "-original"),
                            frozen=verified_frozen)
    return func(publication, snapshot, transport, repo=REPO, check_publication=actual_publication)


def main():
    """Only render the zero-device identity preflight from this CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--cpu-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    auth.need(args.output.parent.resolve(strict=True).is_relative_to(REPO / "local-only"), "PRIVATE_OUTPUT_PARENT")
    raw = auth.read(args.cpu_terminal.resolve(strict=True), 2100, auth.AUDIT3090_SHA)
    scope = auth.js(args.scope.resolve(strict=True))
    body, _ = frozen(scope, args.publication, 0, raw, preflight=True)
    with args.output.open("xb") as stream:
        stream.write(body)
    print(json.dumps({"bytes": len(body), "SHA256": auth.sha(body), "target_calls": 0}))


if __name__ == "__main__":
    main()
