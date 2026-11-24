"""Read-only Dev24 input and source provenance for the original VGGT .1318 checkpoint."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
TRAIN = "training-vggt-tum-eight-pairs25-recovery-v2"
DEV = "training-vggt-tum-cross-camera-dev24-v1-staged"
PILOT = "training-vggt-tum-two-scene-bidir-dev24-pair-v2"
INPUT_SHA = "3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6"
INPUT_PATH = "inputs/inputs-private.json"
STAGING_SHA = "d221478fe51e14ccdf8ca858632c60aee67993e693c58f29b2c9357d373dc903"
BASE_WEIGHTS = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
BASE_CAMERA = "90d2dc64acde99cd90bc4fb4a71a5fbda3de9267719c84153ff65a679cbaa202"
BASE_AGGREGATOR = "6590aab13e36a0309bd601226cc053c59fbef3f633ae3ce478fe329dd247aac4"
HEAD = "arms/continuous/train/camera_head.pt"
CHECKPOINT = "arms/continuous/train/checkpoint-25.pt"
ORIGINAL_CHECKPOINT = "720b4adb7eea6173dd3fb2b9b834862fc371126031932fbe24e084efbeb352f0"
REQUEST = "d7ff76e21b8248efbc8b0a9830c8344c"
ARM = "eight_pairs_1318_25"
SCORER = "tools/score_vggt_tum_cross_camera_dev.py"
READER = "tools/audit_vggt_tum_cross_camera_dev.py"
SCORER_INDEX = "coordination/training/vggt-tum-cross-camera-controller-files-v1.json"
ROUNDOFF = "tools/read_vggt_dev24_roundoff_audit.py"
REFERENCE_BACKEND = "49bfe39b7aa6a14df9fdf1dda73e6db00836a81d2806012b312c16b2d3fbed6c"
UPSTREAM_PATH = RESULTS / "training-vggt-native-provider-HIP-CPU-v1/upstream"
BASE_PATH = RESULTS / "training-vggt-original-public-model-v1/model/model.pt"
FRAMEWORK_PATH = RESULTS / "training-qwen-a68952c-swift-initial-CPU-v1/framework/generation/src/gencuda/backends/vggt_training.py"
LOCAL_DATA = ROOT / "local-only/training-vggt-domain-eight-pairs25-v1/data"
PRIVATE_ZIP = ROOT / "local-only/training-vggt-tum-public-cross-camera-dev-trajectories-v1/VGGT-TUM-dev24-combined-private.zip"
PRIVATE_TERMINAL = ROOT / "local-only/training-vggt-tum-eight-pairs25-recovery-v2-original/independent-terminal.json"
PINS = {
    "training": ("results/bw1000/2026-11-22/training-vggt-eight-pairs-original1318-native25-resume20-terminal.json", "aee19d547e04bf99c3f0512e2e8ef07a335ef7f27bd6270e0a2dc41d32a52963"),
    "input": ("results/bw1000/2026-11-21/training-vggt-tum-both-official-scenes-combined-dev24.json", "3203cd451711e2963993b4431922defdcdbbd95fbde0edd4d1223c7426f0c5db"),
    "stage": ("results/bw1000/2026-11-22/training-vggt-dev24-target-staged-and-original-heads-bound.json", "f7dd656e1db03398489915817f69d4bb4cc8020a71086dea3774b74131ccc275"),
    "608": ("results/bw1000/2026-11-22/training-vggt-dev24-GPU608-three-heads-independent-auc.json", "829851a54c4e85c480525184ec6b1e308ded910a8fcbe385fd4bd0021f5a673a"),
    "1038": ("results/bw1000/2026-11-22/training-vggt-near7-head-original1038-dev24-terminal.json", "565a0482de717cbbfc1634323b47e0c71644c8bf78a117e58543258cd4a5e36f"),
    "1084": ("results/bw1000/2026-11-22/training-vggt-near7-bidir14-dev24-original1084-terminal.json", "0272f123d4f82a21ea5e9ff9ec0e0a4e9bf5e6114ff73bdc060731f3a29cdac8"),
    "1506": ("results/bw1000/2026-11-23/training-vggt-two-scene-dev24-v2-original1506-terminal.json", "5eca19c83c42ade3a61836285f4f4c5fdfb07621673d0de8a4b83bb24724f952"),
    "1078_head": ("code/bw1000/training-vggt-tum-near7-bidir14-dev24-pair-v1/HEAD_BINDING.json", "637d1fc289429db3b90ec031c05f97b8aa87aa74b81f1c4fa0d1409b730a674b"),
    "498_head": ("code/bw1000/" + PILOT + "/HEAD_BINDING.json", "a3483bcf491cb745cc45002c54a8f32e35becfe2663ff75a191f89eaed6a90f4"),
    "1506_launcher": ("code/bw1000/" + PILOT + "/launch.py", "5a4f895c109f19a92de737a9b01ee37650dc4e080a6f24b620f43be6c35aeb47"),
    "1506_decision": ("code/bw1000/" + PILOT + "/dev24_decision.py", "32c00078270c33ec626d3da748ec1963336c0521a4e19f9315c84e3932668389"),
}
SHA = re.compile(r"[0-9a-f]{64}")


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def encode(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def identity(raw):
    return {"bytes": len(raw), "SHA256": sha256(raw).hexdigest()}


def extent(path):
    value = path.lstat()
    need(stat.S_ISREG(value.st_mode) and path.resolve(strict=True) == path.absolute(), "NONCANONICAL_FILE")
    return value.st_size


def read(path, cap, pin=None):
    count = extent(path)
    need(0 < count <= cap, "EXACT_BOUNDED_EXTENT")
    with path.open("rb") as stream:
        raw = stream.read(count + 1)
    need(len(raw) == count and (pin is None or identity(raw)["SHA256"] == pin), "EXACT_MEMBER_DIGEST")
    return raw


def read_json(path, pin):
    if isinstance(pin, str):
        return json.loads(read(path, 131072, pin))
    raw = read(path, pin["bytes"], pin["SHA256"])
    need(len(raw) == pin["bytes"], "EXACT_METADATA_EXTENT")
    return json.loads(raw)


def public():
    out = {name: read_json(ROOT / relative, digest) for name, (relative, digest) in PINS.items() if not relative.endswith(".py")}
    train = out["training"]
    report = train["arms"]["continuous"]
    checkpoint = report["checkpoint_records"]["25"]
    need(train["request"] == REQUEST and train["actual_accounting"]["gpu"][:1] == ["1103606.1318"]
         and train["actual_accounting"]["gpu"][2:4] == ["COMPLETED", "0:0"]
         and train["actual_accounting"]["post"][:1] == ["1103606.1319"] and train["independent_post_clean"]
         and report["completed"] == 25 and checkpoint["file_SHA256"] == ORIGINAL_CHECKPOINT
         and checkpoint == train["arms"]["resume20"]["checkpoint_records"]["25"], "ORIGINAL_1318_STEP25")
    need(out["stage"]["independent_target"]["capture"]["inputs_SHA256"] == INPUT_SHA
         and out["stage"]["independent_target"]["stage_SHA256"] == STAGING_SHA
         and out["input"]["private_bundle"]["inputs_SHA256"] == INPUT_SHA
         and out["1506"]["source_training_step"] == "1103606.498"
         and out["1506"]["actual_terminal_accounting"]["gpu"][0] == "1103606.1506"
         and out["1506"]["seen_predictions"] == 44, "ORIGINAL_DEV24_SCOPE")
    need(out["608"]["rows"]["gpu"][0] == "1103606.608"
         and out["608"]["actual_prediction_records"] == 66
         and out["1038"]["rows"]["gpu"][0] == "1103606.1038"
         and out["1038"]["independent"]["actual_prediction_events"] == 44
         and out["1084"]["independent"]["accounting"]["gpu"][0] == "1103606.1084"
         and out["1084"]["independent"]["independent"]["actual_prediction_events"] == 44
         and out["1078_head"]["source_gpu_step"] == "1103606.1078"
         and out["498_head"]["source_gpu_step"] == "1103606.498", "EXACT_EARLIER_DEV24_SCORE_HISTORY")
    return out


def training_recordings(root, pin):
    bundle = read_json(root / "bundle.json", pin)
    need(bundle["schema"] == "vggt-tum-eight-training-pairs-v1" and len(bundle["pairs"]) == bundle["training_pair_count"] == 8,
         "ORIGINAL_TRAINING_COUNT")
    rows = []
    for item in bundle["pairs"]:
        need(item["manifest"] == item["root"] + "/pair.json" and "/" not in item["root"], "TRAINING_MANIFEST_PATH")
        record = read_json(root / item["manifest"], item["manifest_SHA256"])
        need(record["sequence"] == item["sequence"] and len(record["rgb_indices"]) == 2,
             "TRAIN_RECORDING_OR_PAIR_IDENTITY")
        rows.append({"recording": record["sequence"], "manifest_SHA256": item["manifest_SHA256"]})
    need(len({row["manifest_SHA256"] for row in rows}) == 8 and len({row["recording"] for row in rows}) == 2,
         "ORIGINAL_PHYSICAL_RECORDING_DENOMINATOR")
    return rows


def pinned_scorer():
    source, wanted = PINS["1506_launcher"]
    raw = read(ROOT / source, 65536, wanted)
    tree = ast.parse(raw)
    names = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    body = names["controller_sources"].body
    start, = (i for i, node in enumerate(body) if isinstance(node, ast.Assign)
              and ast.unparse(node.targets[0]) == "scorer_path")
    stop, = (i for i, node in enumerate(body) if isinstance(node, ast.Assign)
             and ast.unparse(node.targets[0]) == "sources[scorer_path]")
    index = read_json(ROOT / SCORER_INDEX, "f224f502dc9b5755182f458591e47cf5214f7484dc0c0480d4028d5dade0a910")
    paths = (SCORER, READER, "tools/bind_vggt_tum_cross_camera_heads.py", "tools/bind_vggt_tum_dev_rgb.py",
             "tools/plan_vggt_tum_public_dev_trajectories.py", "code/bw1000/training-domain-pilot-v1/score_vggt_tum_pair.py")
    sources = {path: read(ROOT / path, 65536, index[path]) for path in paths}
    code = ast.Module(body=[names["replace_unique"], *body[start:stop + 1]], type_ignores=[])
    exec(compile(code, "<original .1506 scorer-only renderer>", "exec"), {"sources": sources, "need": need})
    return sources


def inputs_and_overlap(manifest_raw, read_input, recordings, scorer_raw):
    need(identity(manifest_raw)["SHA256"] == INPUT_SHA, "FROZEN_DEV24_INPUTS")
    document = json.loads(manifest_raw)
    windows = document["windows"]
    trained = {row["recording"] for row in recordings}
    scope = {"freiburg1_plant", "freiburg2_dishes"}
    need(set(document["scenes"]) == scope and len(windows) == 24
         and {(row["sequence"], row["window"]) for row in windows} == {(seq, win) for seq in scope for win in range(1, 13)},
         "FROZEN_24_WINDOW_DENOMINATOR")
    virtual = Path("/dev24-readonly/inputs-private.json")
    scorer = {"__file__": str(ROOT / SCORER), "__name__": "vg_gt_dev24_existing_scorer"}
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        exec(compile(scorer_raw, str(ROOT / SCORER), "exec"), scorer)
    finally:
        sys.path.pop(0)
    canonical = scorer["canonical_scorer"]()
    original_reader = scorer["read_bounded"]

    def mounted(path, bound):
        if path.parent != virtual.parent:
            return original_reader(path, bound)
        if path == virtual:
            return manifest_raw
        return read_input(path.name, bound)

    scorer["read_bounded"] = mounted
    actual, _ = scorer["read_cases"](virtual, INPUT_SHA, canonical["relative_pose"])
    byte_verified = {(row["sequence"], row["window"]): row["input_SHA256"] for row, _, _ in actual}
    statuses = []
    for row in sorted(windows, key=lambda item: (item["sequence"], item["window"])):
        key = row["sequence"], row["window"]
        overlap = key[0] in trained
        state = "TRAIN_RECORDING_OVERLAP" if overlap else ("COMMON_EVAL" if key in byte_verified else "ORIGINAL_MISSING")
        statuses.append(dict(recording=key[0], window=key[1], state=state, original_input_state=row["state"],
                             original_failure=row.get("reason"), rgb_SHA256=byte_verified.get(key), train_recording_overlap=overlap))
    return statuses


def prior_known(proof):
    scored = {name: record for name, record in proof["stage"]["head_ids"].items()}
    scored["near7_bidir25"] = proof["1078_head"]
    scored["two_scene_bidir25"] = proof["498_head"]
    return [{"training_gpu_step": row.get("gpu_step", row.get("source_gpu_step")),
             "file_SHA256": row["file_SHA256"], "module_SHA256": row["module_SHA256"]} for row in scored.values()]


def check_target(proof, score):
    target = RESULTS / TRAIN
    train = proof["training"]
    outcome = read_json(target / "outcome.json", train["controller_terminal"])
    native = read_json(target / "native-result.json", train["native_result"])
    receipt_name = "arms/continuous/receipt.json"
    receipt = read_json(target / receipt_name, train["arms"]["continuous"]["receipt"])
    recorded = native["receipts"]["continuous"]
    need(recorded["value"] == receipt and recorded["SHA256"] == train["arms"]["continuous"]["receipt"]["SHA256"]
         and outcome["request"] == REQUEST and outcome["passed"] is True
         and outcome["result_SHA256"] == train["native_result"]["SHA256"] and native["formal_CO3D_eligible"] is False
         and all(outcome["stages"][name]["row"] == row for name, row in train["actual_accounting"].items())
         and receipt["checkpoints"]["25"] == train["arms"]["continuous"]["checkpoint_records"]["25"]
         and receipt["arm"] == "continuous" and receipt["executed_optimizer_steps"] == 25, "EXACT_ORIGINAL_1318_TARGET")
    result_path = target / "arms/continuous/native-result.json"
    result = read_json(result_path, receipt["result_SHA256"])
    metrics = result["metrics"]
    assets = [a["sha256"] for a in result["artifacts"] if a["path"] == "/pilot/" + HEAD]
    need(len(assets) == 1 and SHA.fullmatch(assets[0]) and SHA.fullmatch(metrics["camera_head_sha_after"])
         and metrics["checkpoint_identity_evidence"]["frozen_model"]["weights_model_pt"] == BASE_WEIGHTS
         and metrics["training_scope"] == "camera_head" and metrics["cached_valid_batches"] == 8
         and metrics["global_optimizer_step"] == 25 and result["checks"]["frozen_aggregator_unchanged"] is True,
         "ORIGINAL_NATIVE_HEAD_EXPORT")
    head_bytes = extent(target / HEAD)
    need(0 < head_bytes <= 4 * 1024**3 and extent(target / CHECKPOINT) > head_bytes, "ORIGINAL_WEIGHT_EXTENTS")
    need(extent(BASE_PATH) == 5026874952, "ORIGINAL_BASE_WEIGHT_EXTENT")
    read(FRAMEWORK_PATH, 512 * 1024, REFERENCE_BACKEND)
    provider = [node for node in ast.parse(score).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PROVIDERS" for target in node.targets)]
    for relative, pin in ast.literal_eval(provider[0].value).items():
        read(UPSTREAM_PATH / "vggt" / relative, 65536, pin)
    dev = RESULTS / DEV
    stage = read_json(dev / "stage-complete.json", STAGING_SHA)
    need(stage["input_staging"]["inputs_SHA256"] == INPUT_SHA, "TARGET_DEV24_STAGE_IDENTITY")
    recordings = training_recordings(target / "data", train["data"]["media_bundle_SHA256"])
    windows = inputs_and_overlap(read(dev / INPUT_PATH, 30000, INPUT_SHA),
                                 lambda name, cap: read((dev / INPUT_PATH).parent / name, cap), recordings, score)
    all_scored = prior_known(proof)
    prior = RESULTS / "training-vggt-tum-near7-head-dev24-pair-v1"
    score_intent = read_json(prior / "score-intent.json", proof["1038"]["score_intent_SHA256"])
    near7_stage = read_json(prior / "stage-complete.json", proof["1038"]["stage_SHA256"])
    need(score_intent["request"] == proof["1038"]["request"] and near7_stage["input_staging"]["inputs_SHA256"] == INPUT_SHA
         and near7_stage["head_authority_SHA256"] == score_intent["head_authority_SHA256"], "ORIGINAL_1038_SCOPE")
    near7 = read_json(prior / "head-authority.json", score_intent["head_authority_SHA256"])["near7_25"]
    need(near7["gpu_step"] == proof["1038"]["trained_head"] and SHA.fullmatch(near7["file_SHA256"])
         and SHA.fullmatch(near7["module_SHA256"]), "ORIGINAL_1038_HEAD")
    all_scored.append(dict(training_gpu_step=near7["gpu_step"], file_SHA256=near7["file_SHA256"], module_SHA256=near7["module_SHA256"]))
    if any(item["file_SHA256"] == assets[0] or item["module_SHA256"] == metrics["camera_head_sha_after"] for item in all_scored):
        return {"state": "STOP_EXACT_HEAD_AND_DEV24_ALREADY_SCORED", "predictions": 0}
    common = sum(row["state"] == "COMMON_EVAL" for row in windows)
    if common == 0:
        return {"state": "STOP_NO_SEQUENCE_DISJOINT_INPUT", "predictions": 0}
    need(common == 22, "ORIGINAL_1506_SCORING_REQUIRES_THE_SAME_22_DISJOINT_INPUTS")
    head = dict(gpu_step="1103606.1318", request=REQUEST, clean_post="1103606.1319", global_step=25,
                training_receipt_SHA256=train["arms"]["continuous"]["receipt"]["SHA256"],
                original_outcome_SHA256=train["controller_terminal"]["SHA256"],
                original_combined_native_SHA256=train["native_result"]["SHA256"],
                original_continuous_result_SHA256=receipt["result_SHA256"], file_SHA256=assets[0],
                module_SHA256=metrics["camera_head_sha_after"], head_bytes_stat=head_bytes, actual_checkpoint_bytes_read=0)
    return {"state": "READ_ONLY_HEAD_RECORDING_AND_INPUT_PROVENANCE_CHECKED", "head_authority": {ARM: head},
            "qualified_common_windows": common, "original_windows": len(windows), "previous_same_input_head_identities": len(all_scored),
            "actual_weight_bytes_read": 0, "predictions": 0}


def replace(raw, old, new):
    need(raw.count(old) == 1, "EXACT_1506_SOURCE_ANCHOR")
    return raw.replace(old, new, 1)


def candidate_source(proof, base):
    receipt = proof["training"]["arms"]["continuous"]["receipt"]["SHA256"]
    old = b'ARMS = {"two_scene_bidir25": ("1103606.498", "ab8a312813524d949608defa7124c3bf", "1103606.507", "4abc5ca7299dccc3915151f5c064d7ff590e67208d16e5b14fb107195fd74af6")}'
    declaration = repr({ARM: ("1103606.1318", REQUEST, "1103606.1319", receipt)}).encode()
    base = replace(base, old, b"ARMS = " + declaration)
    return replace(base, b'roots = {"two_scene_bidir25": args.candidate}', ("roots = {" + repr(ARM) + ": args.candidate}").encode())


def prior_checkpoint_search(checkpoint):
    needles = (checkpoint["file_SHA256"], checkpoint["model_SHA256"], checkpoint["full_boundary_SHA256"])
    scopes = (ROOT / "results/bw1000", ROOT / "code/bw1000", ROOT / "coordination")
    exact = set()
    for scope in scopes:
        for path in scope.rglob("*"):
            if path.suffix not in {".json", ".md", ".py"} or path.parent == Path(__file__).resolve().parent:
                continue
            if "vggt" not in path.as_posix().lower() or path.stat().st_size > 2 * 1024**2:
                continue
            raw = path.read_bytes()
            if any(needle.encode() in raw for needle in needles):
                exact.add(path.relative_to(ROOT).as_posix())
    return sorted(exact)


def render_layout(output, sources, score):
    decision_path, decision_sha = PINS["1506_decision"]
    decision = read(ROOT / decision_path, 8192, decision_sha)
    decision = replace(decision, b'ARMS = ("base", "two_scene_bidir25")',
                       ("ARMS = " + repr(("base", ARM))).encode())
    decision = replace(decision, b"STOP_TWO_SCENE_AUGMENTATION", b"STOP_ORIGINAL_1318_AUGMENTATION")
    decision += b"\nif __name__ == '__main__':\n    import json, sys\n    from pathlib import Path\n    print(json.dumps(evaluate(json.loads(Path(sys.argv[1]).read_bytes())), sort_keys=True, allow_nan=False))\n"
    roundoff = read(ROOT / ROUNDOFF, 16384, "4a0fa432a61734ac21c102c274a1028217d6a212e9ba8b19acfd83c054f43d97")
    package = dict(sources, **{SCORER: score, ROUNDOFF: roundoff, "tools/dev24_decision_1318.py": decision})
    output.mkdir(mode=0o700)
    for name, raw in package.items():
        compile(raw, name, "exec")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
    members = {name: identity(raw) for name, raw in sorted(package.items())}
    with (output / "SOURCE_FILES.json").open("xb") as handle:
        handle.write(encode(members))
    return identity(encode(members))


def local(proof, score):
    train = proof["training"]
    independent = read_json(PRIVATE_TERMINAL, train["independent_terminal_receipt"])
    need(independent["arms"] == train["arms"] and independent["post_clean"] is True, "EXACT_PRIVATE_1318_ORIGINAL")
    training = training_recordings(LOCAL_DATA, train["data"]["media_bundle_SHA256"])
    packet = proof["input"]["private_bundle"]
    bundle = read(PRIVATE_ZIP, packet["bundle_bytes"], packet["bundle_SHA256"])
    import io
    with ZipFile(io.BytesIO(bundle)) as archive:
        need(len(archive.namelist()) == len(set(archive.namelist())) == 23, "FROZEN_ZIP_INVENTORY")
        manifest = archive.read("inputs-private.json")
        def member(name, cap):
            item = archive.getinfo(name)
            need(0 < item.file_size <= cap, "DEV24_ARCHIVE_MEMBER_EXTENT")
            return archive.read(item)
        windows = inputs_and_overlap(manifest, member, training, score)
    common = sum(row["state"] == "COMMON_EVAL" for row in windows)
    previous_mentions = prior_checkpoint_search(train["arms"]["continuous"]["checkpoint_records"]["25"])
    result = {"state": "LOCAL_SOURCE_AND_DEV24_RECORDING_AUDIT_COMPLETE", "windows": windows,
              "training_pairs": training, "original_dev24": 24, "common_pose_input_windows": common,
              "train_recording_overlap_windows": sum(row["train_recording_overlap"] for row in windows),
              "missing_original_windows": sum(row["original_failure"] is not None for row in windows),
              "shared_arm_prediction_schedule": ["base", ARM], "paired_prediction_slots_if_admitted": common * 2,
              "original_missing_scoring_slots": (24-common)*2, "actual_predictions": 0,
              "original_train_receipt_SHA256": PINS["training"][1], "input_manifest_SHA256": INPUT_SHA,
              "original_1318_checkpoint_filename": CHECKPOINT, "original_1318_checkpoint_file_SHA256": ORIGINAL_CHECKPOINT,
              "original_1318_checkpoint_model_tree_SHA256": train["arms"]["continuous"]["checkpoint_records"]["25"]["model_SHA256"],
              "original_1318_head_export_file_SHA256": None, "original_1318_head_module_SHA256": None,
              "deduplication": "TARGET_EXPORT_AND_ORIGINAL_1038_AUTHORITY_REQUIRED",
              "already_scored_same_input_known_export_and_module_pins": prior_known(proof),
              "source_1038_head_currently_unresolved": True, "exact_checkpoint_pin_references": previous_mentions,
              "same_complete_checkpoint_sha_found_in_reviewed_scoring_evidence": any(
                  name != PINS["training"][0] for name in previous_mentions),
              "target_read": False, "actual_optimizer_updates": 0, "Slurm": 0, "GPU": 0}
    if common == 0:
        result["state"] = "STOP_NO_SEQUENCE_DISJOINT_INPUT"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("local", "target-read-only", "render-layout"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    proof = public()
    source = pinned_scorer()
    raw = candidate_source(proof, source[SCORER])
    compile(raw, SCORER, "exec")
    if args.mode == "target-read-only":
        need(sys.platform == "linux" and RESULTS.is_dir() and args.output is None, "READ_ONLY_REAL_TARGET_ONLY")
        print(encode(check_target(proof, raw)).decode(), end="")
    else:
        need(args.output is not None and args.output.parent.resolve(strict=True).is_relative_to(ROOT / "local-only"),
             "PRIVATE_OUTPUT_ONLY")
        if args.mode == "render-layout":
            print(json.dumps(render_layout(args.output, source, raw), sort_keys=True))
            return
        result = encode(dict(local(proof, raw), rendered_scorer=identity(raw)))
        with args.output.open("xb") as handle:
            handle.write(result)
        print(json.dumps(identity(result), sort_keys=True))


if __name__ == "__main__":
    main()
