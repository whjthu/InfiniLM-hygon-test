"""Fixed source and target authority for the .1318 Dev24 pair."""

from hashlib import sha256
import json
from pathlib import Path
import re


PROJECT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
RESULTS = PROJECT / "results"
CPU_REQUEST = "46d30a11a7304ae38c1ce6f62286209b"
CPU_DIR = "training-vggt-original1318-dev24-cpu-readback-" + CPU_REQUEST
TRAIN_DIR = "training-vggt-tum-eight-pairs25-recovery-v2"
DEV_DIR = "training-vggt-tum-cross-camera-dev24-v1-staged"
ARM = "eight_pairs_1318_25"
CANDIDATE_TRAIN_REQUEST = "d7ff76e21b8248efbc8b0a9830c8344c"
CANDIDATE_STEP = "1103606.1318"
INPUT_SHA = "3e6401b9b3adc9d6516f8c75c33b827bcf71dcf0e20ccc258f75e8235b2affa6"
DEV_STAGE_SHA = "d221478fe51e14ccdf8ca858632c60aee67993e693c58f29b2c9357d373dc903"
BASE = {"weight": "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0",
        "camera": "90d2dc64acde99cd90bc4fb4a71a5fbda3de9267719c84153ff65a679cbaa202",
        "aggregator": "6590aab13e36a0309bd601226cc053c59fbef3f633ae3ce478fe329dd247aac4"}
HEAD = "arms/continuous/train/camera_head.pt"
TRAIN = "results/bw1000/2026-11-22/training-vggt-eight-pairs-original1318-native25-resume20-terminal.json"
TRAIN_SHA = "aee19d547e04bf99c3f0512e2e8ef07a335ef7f27bd6270e0a2dc41d32a52963"
PUBLIC_3090 = "results/bw1000/2026-11-25/training-astra-sol-round16-640-hour-vggt-n02-and-ordinary58-actual.json"
PUBLIC_3090_SHA = "e8fe245a96570ba0f18dbaac1fd37a173bec69da7743f2a4f5ebd313dc757f8b"
AUDIT3090_SHA = "67bc149245b2c1efd3e845726a0580a2cb6bd1b9488b00b85a7f4598176d8f40"
CPU_RESULT_SHA = "347ff8c65eb95961cb3fe4b4946f324e2e08f019b46b5afa0d56325ea869bc0a"
HEAD_READER_SHA = "bd33423085dad14de6e9e280f1c05714f1d52e0f796e0315ee95fec6b88d9666"
HEAD_FILE_SHA = "da50b4d1d650977da2b0fa5fb6aab680a263fc95a64b5e614ac9aff5d0439b2c"
HEAD_MODULE_SHA = "2d4c2983dd567e2283d61cb160a05d36876b9b425f5ad5a9d8f0d4f78bbb8985"
HEAD_NATIVE_SHA = "278422b9ddeb26b776e7e343ca394e4b418156ebf4f87a83cc0f26f870e55e42"
PROVIDER_SHA = "3954a96d84650573371fdce1b43d00dfc23f477b5df2b024db6cd867abab698e"
SHA = re.compile(r"[0-9a-f]{64}")
CPU_STEP = "1103606.3090"
READER = "tools/audit_vggt_tum_cross_camera_dev.py"
SCORER = "tools/score_vggt_tum_cross_camera_dev.py"
LEGACY_INDEX = "coordination/training/vggt-tum-cross-camera-controller-files-v1.json"
OLD_RUNS = {
    "1038": {"target": "training-vggt-tum-near7-head-dev24-pair-v1", "step": "1103606.978",
             "label": "near7_25", "score_request": "58dac28ab5514d5e91ae9109faa84f70",
             "source_index": "f20824ae94981459acf70d03421dd89a291e769f917d1633640aa10bc3173b81",
             "intent": "56787251bb985d778225ef986f3762368ac393984b0bba6f381ee0afbb270f19",
             "outcome": "b7cb970db2483bfc5f8462f04b7487d164222770d0cefb491cad8760b9f92c4f",
             "stage": "58c1f7669e90fd4f719db9cd2e35bb647ff1f986cc4252980b05623f3534dc9e",
             "progress": "9e24394764e9bc9a7396f07813c98394ade20f184f2d1f8de25f3be5f2aeedc6"},
    "1506": {"target": "training-vggt-tum-two-scene-bidir-dev24-pair-v2", "step": "1103606.498",
             "label": "two_scene_bidir25", "score_request": "6bd97bc61f0249c0a255ad26fe55a162",
             "source_index": "1bcb4fdbb1caabc8c483675d0b7a0c850ce6024cb0bef0cfd2c6dee57d4287de",
             "file": "be1be2219818a3af42c22d8973ea499f76400a06a1b3189fc0077ca8db23090a",
             "module": "f78b13bc07e62fbe1088838a39bbac6ce594abbe2cb21bede558fb381bfea718"},
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return sha256(raw).hexdigest()


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path, cap, wanted=None):
    need(path.is_file() and path.resolve(strict=True) == path.absolute(), "ORIGINAL_CANONICAL_FILE")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    need(0 < len(raw) <= cap and (wanted is None or sha(raw) == wanted), "ORIGINAL_BOUNDED_FILE_SHA")
    return raw


def js(path, want=None, cap=65536):
    return json.loads(read(path, cap, want))


def cpu_evidence(public_raw, private_raw):
    need(sha(public_raw) == PUBLIC_3090_SHA and sha(private_raw) == AUDIT3090_SHA and len(private_raw) == 1921,
         "ORIGINAL_3090_INDEPENDENT_EVIDENCE")
    public = json.loads(public_raw)
    receipt = json.loads(private_raw)
    row = public["VGGT"]["accounting_nine_columns_UTC"]
    response = receipt["response"]
    need(public["private_receipt_SHA256"]["vggt_terminal"] == AUDIT3090_SHA and receipt["rc"] == 0
         and receipt["request"] == response["request"] == CPU_REQUEST and receipt["replay_allowed"] is False
         and response["state"] == "PROVIDER_AND_SAME_22_OF_24_PROVENANCE_CHECKED"
         and response["accounting_nine_columns"] == row and row[0] == CPU_STEP and row[2:5] == ["COMPLETED", "0:0", "2"]
         and row[6] == "cpu=2,mem=2G,node=1" and response["exact_cpu_step_released"] is True
         and response["inputs"] == public["VGGT"]["inputs"] == {"actually_qualified": 22, "original_missing": 2,
                                                              "original_windows": 24, "reference_available": 22}
         and response["provider_receipt"] == {"SHA256": PROVIDER_SHA, "bytes": 1066}
         and response["provider"]["version"] == "1.25.0" and response["independently_bound_provider"] is True
         and response["predictions"] == response["optimizer_updates"] == response["GPU"] == 0,
         "ORIGINAL_3090_SCOPE_OR_DENOMINATOR")
    return response


def selected_head(evidence, train_raw):
    need(sha(train_raw) == TRAIN_SHA, "ORIGINAL_1318_PUBLIC_TRAINING")
    train = json.loads(train_raw)
    cpu = PROJECT / "incoming" / CPU_DIR / "evidence"
    intent = js(cpu / "INTENT.json", evidence["intent_SHA256"])
    provider = js(cpu / "PROVIDER.json", PROVIDER_SHA)
    actual = js(cpu / "RESULT.json", CPU_RESULT_SHA)
    binding = actual["binding"]
    proof = actual["audit_result"]
    need(actual["state"] == evidence["state"] and actual["provider_receipt"] == evidence["provider_receipt"]
         and binding == provider["binding"] and binding["request"] == intent["request"] == CPU_REQUEST
         and binding["step"] == CPU_STEP and binding["epoch"] == intent["epoch"] and binding["deadline"] == intent["deadline"]
         and intent["admission_SHA256"] == evidence["admission_SHA256"] and provider["provider"] == evidence["provider"]
         and proof["qualified_common_windows"] == actual["actual_target_inputs_qualified"] == 22
         and proof["original_windows"] == actual["frozen_dev24_windows"] == 24
         and actual["predictions"] == actual["optimizer_updates"] == actual["weight_payload_bytes"] == 0,
         "ORIGINAL_3090_PERSISTED_RESULT_OR_BINDING")
    need(set(proof["head_authority"]) == {ARM}, "EXACT_1318_HEAD_ONLY")
    head = proof["head_authority"][ARM]
    root = RESULTS / TRAIN_DIR
    merged = js(root / "native-result.json", train["native_result"]["SHA256"])
    outcome = js(root / "outcome.json", train["controller_terminal"]["SHA256"])
    receipt = js(root / "arms/continuous/receipt.json", train["arms"]["continuous"]["receipt"]["SHA256"])
    native = js(root / "arms/continuous/native-result.json", receipt["result_SHA256"])
    metrics = native["metrics"]
    exported = [r["sha256"] for r in native["artifacts"] if r["path"] == "/pilot/" + HEAD]
    path = root / HEAD
    need(train["request"] == outcome["request"] == head["request"] == CANDIDATE_TRAIN_REQUEST
         and outcome["stages"]["gpu"]["row"][0] == head["gpu_step"] == CANDIDATE_STEP
         and head["clean_post"] == outcome["stages"]["post"]["row"][0] == "1103606.1319"
         and merged["receipts"]["continuous"]["value"] == receipt
         and receipt["checkpoints"]["25"] == train["arms"]["continuous"]["checkpoint_records"]["25"]
         and metrics["global_optimizer_step"] == receipt["executed_optimizer_steps"] == head["global_step"] == 25
         and metrics["checkpoint_identity_evidence"]["frozen_model"]["weights_model_pt"] == BASE["weight"]
         and native["checks"]["frozen_aggregator_unchanged"] is True
         and exported == [head["file_SHA256"]] and SHA.fullmatch(head["file_SHA256"])
         and head["file_SHA256"] == HEAD_FILE_SHA and head["module_SHA256"] == HEAD_MODULE_SHA
         and head["module_SHA256"] == metrics["camera_head_sha_after"] and SHA.fullmatch(head["module_SHA256"])
         and head["original_continuous_result_SHA256"] == receipt["result_SHA256"] == HEAD_NATIVE_SHA
         and head["training_receipt_SHA256"] == train["arms"]["continuous"]["receipt"]["SHA256"]
         and head["original_combined_native_SHA256"] == train["native_result"]["SHA256"]
         and head["original_outcome_SHA256"] == train["controller_terminal"]["SHA256"]
         and path.is_file() and path.resolve(strict=True) == path and path.stat().st_size == head["head_bytes_stat"] == 864719552,
         "CONTINUOUS_25_CAMERA_HEAD_EXPORT_PROVENANCE")
    return head


def compatible(candidate, historical, execution_sha):
    """Tensor identity plus exactly the same base, inputs, and numerical scorer."""
    need(candidate["gpu_step"] == CANDIDATE_STEP and candidate["request"] == CANDIDATE_TRAIN_REQUEST,
         "CANDIDATE_STEP_IS_NEVER_A_PRECEDENT_ADAPTER")
    if candidate["file_SHA256"] == historical["file_SHA256"]:
        need(candidate["module_SHA256"] == historical["module_SHA256"], "FILE_AND_MODULE_IDENTITY_CONTRADICTION")
    return (candidate["module_SHA256"] == historical["module_SHA256"] and historical["base"] == BASE
            and historical["inputs_SHA256"] == INPUT_SHA and historical["scorer_execution_SHA256"] == execution_sha)


def verified_dev24():
    root = RESULTS / DEV_DIR
    stage = js(root / "stage-complete.json", DEV_STAGE_SHA)
    manifest = js(root / "inputs/inputs-private.json", INPUT_SHA)
    rows = manifest["windows"]
    source = stage["input_staging"]
    need(source["inputs_SHA256"] == INPUT_SHA and source["planned_windows"] == len(rows) == 24
         and source["input_pairs"] == 22 and source["verified_members"] == 23, "ORIGINAL_DEV24_STAGE")
    keys = set()
    available = []
    missing = []
    for row in rows:
        key = row["sequence"], row["window"]
        need(key not in keys, "UNIQUE_24_WINDOW_KEYS")
        keys.add(key)
        if row["state"] == "RGB224_AND_RELATIVE_POSE_BOUND":
            name = f"{key[0]}-window-{key[1]:02d}-rgb224.npy"
            need(row["rgb_file"] == name and len(read(root / "inputs" / name, 310000, row["rgb_file_SHA256"])) ==
                 row["rgb_file_bytes"], "EXACT_TARGET_RGB_BYTES")
            available.append(key)
        else:
            missing.append((*key, row["reason"]))
    need(len(available) == 22 and sorted(missing) == [
        ("freiburg1_plant", 1, "RGB_TIMESTAMP_GAP"), ("freiburg2_dishes", 1, "LOW_TRANSLATION")],
        "PRESERVE_22_INPUTS_AND_TWO_ORIGINAL_FAILURES")
    return stage, sorted(available), missing


def prior_pair(number, old_identity, executable_sha):
    old = OLD_RUNS[number]
    root = RESULTS / old["target"]
    source = PROJECT / "incoming" / old["target"]
    source_index = js(source / LEGACY_INDEX, old["source_index"])
    scorer = read(source / SCORER, 65536, source_index[SCORER])
    need(sha(scorer) == old_identity["full_scorer_SHA256"], "ORIGINAL_HISTORICAL_SCORER_BYTES")
    canonical = read(source / "code/bw1000/training-domain-pilot-v1/score_vggt_tum_pair.py", 65536,
                     old_identity["canonical_SHA256"])
    need(sha(canonical) == old_identity["canonical_SHA256"], "HISTORICAL_ORIGINAL_BASE_SCORER")
    intent = js(root / "score-intent.json", old.get("intent"))
    stage = js(root / "stage-complete.json", old.get("stage"))
    outcome = js(root / "score-outcome.json", old.get("outcome"))
    heads = js(root / "head-authority.json", intent["head_authority_SHA256"])
    head = heads[old["label"]]
    need(intent["request"] == outcome["request"] == old["score_request"]
         and intent["manifest_SHA256"] == old["source_index"] and intent["head_authority_SHA256"] == stage["head_authority_SHA256"]
         and stage["input_staging"]["inputs_SHA256"] == INPUT_SHA and old_identity["base"] == BASE
         and head["gpu_step"] == old["step"] and SHA.fullmatch(head["file_SHA256"]) and SHA.fullmatch(head["module_SHA256"])
         and outcome["stages"]["gpu"]["row"][0] == "1103606." + number and outcome["completed"] is True
         and (number != "1506" or (head["file_SHA256"], head["module_SHA256"]) == (old["file"], old["module"])),
         "EXACT_1038_OR_1506_SCORING_AND_TRAINING_IDENTITY")
    return dict(head, base=BASE, inputs_SHA256=INPUT_SHA, scorer_execution_SHA256=old_identity["execution_SHA256"],
                scorer_full_SHA256=sha(scorer), expected_semantic_match=old_identity["execution_SHA256"] == executable_sha)


def reuse_rows(number, available):
    old = OLD_RUNS[number]
    root = RESULTS / old["target"]
    intent = js(root / "score-intent.json", old.get("intent"))
    path = root / "score/progress-private.jsonl"
    source = read(path, 2 * 1024**2, old.get("progress"))
    events = [json.loads(line) for line in source.splitlines()]
    opening, *rows, closing = events
    need(opening["state"] == "PLANNED" and opening["inputs_SHA256"] == INPUT_SHA
         and opening["heads_SHA256"] == intent["head_authority_SHA256"] and len(opening["rows"]) == 24
         and closing["state"] == "COMPLETED" and closing["optimizer_updates"] == 0
         and closing["planned_windows"] == 24 and closing["input_pairs"] == 22
         and [(e["arm"], (e["sequence"], e["window"])) for e in rows] ==
         [(arm, key) for arm in ("base", old["label"]) for key in available], "EXACT_PRIOR_44_WINDOW_ROWS")
    return {"source_GPU_step": "1103606." + number, "original_progress_path": path.as_posix(),
            "progress_SHA256": sha(source), "progress_bytes": len(source), "existing_prediction_rows": len(rows),
            "independent_numeric_recompute_required": True, "new_GPU_invocations": 0}
