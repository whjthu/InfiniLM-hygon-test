"""Run identity and frozen inputs for the two-node GLM FFN observation."""

import hashlib
import json
import math
import os
from pathlib import Path
import re

from native_arm import ARMS, require, sha256
from budget import GPU_SECONDS, INSIDE_SECONDS, WORK_SECONDS, PAIR_SECONDS

D = Path(__file__).resolve().parent
R = Path("/refactor")
HOST = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
PARENT = "1103606"
NODES = ["f13r1n05", "f13r1n07"]
IPS = ["10.6.13.5", "10.6.13.7"]
PORT = 43844
FROZEN_BINDINGS = {
    "CPU_BINDING.json": "5be453bedb4a2ea4cca09c07aedef9125921c52355bb209248c77a73a6eeeda7",
    "INPUT_BINDING.json": "81f6b2d68228aafb79993306847b0edb04c1ba101a35ab21bb822ac96ca2ba5e",
    "TRIAL.json": "ead4535a11ef738268d9b029bbca212226da8eb83859e0872022bf60f66c4658",
}
REQUIRED_FILES = frozenset((
    "budget.py", "control.py", "shared.py", "inside.sh", "ENV.sh", "owner.py", "native_arm.py",
    "CPU_BINDING.json", "INPUT_BINDING.json", "TRIAL.json", "checkpoint.py",
    "observer_trial.py", "prefill_schedule.py", "ready_socket_parent.py",
    "resident_trial.py", "scoring.py", "typed_binding.py",
    "bootstrap.py", "model_transport.py", "verify_transport.py", "ffn_evidence.py",
))


def namespace(run_id, arm):
    require(re.fullmatch(r"[0-9a-f]{32}", run_id) is not None, "invalid paired run ID")
    require(arm in ARMS, "invalid paired Native arm")
    label = f"GLM474-ffn-{run_id}-{arm}"
    return label, R / "results" / label


def step_id():
    require(os.environ.get("SLURM_JOB_ID") == PARENT, "wrong Slurm parent")
    step = os.environ["SLURM_STEP_ID"]
    require(re.fullmatch(r"[0-9]+", step) is not None, "invalid Slurm step")
    return PARENT + "." + step


def H(path):
    return sha256(Path(path))


def save(path, value):
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def load_pinned(path, digest):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, f"{path.name}: hash mismatch")
    return json.loads(raw)


def verify_manifest(manifest, run_id, arm, epoch):
    _, out = namespace(run_id, arm)
    require(math.isfinite(epoch) and epoch > 0, "invalid launch epoch")
    expected = dict(
        schema="glm474-ffn-worker-v1", run_id=run_id, arm=arm, epoch=epoch,
        parent=PARENT, nodes=NODES, output=str(out), Native_SHA256=ARMS[arm][1],
        GPU_GO=True, Model_execution_authorized=True, Model_qualified=False,
    )
    require(
        manifest.keys() == expected.keys() | {"source_manifest_SHA256", "claim_SHA256", "transport_key_SHA256"}
        and re.fullmatch(r"[0-9a-f]{64}", manifest["transport_key_SHA256"]) is not None
        and all(type(manifest[k]) is type(v) and manifest[k] == v for k, v in expected.items()),
        "run manifest does not match selected arm and resources",
    )


def verify_inputs(inp, trial, plan):
    from resident_trial import phases, schedule

    require(
        trial["planned"] == 33 and trial["schedule"] == schedule()
        and trial["phases"] == phases() and trial["capacity"] == 32768
        and trial["memory_floor_bytes"] == 8 * 1024**3
        and trial["max_new_tokens_each"] == 4096,
        "request schedule changed",
    )
    rows = plan["requests"]
    require(
        plan["planned"] == len(rows) == 32
        and [row["ordinal"] for row in rows] == inp["selected_ordinals"]
        and len({row["case_id"] for row in rows}) == 32
        and sum(row["suite"] == "gsm8k" for row in rows) == 16
        and sum(row["suite"] == "mmlu" for row in rows) == 16,
        "private screen cohort changed",
    )
    pins = inp["private_plan"]["requests"]
    require(len(pins) == len(rows), "private screen request pin count changed")
    for row, pin in zip(rows, pins):
        ids = row["prompt_token_ids"]
        require(
            all(row[k] == value for k, value in pin.items())
            and row["body"].get("temperature") == 0.0
            and row["max_new_tokens"] == row["prefill_window"] == 4096
            and 0 < len(ids) == row["prompt_tokens"] <= 32768 - 4096
            and hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            == pin["prompt_IDs_SHA256"],
            f"request {row['ordinal']}: input changed",
        )


def verify_entry(pin, arm):
    run_id = os.environ["GLM474_PAIR_ID"]
    require(arm == os.environ["GLM474_PAIR_ARM"], "owner/worker arm mismatch")
    _, out = namespace(run_id, arm)
    epoch = float(os.environ["GLM470_EPOCH"])
    step_id()
    manifest = load_pinned(out / "RUN_MANIFEST.json", pin)
    verify_manifest(manifest, run_id, arm, epoch)
    require(os.environ.get("GLM474_TRANSPORT") == arm, "transport selection differs from arm")
    key = os.environ.get("GLM474_TRANSPORT_KEY", "")
    require(re.fullmatch(r"[0-9a-f]{64}", key) is not None
            and hashlib.sha256(bytes.fromhex(key)).hexdigest() == manifest["transport_key_SHA256"],
            "transport key identity differs from run manifest")
    claim = load_pinned(D / "GPU_RESOURCE_CLAIM.json", manifest["claim_SHA256"])
    expected_claim = dict(parent=PARENT, nodes=NODES, GPU_cards=16, CPU_per_node=64,
                          memory_MiB_per_node=224000, GPU_seconds=GPU_SECONDS,
                          work_seconds=WORK_SECONDS, inside_seconds=INSIDE_SECONDS, whole_seconds=PAIR_SECONDS, max_runs_per_arm=1, GPU_GO=True,
                          schema="glm474-ffn-model-claim-v1", arm_order=["socket"],
                          model_context=32768, memory_floor_bytes=8 * 1024**3,
                          max_new_tokens=4096, planned_requests_per_arm=33)
    require(
        all(type(claim[k]) is type(v) and claim[k] == v for k, v in expected_claim.items())
        and claim["source_manifest_SHA256"] == manifest["source_manifest_SHA256"]
        and claim["run_id"] == run_id
        and claim["native_arms"] == {k: v[1] for k, v in ARMS.items()},
        "claim does not authorize this paired entry",
    )
    source = load_pinned(D / "SOURCE_MANIFEST.json", manifest["source_manifest_SHA256"])
    files = source["files_SHA256"]
    require(files.keys() == REQUIRED_FILES, "worker source set changed")
    require(all(files[name] == digest for name, digest in FROZEN_BINDINGS.items()),
            "frozen input or provider bindings changed")
    for name, digest in files.items():
        require(H(D / name) == digest, f"{name}: source mismatch")
    inp = json.loads((D / "INPUT_BINDING.json").read_bytes())
    private = inp["private_plan"]
    private_path = R / Path(private["path"]).relative_to(HOST)
    require(private_path.stat().st_size == private["bytes"], "private plan length changed")
    plan = load_pinned(private_path, private["SHA256"])
    verify_inputs(inp, json.loads((D / "TRIAL.json").read_bytes()), plan)
    model = Path(inp["model_path"])
    require(H(model / "config.json") == inp["model_config_SHA256"], "model config changed")
    require(H(model / "model.safetensors.index.json") == inp["model_index_SHA256"], "model index changed")
    require(H(inp["tokenizer"]["path"]) == inp["tokenizer"]["SHA256"], "tokenizer changed")
    return manifest, inp, plan
