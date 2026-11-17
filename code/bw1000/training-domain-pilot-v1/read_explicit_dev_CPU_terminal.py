"""Read only the original330 native-development-input CPU qualification."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014/results")
RUN = ROOT / "training-qwen-treasury-explicit-dev-CPU-v1"
COLS = "JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES"

def raw(name, limit=262144):
    p = RUN / name
    assert p.is_file() and not p.is_symlink() and p.stat().st_size <= limit
    return p.read_bytes()

def load(name):
    b = raw(name)
    return json.loads(b), hashlib.sha256(b).hexdigest()

outcome, outcome_sha = load("outcome.json")
intent, intent_sha = load("intent.json")
claim, claim_sha = load("source/CLAIM.json")
mf, mf_sha = load("source/SOURCE_FILES.json")
assert claim_sha == "63e64c0902051894d01bb8abe993bcea3c1bcfe6ec0aeac728519fd926468d69"
assert mf_sha == "c0efd831d4882f1b4dc7fbc1efed49a96fb3093abdefd56cf5d371cb9f292e9a"
assert len(mf["files"]) == 12
for name, pin in mf["files"].items():
    assert Path(name).name == name
    b = raw("source/" + name)
    assert len(b) == pin["bytes"] and hashlib.sha256(b).hexdigest() == pin["SHA256"]
assert {p.name for p in (RUN / "source").iterdir()} == set(mf["files"]) | {"SOURCE_FILES.json"}
assert intent["publication"] == "16b87337da6ef9a6059411c755383d60f418e631"
assert outcome["epoch"] == intent["epoch"] == 1794880852.881809
assert outcome["deadline"] == intent["deadline"] == outcome["epoch"] + 600
assert outcome["observed_epoch"] < outcome["deadline"]
assert intent["claim_SHA256"] == claim_sha and intent["maximum_invocations"] == 1
assert [x for x in intent["command"] if x.startswith("--time=")] == ["--time=00:08:00"]
assert outcome["accounting_columns"] == COLS and outcome["current_step_outcome"] == "TERMINAL"
text = subprocess.check_output(["sacct", "-n", "-P", "-j", "1087656", "-o", COLS], text=True, timeout=9)
rows = [line.split("|") for line in text.splitlines() if line.split("|")[0] == "1087656.330"]
assert rows == [outcome["accounting"]["row"]]
assert rows[0][1:4] == ["training-qwen-treasury-explicit-dev-CPU-v1", "COMPLETED", "0:0"]
assert rows[0][5:] == ["a01r4n14", "cpu=8,mem=12G,node=1"]
log = raw("step.log", 4 * 1024**2)
assert len(log) == outcome["log_bytes"] and hashlib.sha256(log).hexdigest() == outcome["bounded_log_SHA256"]
metrics = []
for line in log.splitlines():
    if line.startswith(b"{"):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if value.get("state") == "NATIVE_CPU_DEVELOPMENT_INFERENCE_INPUTS_PASS":
            metrics.append(value)
assert metrics == [outcome["metrics"]]
assert outcome["passed"] and outcome["rc"] == 0 and not outcome["client_timeout"]
inventory, inventory_sha = load("inventory.json")
assert inventory_sha == "07d906e3ed840d828ee41509cae21b45a754ddef0825d411ca3d8216f606b0d8"
for name, pin in inventory.items():
    if name.startswith("media/"):
        b = (ROOT / "training-qwen-treasury-dev-native-CPU-v1/dev" / name).read_bytes()
    else:
        b = raw("dev/" + name)
    assert hashlib.sha256(b).hexdigest() == pin
m = outcome["metrics"]
assert m["dev_pins_SHA256"] == inventory_sha and m["actual_GPU_cards"] == m["actual_generations"] == 0
assert set(m["arms"]) == {"visual", "text_only"} and all(x["rows"] == 108 for x in m["arms"].values())
assert m["versions"] == {"ms-swift": "4.4.2", "qwen-vl-utils": "0.0.14", "transformers": "5.12.1"}
print(json.dumps(dict(root_independent_audit_passed=True, actual_accounting=rows[0],
    original_intent_SHA256=intent_sha, original_outcome_SHA256=outcome_sha,
    source_manifest_SHA256=mf_sha, claim_SHA256=claim_sha, source_inventory_verified=True,
    original_input_inventory_SHA256=inventory_sha, all39_original_metadata_and_media_verified=True,
    log_SHA256=outcome["bounded_log_SHA256"], epoch=outcome["epoch"], deadline=outcome["deadline"],
    metrics=m, formal_training_acceptance=False, actual_new_training_updates=0)))
