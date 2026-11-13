"""Run one published SAGE-GRPO reference capture on the original A100 host."""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
import time

home = Path("/home/whj/nvidia-baseline")
original = home / "alignment-assets/hygon-hunyuan-sage-first-policy-A100-v1"
work = home / "alignment-assets/hygon-hunyuan-sage-fixed-replay-A100-v1"
reference = original / "results/hunyuanvideo-1.5/rl/first-policy/trajectories/step-0001.pt"
reference_sha = "55a879cafa7880840ac0aead8dec4fd4c55f3732e041dcfbfbad0cba790dccff"
assert reference.stat().st_size == 21983599
assert hashlib.sha256(reference.read_bytes()).hexdigest() == reference_sha
base = Path("/mnt/data/nvidia-baseline-20261010/sources")
scheduler = base / "sage-grpo/hyvideo/schedulers/scheduling_flow_match_discrete.py"
assert hashlib.sha256(scheduler.read_bytes()).hexdigest() == "56f61a5b36275528392d04c259902642291bcfcb69c9f4f4f4f076d0c148b943"

def gpu_query(fields):
    return subprocess.check_output(["nvidia-smi", fields, "--format=csv,noheader,nounits"],
                                   text=True, timeout=10).strip().splitlines()

def verify_idle(expected=None):
    cards = [row.split(", ") for row in gpu_query("--query-gpu=index,uuid,pci.bus_id,name,memory.used,utilization.gpu")]
    selected = {row[0]: row for row in cards if row[0] in ("0", "1")}
    assert set(selected) == {"0", "1"}
    assert all(row[3] == "NVIDIA A100-SXM4-40GB" and int(row[4]) <= 16 and row[5] == "0"
               for row in selected.values())
    apps = {row.split(",")[0] for row in gpu_query("--query-compute-apps=gpu_uuid,pid")}
    assert not {row[1] for row in selected.values()} & apps
    identities = {index: row[1:3] for index, row in selected.items()}
    assert expected is None or identities == expected
    return identities

with tarfile.open(ARCHIVE, mode="r:gz") as archive:
    member = archive.getmember(PIN["path"])
    assert member.isfile() and member.size == PIN["bytes"] <= 4096
    with archive.extractfile(member) as stream:
        raw = stream.read(PIN["bytes"] + 1)
assert len(raw) == PIN["bytes"] and hashlib.sha256(raw).hexdigest() == PIN["sha256"]
launch = json.loads(raw)
config = launch["config"]
assert config["cards"] == [0, 1] and config["num_generations"] == config["sampling_steps"] == 4
assert Path(launch["source"]) == base / "hunyuanvideo-1.5"
assert shutil.disk_usage(work.parent).free >= 8 * 1024**3
identities = verify_idle()
work.mkdir(mode=0o700)
epoch = time.time()

def write(name, data):
    with (work / name).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)

intent = dict(request_id=REQUEST, claim=CLAIM, epoch=epoch, deadline=epoch + 1800,
              gpu_identities=identities, framework_sha256=hashlib.sha256(FRAMEWORK_TAR).hexdigest())
write("intent.json", intent)
print(json.dumps(dict(stage="intent", request_id=REQUEST)), flush=True)
with tarfile.open(fileobj=io.BytesIO(FRAMEWORK_TAR)) as archive:
    for member in archive:
        path = Path(member.name)
        assert not path.is_absolute() and ".." not in path.parts
        if member.isdir():
            continue
        assert member.isfile() and path.parts[0] in {"generation", "tools"}
        destination = work / "framework" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
(work / "launch.json").write_bytes(raw)
sys_path = work / "framework/tools"
import sys
sys.path.insert(0, str(sys_path))
import hygon_training_recipe as recipe
listing = recipe.build_generation_plan("T09.sage-grpo", raw, None)
write("assets.json", {path: path for path in listing["original_assets"]})
arguments = ["T09.sage-grpo", "--a100-launch", str(work/"launch.json"), "--asset-map", str(work/"assets.json"),
             "--run", "--replay-first-trajectory", str(reference), "--replay-first-trajectory-sha256", reference_sha, "--cards", "0,1", "--source-root", str(base),
             "--output-root", str(work/"results"), "--run-name", "first-policy",
             "--lock-root", str(home/"alignment-assets/hygon-card-leases"), "--lock-host", socket.gethostname()]
command = ["/home/whj/nvidia-baseline/env-vlm/bin/python", "-B", "-u", "-c",
           "import sys;sys.path.insert(0,sys.argv[1]);import hygon_training_recipe as r;"
           "raise SystemExit(r.main(sys.argv[2:],generation_accelerator='cuda'))", str(sys_path), *arguments]
environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
environment.pop("CUDA_VISIBLE_DEVICES", None)
verify_idle(identities)
remaining = int(intent["deadline"] - time.time()) - 20
assert remaining > 0
with (work / "runner.log").open("xb") as log:
    child = subprocess.run(["timeout", "--kill-after=10", str(remaining), *command], cwd=work,
                           env=environment, stdout=log, stderr=subprocess.STDOUT)
output = work/"results/hunyuanvideo-1.5/rl/first-policy"
files = {}
for name in ("manifest.json", "first-policy-update.json"):
    path = output / name
    if path.is_file():
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        files[name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
passed = False
if child.returncode == 0 and len(files) == 2:
    policy = json.loads((output/"first-policy-update.json").read_bytes())
    manifest = json.loads((output/"manifest.json").read_bytes())
    passed = (manifest["status"] == "completed" and policy["global_step"] == 1
              and policy["trajectory"]["sha256"] == reference_sha
              and policy["adapter_before"] != policy["adapter_after"]
              and len(policy["rollouts"]) == 4)
result = dict(request_id=REQUEST, child_rc=child.returncode, files=files, passed=passed)
if passed:
    prior_path = reference.parent.parent / "first-policy-update.json"
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == "c1f8e0283e9ce7ad8f1dd64b7d6510d616a8fa8418e96d6f86f57d1f0b36f4d3"
    prior = json.loads(prior_path.read_bytes())
    result["adapter_after_identical"] = policy["adapter_after"] == prior["adapter_after"]
    result["reward_exact"] = policy["reward"] == prior["reward"]
    result["original_old_log_max_abs"] = max(abs(x-y) for old, new in zip(prior["rollouts"],policy["recomputed_old_log_probs"])
                                               for x,y in zip(old["old_log_probs"],new))
    result["max_abs"] = {key:max(abs(x-y) for a,b in zip(prior["rollouts"],policy["rollouts"]) for x,y in zip(a[key],b[key]))
                          for key in ("new_log_probs","ratios","policy_losses","moving_kls")}
    result["advantage_max_abs"] = max(abs(a["advantage"]-b["advantage"]) for a,b in zip(prior["rollouts"],policy["rollouts"]))
try:
    result["post_idle"] = bool(verify_idle(identities))
except (AssertionError, subprocess.SubprocessError):
    result["post_idle"] = False
write("outcome.json", result)
print(json.dumps(result), flush=True)
