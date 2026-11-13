from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parent
baseline = Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
source = (root / "hygon-training-integration/generation/src/gencuda/hunyuanvideo_policy_compare.py").read_bytes()
remote = '''
from pathlib import Path
import hashlib, json, sys
import torch

base = Path("/home/whj/nvidia-baseline/alignment-assets")
strict = base / "hygon-hunyuan-sage-determinism-A100-v1"
sys.path.insert(0, str(strict / "framework/generation/src"))
module = {"__name__": "gencuda.policy_audit_control", "__package__": "gencuda"}
exec(compile(SOURCE, "policy_compare", "exec"), module)
compare = module["compare_updates"]
result = {}
for kind, folder, request in (
    ("strict", strict, "f27e96ca4c6b4d7db618a6b57916b46b"),
    ("ordinary", base / "hygon-hunyuan-sage-block-internal-A100-v1", "7411601a93354c5a990635f7668cc92c"),
):
    outcome = json.loads((folder / "outcome.json").read_bytes())
    assert outcome["request_id"] == request and outcome["post_idle"]
    assert all(outcome[arm]["passed"] for arm in ("first", "second"))
    roots = [folder/"results/hunyuanvideo-1.5/rl"/arm for arm in ("first", "second")]
    pins = [outcome[arm]["files"]["first-policy-update.json"]["sha256"] for arm in ("first", "second")]
    result[kind] = compare(*roots, *pins)
    assert result[kind]["trainable_tensors"] == 216
    assert all(len(rows) == 4 for rows in result[kind]["grouped_tensor_metrics"].values())
    counts = [sum(group["different_tensors"] for group in rows.values())
              for rows in result[kind]["grouped_tensor_metrics"].values()]
    assert all(count == 0 for count in counts) if kind == "strict" else all(count > 0 for count in counts)
    assert result[kind]["optimizer_after"]["tree_equal"] == (kind == "strict")
try:
    compare(*roots, "0"*64, pins[1])
except ValueError as e:
    assert "SHA256 differs" in str(e)
else:
    raise AssertionError("accepted a bad policy digest")
assert not torch.cuda.is_initialized()
print(json.dumps(dict(results=result, wrong_policy_sha_rejected=True,
                      comparator_sha256=hashlib.sha256(SOURCE).hexdigest(), cuda_initialized=False), allow_nan=False))
'''
cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-F", str(baseline/"ssh_config"),
       "a100-26", "timeout 90 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/whj/nvidia-baseline/env-vlm/bin/python -B -u -"]
result = subprocess.run(cmd, input=(f"SOURCE={source!r}\n"+remote).encode(), capture_output=True, timeout=105)
if result.returncode:
    raise SystemExit(result.stderr.decode(errors="replace").splitlines()[-4:])
receipt = json.loads(result.stdout)
with (root/"hunyuan-sage-policy-comparator-controls-final.json").open("x", encoding="utf8", newline="\n") as out:
    json.dump(receipt, out, indent=2)
print(json.dumps({name: {"different_tensor_counts": {
    field: sum(row["different_tensors"] for row in groups.values())
    for field, groups in arm["grouped_tensor_metrics"].items()},
    "optimizer_equal": arm["optimizer_after"]["tree_equal"]} for name, arm in receipt["results"].items()}))
