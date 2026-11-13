from pathlib import Path
import json, subprocess
root=Path(__file__).resolve().parent
a=Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
request=json.loads((root/"hunyuan-sage-A100-replay-launch-intent.json").read_bytes())["request_id"]
remote=r"""
from pathlib import Path
import hashlib,json,math
base=Path("/home/whj/nvidia-baseline/alignment-assets")
original=base/"hygon-hunyuan-sage-first-policy-A100-v1"
replay=base/"hygon-hunyuan-sage-fixed-replay-A100-v1"
leaf=Path("results/hunyuanvideo-1.5/rl/first-policy")
def pinned(path,sha):
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sha
    return json.loads(raw)
out=json.loads((replay/"outcome.json").read_bytes())
assert out["request_id"]==REQUEST and out["child_rc"]==0 and out["post_idle"]
manifest=pinned(replay/leaf/"manifest.json",out["files"]["manifest.json"]["sha256"])
assert manifest["status"]=="completed" and manifest["metrics"]["policy_updates"]==1
assert manifest["metrics"]["diagnostic_fixed_trajectory_replay"] and not manifest["metrics"]["performance_accepted"]
prior=pinned(original/leaf/"first-policy-update.json","c1f8e0283e9ce7ad8f1dd64b7d6510d616a8fa8418e96d6f86f57d1f0b36f4d3")
current=pinned(replay/leaf/"first-policy-update.json",out["files"]["first-policy-update.json"]["sha256"])
assert prior["adapter_before"]==current["adapter_before"]
assert prior["adapter_before"]!=prior["adapter_after"] and current["adapter_before"]!=current["adapter_after"]
assert prior["trajectory"]["sha256"]==current["trajectory"]["sha256"]=="55a879cafa7880840ac0aead8dec4fd4c55f3732e041dcfbfbad0cba790dccff"
assert len(prior["rollouts"])==len(current["rollouts"])==4
assert prior["rollouts"]==current["rollouts"]
assert prior["reward"]==current["reward"]==current["reference_reward"]
assert [row["old_log_probs"] for row in prior["rollouts"]]==current["recomputed_old_log_probs"]
norms=[d["grad_norm_before_clip"] for d in (prior,current)]
assert all(math.isfinite(n) and n>0 for n in norms)
print(json.dumps(dict(outcome=out,policy_metrics_exact=True,initial_adapter_same=True,
    original_grad_norm_before_clip=norms[0],replay_grad_norm_before_clip=norms[1],
    grad_norm_abs_difference=abs(norms[0]-norms[1]),adapter_after_same=prior["adapter_after"]==current["adapter_after"],
    full_gradient_values_captured=False,updated_tensor_values_captured=False,original_launcher_passed=False)))
"""
cmd=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(a/"ssh_config"),"a100-26","timeout 25 python3 -B -u -"]
call=subprocess.run(cmd,input=(f"REQUEST={request!r}\n"+remote).encode(),capture_output=True,timeout=35)
if call.returncode: print(call.stderr.decode(errors="replace").splitlines()[-2:]);raise SystemExit(call.returncode)
value=json.loads(call.stdout)
with (root/"hunyuan-sage-A100-replay-independent.json").open("x",encoding="utf8") as out:json.dump(value,out,indent=2)
print(json.dumps(value))
