"""Compare both A100 SAGE-GRPO update-state snapshots without moving tensor bytes."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
A100 = Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
request = json.loads((ROOT / "hunyuan-sage-A100-boundaries-launch-intent.json").read_bytes())["request_id"]
remote = r"""
from pathlib import Path
import hashlib,io,json,math
import torch

workspace=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sage-boundaries-A100-v1")
out=json.loads((workspace/"outcome.json").read_bytes())
assert out["request_id"]==REQUEST and out["post_idle"]
assert all(out[label]["passed"] for label in ("capture","replay"))

def get(label,name):
    raw=(workspace/"results/hunyuanvideo-1.5/rl"/label/name).read_bytes()
    pin=out[label]["files"][name]
    assert len(raw)==pin["bytes"] and hashlib.sha256(raw).hexdigest()==pin["sha256"]
    return raw

states=[torch.load(io.BytesIO(get(label,"first-policy-update-state.pt")),map_location="cpu",weights_only=True)
        for label in ("capture","replay")]
policies=[json.loads(get(label,"first-policy-update.json")) for label in ("capture","replay")]
for state,policy,label in zip(states,policies,("capture","replay")):
    assert state["schema"]=="hunyuanvideo-sage-policy-update-v1" and state["global_step"]==1
    assert len(state["transition_losses"])==16 and all(map(math.isfinite,state["transition_losses"]))
    assert policy["update_state"]["sha256"]==out[label]["files"]["first-policy-update-state.pt"]["sha256"]
    assert set(state["parameters_before"])==set(state["parameters_after"])==set(state["gradients_before_clip"])==set(state["gradients_after_clip"])
assert policies[0]["trajectory"]["sha256"]==policies[1]["trajectory"]["sha256"]
assert states[0]["optimizer_parameter_names"]==states[1]["optimizer_parameter_names"]
fields=("actor_output_sha256","actor_output_gradient_sha256","anchor_mean_sha256",
        "accumulated_lora_gradients_sha256")
expected=[(i,j) for i in range(4) for j in range(4)]
boundaries=[state["transition_boundaries"] for state in states]
for rows in boundaries:
    assert [(x["rollout"],x["transition"]) for x in rows]==expected
    assert all(len(row[key])==64 and not set(row[key])-set("0123456789abcdef") for row in rows for key in fields)
mismatches={key:[list(step) for step,x,y in zip(expected,*boundaries) if x[key]!=y[key]] for key in fields}
assert not torch.cuda.is_initialized()
print(json.dumps(dict(outcome=out,transition_rows=len(expected),mismatches=mismatches,
    trajectory_sha256=policies[0]["trajectory"]["sha256"],
    scalar_losses_equal=states[0]["transition_losses"]==states[1]["transition_losses"],
    policy_rollouts_equal=policies[0]["rollouts"]==policies[1]["rollouts"],
    adapter_after_equal=policies[0]["adapter_after"]==policies[1]["adapter_after"])))

"""
cmd=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(A100/"ssh_config"),"a100-26",
     "timeout 90 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/whj/nvidia-baseline/env-vlm/bin/python -B -u -"]
call=subprocess.run(cmd,input=(f"REQUEST={request!r}\n"+remote).encode(),capture_output=True,timeout=105)
if call.returncode:
    print(call.stderr.decode(errors="replace").splitlines()[-4:])
    raise SystemExit(call.returncode)
value=json.loads(call.stdout)
with (ROOT/"hunyuan-sage-A100-boundaries-independent.json").open("x",encoding="utf8") as out:
    json.dump(value,out,indent=2)
print(json.dumps(value))
