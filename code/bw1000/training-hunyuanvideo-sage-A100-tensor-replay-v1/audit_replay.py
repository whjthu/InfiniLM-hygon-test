"""Compare both A100 SAGE-GRPO update-state snapshots without moving tensor bytes."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
A100 = Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
request = json.loads((ROOT / "hunyuan-sage-A100-tensor-replay-launch-intent.json").read_bytes())["request_id"]
remote = r"""
from pathlib import Path
import hashlib,io,json,math
import torch

workspace=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sage-tensor-pair-A100-v1")
replay=workspace.with_name("hygon-hunyuan-sage-tensor-replay-A100-v1")
receipt=json.loads((replay/"outcome.json").read_bytes())
assert receipt["request_id"]==REQUEST and receipt["passed"] and receipt["child_rc"]==0
out={"capture":json.loads((workspace/"capture-outcome.json").read_bytes()),"replay":receipt}
assert out["capture"]["passed"]

def get(label,name):
    if label=="capture":
        base=workspace/"results/hunyuanvideo-1.5/rl/capture"
    else:
        base=replay/"results/hunyuanvideo-1.5/rl/first-policy"
    raw=(base/name).read_bytes()
    pin=out[label]["files"][name]
    assert len(raw)==pin["bytes"] and hashlib.sha256(raw).hexdigest()==pin["sha256"]
    return raw

for label in ("capture","replay"):
    doc=json.loads(get(label,"manifest.json"))
    assert doc["status"]=="completed" and doc["metrics"]["policy_updates"]==1
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
def summarize(left,right):
    info=dict(tensors=0,different_tensors=0,elements=0,different_elements=0,max_abs=0.0,scalar_differences=0,structural_differences=0)
    def visit(a,b):
        if type(a)!=type(b):
            info["structural_differences"]+=1
        elif isinstance(a,torch.Tensor):
            assert a.shape==b.shape and a.dtype==b.dtype and a.device.type==b.device.type=="cpu"
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            diff=(a!=b).sum().item()
            info["tensors"]+=1
            info["different_tensors"]+=bool(diff)
            info["elements"]+=a.numel()
            info["different_elements"]+=diff
            if diff:
                info["max_abs"]=max(info["max_abs"],(a.to(torch.float64)-b.to(torch.float64)).abs().max().item())
        elif isinstance(a,dict):
            info["structural_differences"]+=len(a.keys()^b.keys())
            for key in sorted(a.keys()&b.keys()):visit(a[key],b[key])
        elif isinstance(a,(tuple,list)):
            info["structural_differences"]+=abs(len(a)-len(b))
            for x,y in zip(a,b):visit(x,y)
        else:
            info["scalar_differences"]+=a!=b
    visit(left,right)
    return info
names=("parameters_before","optimizer_before","transition_losses","gradients_before_clip",
       "gradients_after_clip","parameters_after","optimizer_after")
comparison={key:summarize(states[0][key],states[1][key]) for key in names}
assert not torch.cuda.is_initialized()
print(json.dumps(dict(outcome=out,comparison=comparison,policy_rollouts_equal=policies[0]["rollouts"]==policies[1]["rollouts"],
    rewards_equal=policies[0]["reward"]==policies[1]["reward"],
    adapter_after_equal=policies[0]["adapter_after"]==policies[1]["adapter_after"])))
"""
cmd=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(A100/"ssh_config"),"a100-26",
     "timeout 90 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/whj/nvidia-baseline/env-vlm/bin/python -B -u -"]
call=subprocess.run(cmd,input=(f"REQUEST={request!r}\n"+remote).encode(),capture_output=True,timeout=105)
if call.returncode:
    print(call.stderr.decode(errors="replace").splitlines()[-4:])
    raise SystemExit(call.returncode)
value=json.loads(call.stdout)
with (ROOT/"hunyuan-sage-A100-tensor-replay-independent.json").open("x",encoding="utf8") as out:
    json.dump(value,out,indent=2)
print(json.dumps(value))
