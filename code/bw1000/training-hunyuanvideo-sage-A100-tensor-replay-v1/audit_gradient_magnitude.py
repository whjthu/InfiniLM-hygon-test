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
capture,replayed=states
def magnitude(ref,other):
    assert ref.shape==other.shape and ref.dtype==other.dtype
    a,b=ref.double(),other.double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta=a-b
    return dict(n=a.numel(),changed=int((a!=b).sum()),max_abs=delta.abs().max().item(),
                delta_sq=delta.square().sum().item(),reference_sq=a.square().sum().item(),
                sign_flips=int(((a*b)<0).sum()),zero_crossings=int(((a==0)!=(b==0)).sum()))

def combine(rows):
    ref=sum(v["reference_sq"] for v in rows)
    return dict(tensors=len(rows),different_tensors=sum(v["changed"]>0 for v in rows),
                elements=sum(v["n"] for v in rows),different_elements=sum(v["changed"] for v in rows),
                max_abs=max(v["max_abs"] for v in rows),relative_l2=(sum(v["delta_sq"] for v in rows)/ref)**.5 if ref else None,
                sign_flips=sum(v["sign_flips"] for v in rows),zero_crossings=sum(v["zero_crossings"] for v in rows))

grouped={}
for name,before in capture["parameters_before"].items():
    projection,adapter=next((layer,arm) for layer in ("img_attn_q","img_attn_v")
                            for arm in ("lora_A","lora_B") if layer in name and arm in name)
    key=f"{projection}.{adapter}"
    ga,gb=(s["gradients_before_clip"][name] for s in states)
    assert all(torch.equal(s["gradients_before_clip"][name],s["gradients_after_clip"][name]) for s in states)
    assert torch.equal(before,replayed["parameters_before"][name])
    da,db=(s["parameters_after"][name].double()-before.double() for s in states)
    group=grouped.setdefault(key,dict(gradients=[],parameter_updates=[]))
    group["gradients"].append(magnitude(ga,gb))
    group["parameter_updates"].append(magnitude(da,db))
assert sum(len(v["gradients"]) for v in grouped.values())==216
assert len(grouped)==4 and not torch.cuda.is_initialized()
print(json.dumps(dict(request_id=REQUEST,grouped={g:{k:combine(v) for k,v in rows.items()}
                      for g,rows in grouped.items()},
                      clipping_changed_values=False,absolute_error_basis="CPU float64 comparison",
                      trace_reexecuted=False)))

"""
cmd=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(A100/"ssh_config"),"a100-26",
     "timeout 90 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/whj/nvidia-baseline/env-vlm/bin/python -B -u -"]
call=subprocess.run(cmd,input=(f"REQUEST={request!r}\n"+remote).encode(),capture_output=True,timeout=105)
if call.returncode:
    print(call.stderr.decode(errors="replace").splitlines()[-4:])
    raise SystemExit(call.returncode)
value=json.loads(call.stdout)
with (ROOT/"hunyuan-sage-A100-gradient-magnitude.json").open("x",encoding="utf8") as out:
    json.dump(value,out,indent=2)
print(json.dumps(value))
