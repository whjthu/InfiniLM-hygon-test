"""Compare both A100 SAGE-GRPO update-state snapshots without moving tensor bytes."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
A100 = Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
request = json.loads((ROOT / "hunyuan-sage-A100-tensor-pair-launch-intent.json").read_bytes())["request_id"]
remote = r"""
from pathlib import Path
import hashlib,io,json,math
import torch

workspace=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sage-tensor-pair-A100-v1")
intent=json.loads((workspace/"intent.json").read_bytes())
assert intent["request_id"]==REQUEST
assert not (workspace/"replay.log").exists() and not (workspace/"replay-outcome.json").exists()
out=json.loads((workspace/"capture-outcome.json").read_bytes())
assert out["passed"] and out["child_rc"]==0
leaf=workspace/"results/hunyuanvideo-1.5/rl/capture"
def read(name):
    raw=(leaf/name).read_bytes()
    pin=out["files"][name]
    assert len(raw)==pin["bytes"] and hashlib.sha256(raw).hexdigest()==pin["sha256"]
    return raw
manifest=json.loads(read("manifest.json"))
policy=json.loads(read("first-policy-update.json"))
assert manifest["status"]=="completed" and manifest["metrics"]["policy_updates"]==1
state=torch.load(io.BytesIO(read("first-policy-update-state.pt")),map_location="cpu",weights_only=True)
assert state["schema"]=="hunyuanvideo-sage-policy-update-v1" and state["global_step"]==1
assert len(state["transition_losses"])==16 and all(map(math.isfinite,state["transition_losses"]))
assert policy["update_state"]["sha256"]==out["files"]["first-policy-update-state.pt"]["sha256"]
named=state["parameters_before"]
assert set(named)==set(state["parameters_after"])==set(state["gradients_before_clip"])==set(state["gradients_after_clip"])
assert set(named)=={name for group in state["optimizer_parameter_names"] for name in group}
changed=0
for name,param in named.items():
    fields=("parameters_after","gradients_before_clip","gradients_after_clip")
    values=[state[key][name] for key in fields]
    assert all(isinstance(v,torch.Tensor) and v.shape==param.shape and torch.isfinite(v).all() for v in [param,*values])
    changed+=int(not torch.equal(param,values[0]))
assert changed and not torch.cuda.is_initialized()
trajectory=leaf/"trajectories/step-0001.pt"
with trajectory.open("rb") as stream: digest=hashlib.file_digest(stream,"sha256").hexdigest()
assert digest==policy["trajectory"]["sha256"]
print(json.dumps(dict(request_id=REQUEST,capture=out,replay_started=False,trainable_tensors=len(named),
                      trainable_elements=sum(v.numel() for v in named.values()),changed_parameter_tensors=changed,
                      optimizer_entries_before=len(state["optimizer_before"]["state"]),
                      optimizer_entries_after=len(state["optimizer_after"]["state"]),
                      trajectory=dict(bytes=trajectory.stat().st_size,sha256=digest))))

"""
cmd=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(A100/"ssh_config"),"a100-26",
     "timeout 90 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/whj/nvidia-baseline/env-vlm/bin/python -B -u -"]
call=subprocess.run(cmd,input=(f"REQUEST={request!r}\n"+remote).encode(),capture_output=True,timeout=105)
if call.returncode:
    print(call.stderr.decode(errors="replace").splitlines()[-4:])
    raise SystemExit(call.returncode)
value=json.loads(call.stdout)
with (ROOT/"hunyuan-sage-A100-tensor-capture-independent.json").open("x",encoding="utf8") as out:
    json.dump(value,out,indent=2)
print(json.dumps(value))
