"""Compare both A100 SAGE-GRPO update-state snapshots without moving tensor bytes."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
A100 = Path("C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline")
intent = json.loads((ROOT / "hunyuan-sage-A100-determinism-launch-intent.json").read_bytes())
remote = r"""
from pathlib import Path
import hashlib,io,json,math
import torch

workspace=Path("/home/whj/nvidia-baseline/alignment-assets/hygon-hunyuan-sage-determinism-A100-v1")
out=json.loads((workspace/"outcome.json").read_bytes())
assert out["request_id"]==REQUEST and out["claim"]==CLAIM and out["post_idle"]
for label in ("first","second"):
    row=out[label]
    if row is None or "log_sha256" not in row:
        continue
    log=workspace/f"{label}.log"
    with log.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest()==row["log_sha256"]
    if row["startup"] is not None:
        with log.open("rb") as stream:
            assert json.loads(stream.readline(1024))==row["startup"]
        assert row["startup"]["strict_determinism"] and row["startup"]["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
if not all(out[label] and out[label]["passed"] for label in ("first","second")):
    import re
    failures={}
    for label in ("first","second"):
        row=out[label]
        if row and not row["passed"] and "log_sha256" in row:
            log=workspace/f"{label}.log"
            with log.open("rb") as stream:
                stream.seek(max(0,log.stat().st_size-16384))
                tail=stream.read(16384).decode("utf8",errors="replace")
            errors=[line[:600] for line in tail.splitlines() if re.match(r"^[A-Za-z_][A-Za-z_0-9.]*(Error|Exception):",line)]
            failures[label]=errors[-1] if errors else None
    print(json.dumps(dict(outcome=out,complete_pair=False,failures=failures)))
    raise SystemExit(0)

def get(label,name):
    raw=(workspace/"results/hunyuanvideo-1.5/rl"/label/name).read_bytes()
    pin=out[label]["files"][name]
    assert len(raw)==pin["bytes"] and hashlib.sha256(raw).hexdigest()==pin["sha256"]
    return raw

states=[torch.load(io.BytesIO(get(label,"first-policy-update-state.pt")),map_location="cpu",weights_only=True)
        for label in ("first","second")]
policies=[json.loads(get(label,"first-policy-update.json")) for label in ("first","second")]
for state,policy,label in zip(states,policies,("first","second")):
    assert state["schema"]=="hunyuanvideo-sage-policy-update-v1" and state["global_step"]==1
    assert len(state["transition_losses"])==16 and all(map(math.isfinite,state["transition_losses"]))
    assert policy["update_state"]["sha256"]==out[label]["files"]["first-policy-update-state.pt"]["sha256"]
    assert set(state["parameters_before"])==set(state["parameters_after"])==set(state["gradients_before_clip"])==set(state["gradients_after_clip"])
assert policies[0]["trajectory"]["sha256"]==policies[1]["trajectory"]["sha256"]==out["trajectory_sha256"]
assert states[0]["optimizer_parameter_names"]==states[1]["optimizer_parameter_names"]
assert policies[0]["adapter_before"]==policies[1]["adapter_before"]
initial=states[0]["parameters_before"]
assert set(initial)==set(states[1]["parameters_before"])
assert all(torch.equal(tensor,states[1]["parameters_before"][name]) for name,tensor in initial.items())
def same_tree(a,b):
    if type(a) is not type(b):
        return False
    if isinstance(a,torch.Tensor):
        return a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b
assert same_tree(states[0]["optimizer_before"],states[1]["optimizer_before"])
fields=("actor_output_sha256","actor_output_gradient_sha256","anchor_mean_sha256",
        "accumulated_lora_gradients_sha256")
expected=[(i,j) for i in range(4) for j in range(4)]
boundaries=[state["transition_boundaries"] for state in states]
for rows in boundaries:
    assert [(x["rollout"],x["transition"]) for x in rows]==expected
    assert all(len(row[key])==64 and not set(row[key])-set("0123456789abcdef") for row in rows for key in fields)
mismatches={key:[list(step) for step,x,y in zip(expected,*boundaries) if x[key]!=y[key]] for key in fields}
blocks=[state["first_actor_block_boundaries"] for state in states]
assert tuple(blocks[0])==tuple(blocks[1])
families={name.split(".")[0] for name in blocks[0]}
assert families and families<={"double_blocks","single_blocks"}
framework_path=workspace/"framework/generation/src/gencuda/backends/hunyuanvideo_sage_grpo.py"
assert hashlib.sha256(framework_path.read_bytes()).hexdigest()==FRAMEWORK_SOURCE_SHA
launch=json.loads((workspace/"launch.json").read_bytes())
config_path=Path(launch["weights"])/"transformer/480p_t2v/config.json"
config_raw=config_path.read_bytes()
config=json.loads(config_raw)
expected_counts={family:config.get(f"mm_{family}_depth",default)
                 for family,default in (("double_blocks",20),("single_blocks",40))}
assert all(type(count) is int and count>=0 for count in expected_counts.values())
for family in ("double_blocks","single_blocks"):
    names=[k for k in blocks[0] if k.startswith(family+".")]
    assert names==[f"{family}.{i}" for i in range(expected_counts[family])], (family, len(names), expected_counts[family])
block_mismatches=[]
for name,left in blocks[0].items():
    right=blocks[1][name]
    assert len(left)==len(right)==(2 if name.startswith("double_blocks.") else 1)
    for output,(a,b) in enumerate(zip(left,right,strict=True)):
        assert set(a)==set(b)
        assert a["requires_grad"]==b["requires_grad"]
        for key in ("output_sha256","output_gradient_sha256"):
            if key in a:
                assert all(len(row[key])==64 and not set(row[key])-set("0123456789abcdef") for row in (a,b))
                if a[key]!=b[key]:
                    block_mismatches.append([name,output,key])
gradient_counts=[sum("output_gradient_sha256" in row for rows in trace.values() for row in rows) for trace in blocks]
assert gradient_counts[0]==gradient_counts[1] and gradient_counts[0]>=len(blocks[0])
prefix="double_blocks.50."
internal=[s["first_actor_internal_boundaries"] for s in states]
expected_names={prefix+f"{stream}_{op}{suffix}"
                for stream in ("img","txt")
                for op in ("attn_q","attn_k","attn_v","attn_proj","mlp")
                for suffix in (("",".input") if op in ("attn_proj","mlp") else ("",))}
assert set(internal[0])==set(internal[1])==expected_names
internal_result={}
for name in sorted(expected_names):
    a,=internal[0][name]
    b,=internal[1][name]
    assert set(a)==set(b)
    assert a["requires_grad"]==b["requires_grad"]
    result=dict(requires_grad=a["requires_grad"])
    for key in ("output_sha256","output_gradient_sha256"):
        if key not in a:continue
        assert all(len(r[key])==64 and not set(r[key])-set("0123456789abcdef") for r in (a,b))
        result[key+"_equal"]=a[key]==b[key]
    internal_result[name]=result
gradient_key="output_gradient_sha256_equal"
expected_image_path={prefix+name for name in ("img_attn_q","img_attn_k","img_attn_v",
    "img_attn_proj","img_attn_proj.input","img_mlp","img_mlp.input")}
assert all(gradient_key in internal_result[name] for name in expected_image_path)
missing_gradients=[name for name,row in internal_result.items() if gradient_key not in row]
comparison={}
for field in ("gradients_before_clip","gradients_after_clip","parameters_after","optimizer_after"):
    comparison[field]=same_tree(states[0][field],states[1][field])
assert not torch.cuda.is_initialized()
print(json.dumps(dict(outcome=out,complete_pair=True,exact_tensor_comparison=comparison,transition_rows=len(expected),mismatches=mismatches,
    trajectory_sha256=policies[0]["trajectory"]["sha256"],
    initial_adapter_equal=True, initial_optimizer_equal=True,
    configured_block_counts=expected_counts, transformer_config_sha256=hashlib.sha256(config_raw).hexdigest(),
    block_names=list(blocks[0]),gradient_counts=gradient_counts,block_mismatches=block_mismatches,
    internal_boundaries=internal_result,internal_boundaries_without_gradient=missing_gradients,
    scalar_losses_equal=states[0]["transition_losses"]==states[1]["transition_losses"],
    policy_rollouts_equal=policies[0]["rollouts"]==policies[1]["rollouts"],
    adapter_after_equal=policies[0]["adapter_after"]==policies[1]["adapter_after"])))

"""
cmd=["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12","-F",str(A100/"ssh_config"),"a100-26",
     "timeout 90 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /home/whj/nvidia-baseline/env-vlm/bin/python -B -u -"]
framework=ROOT/"hygon-training-integration/generation/src/gencuda/backends/hunyuanvideo_sage_grpo.py"
framework_sha=hashlib.sha256(framework.read_bytes()).hexdigest()
call=subprocess.run(cmd,input=(f"FRAMEWORK_SOURCE_SHA={framework_sha!r}\nREQUEST={intent['request_id']!r}\nCLAIM={intent['claim']!r}\n"+remote).encode(),capture_output=True,timeout=105)
if call.returncode:
    print(call.stderr.decode(errors="replace").splitlines()[-4:])
    raise SystemExit(call.returncode)
value=json.loads(call.stdout)
with (ROOT/"hunyuan-sage-A100-determinism-independent.json").open("x",encoding="utf8") as out:
    json.dump(value,out,indent=2)
print(json.dumps(value))
