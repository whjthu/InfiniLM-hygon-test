import hashlib,json,os,pathlib,socket,sys
sys.path.insert(0,"/paired-worker")
import shared,native_arm
from snapshot import snapshot
def check(x):
    if not x: raise ValueError("preflight boundary")
root=pathlib.Path(sys.argv[1])
rank=int(os.environ["SLURM_PROCID"])
check(shared.step_id().startswith("1103606."))
check(shared.NODES[rank]==socket.gethostname().split(".")[0])
check([socket.gethostbyname(n) for n in shared.NODES]==shared.IPS)
check(all(os.environ[k]=="-1" for k in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES")))
for name in ("SOURCE_MANIFEST.json","HOST_MANIFEST.json"):
    p=shared.D/name
    pins=json.loads(p.read_bytes())["files_SHA256"]
    check(all(shared.H(shared.D/n)==h for n,h in pins.items()))
inp=json.loads((shared.D/"INPUT_BINDING.json").read_bytes())
pin=inp["private_plan"]
plan=shared.load_pinned(shared.R/pathlib.Path(pin["path"]).relative_to(shared.HOST),pin["SHA256"])
shared.verify_inputs(inp,json.loads((shared.D/"TRIAL.json").read_bytes()),plan)
model=pathlib.Path(inp["model_path"])
check(shared.H(model/"config.json")==inp["model_config_SHA256"])
check(shared.H(model/"model.safetensors.index.json")==inp["model_index_SHA256"])
check(shared.H(inp["tokenizer"]["path"])==inp["tokenizer"]["SHA256"])
for arm in native_arm.ARMS:
    native_arm.verified_library(arm)
with pathlib.Path("/local/preflight").open("x") as f:
    f.write("local writable")
state=snapshot(False)
check(state["kfd_holder_count"]==0)
report=dict(step=state["step"],node=state["node"],rank=rank,pci_count=len(state["pci"]),arms=2,requests=33,
            host_manifest_SHA256=shared.H(shared.D/"HOST_MANIFEST.json"),worker_manifest_SHA256=shared.H(shared.D/"SOURCE_MANIFEST.json"),
            GPU_cards=0,Native_loaded=False,weight_bytes_read=0,
            kernel_kfd_holder_count=state["kernel_kfd_holder_count"],
            same_uid_fd_denied_count=state["same_uid_fd_denied_count"],
            holder_scope=state["holder_scope"])
with (root/f"rank-{rank}.json").open("x") as f:
    json.dump(report,f,sort_keys=True)
