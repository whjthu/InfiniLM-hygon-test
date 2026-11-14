import hashlib,json,os,pathlib,socket,sys
sys.path.insert(0,"/paired-worker")
import shared,native_arm
from verify_transport import INFINICCL_SHA256, RCCL_SHA256, PLUGIN_SHA256
def check(x):
    if not x: raise ValueError("preflight boundary")
root=pathlib.Path(sys.argv[1])
rank=int(os.environ["SLURM_PROCID"])
check(shared.step_id().startswith("1103606."))
check(("f13r1n08","f13r1n09")[rank]==socket.gethostname().split(".")[0])
check(len({socket.gethostbyname(n) for n in ("f13r1n08","f13r1n09")})==2)
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
# The preflight runs on the spare I02 pair. Reuse only the snapshot inventory
# functions and leave the launcher's intended-node predicate unchanged.
from snapshot import drm_inventory, gpu_holders
pci=drm_inventory()
holders,kernel,denied=gpu_holders()
check(holders==0 and len(set(pci))==8)
provider_paths={
    "/refactor/install/hygon-glm-5f91ad7-r3672/lib/libinfiniccl.so":INFINICCL_SHA256,
    "/opt/dtk/lib/librccl.so.1.0":RCCL_SHA256,
    "/candidate-opt/rccl-shca-net/lib/librccl-net-shca.so.0.0.0":PLUGIN_SHA256,
}
for path,digest in provider_paths.items():
    check(shared.H(path)==digest)
dependencies=[
    f"/candidate-opt/ucx-shca/lib/{name}.so.0" for name in ("libucp","libucs","libucm","libuct")
]+["/rdma-provider-stack/libibverbs.so.1"]
dependency_sha={path:shared.H(path) for path in dependencies}
check(pathlib.Path("/candidate-opt/ucx-shca/lib/ucx").is_dir())
check(any(pathlib.Path("/rdma-provider-stack").glob("libshca*")))
report=dict(step=shared.step_id(),node=socket.gethostname().split(".")[0],rank=rank,pci_count=len(pci),arms=2,requests=33,
            host_manifest_SHA256=shared.H(shared.D/"HOST_MANIFEST.json"),worker_manifest_SHA256=shared.H(shared.D/"SOURCE_MANIFEST.json"),
            GPU_cards=0,Native_loaded=False,weight_bytes_read=0,
            kernel_kfd_holder_count=kernel, same_uid_fd_denied_count=denied,
            provider_SHA256=provider_paths, dependency_SHA256=dependency_sha,
            launch_target_pair_checked=False)
with (root/f"rank-{rank}.json").open("x") as f:
    json.dump(report,f,sort_keys=True)
