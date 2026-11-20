"""One bounded four-arm Socket/SHCA comparison on the same two-node lease."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from admission import parent_gate, step_gate
from qualified_sampler import node_map

ROOT = Path(__file__).resolve().parent
REF = ROOT.parents[1]
PARENT = "1103606"
NODES = ("f13r1n11", "f13r1n12")
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PROVIDER = REF / "build-deps/rdma-host-stack"
OPT = REF / "candidates/public-shca-layers-r2228/layer3-selected/opt"
SHELL = (
    'set -euo pipefail; source /opt/dtk/env.sh; '
    'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET; '
    'if [ "$4" = "shca" ]; then '
    'export LD_LIBRARY_PATH=/candidate-opt/rccl-shca-net/lib:/candidate-opt/ucx-shca/lib:"$2/verbs-aliases":/rdma-provider-stack:$LD_LIBRARY_PATH '
    'RDMAV_DRIVERS=/rdma-provider-stack/libshca NCCL_NET=IBext_v8 NCCL_NET_PLUGIN=shca NCCL_IB_DISABLE=0 '
    'UCX_MODULE_DIR=/candidate-opt/ucx-shca/lib/ucx UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self; '
    'else export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1; fi; '
    'exec python3 -B -u "$1" "$2" "$3" "$4" "$5"'
)
ACCT = "JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"
POST = (
    'import json, os, socket; from node_sample import fresh_inventory; from pathlib import Path; '
    'print(json.dumps(dict(node=socket.gethostname().split(".")[0], '
    'step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"], '
    'holders=len(list(Path("/sys/class/kfd/kfd/proc").iterdir())), pci=sorted(fresh_inventory()))))'
)


def save(name, data):
    with (ROOT / name).open("x") as file:
        json.dump(data, file, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())


def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=9).splitlines()


def rows(label):
    return [r.split("|") for r in query("sacct", "-nP", "-j", PARENT, "--format=" + ACCT)
            if r.split("|")[1:2] == [label]]


def wait_row(label):
    for _ in range(10):
        seen = rows(label)
        if len(seen) == 1 and seen[0][2] not in ("PENDING", "RUNNING", "COMPLETING"):
            return seen[0]
        time.sleep(0.7)
    raise TimeoutError("ORIGINAL_ARM_ACCOUNTING")


def verify_providers():
    files = json.loads((ROOT / "PROVIDERS.json").read_bytes())["files"]
    for path, pin in files.items():
        if path.startswith("/rdma-provider-stack/"):
            candidate = PROVIDER / path.split("/", 2)[2]
        elif path.startswith("/candidate-opt/"):
            candidate = OPT / path.split("/", 2)[2]
        else:
            raise ValueError("UNEXPECTED_PROVIDER_SCOPE")
        raw = candidate.read_bytes()
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["SHA256"]:
            raise ValueError("EXACT_PROVIDER_PIN")
    aliases = ROOT / "verbs-aliases"
    aliases.mkdir()
    for name in ("libibverbs.so", "libibverbs.so.1"):
        (aliases / name).symlink_to("/rdma-provider-stack/libibverbs.so.1")


def clean_before(label, epoch):
    program = "import sys;sys.path.insert(0," + repr(str(ROOT)) + ");" + POST
    with (ROOT / (label + ".stdout")).open("x") as out, (ROOT / (label + ".stderr")).open("x") as err:
        result = subprocess.run(slurm(label, False) +
            ["/usr/local/python3.12/bin/python3", "-B", "-c", program],
            stdout=out, stderr=err, timeout=min(30, epoch+650-time.time()))
    ledger = wait_row(label)
    probes = [json.loads(x) for x in (ROOT / (label + ".stdout")).read_text().splitlines() if x.startswith("{")]
    if (result.returncode or ledger[2:4] != ["COMPLETED", "0:0"] or "gres/dcu" in ledger[-1] or
        {p["node"] for p in probes} != set(NODES) or any(p["holders"] or len(set(p["pci"])) != 8 for p in probes)):
        raise ValueError("INDEPENDENT_KFD_GATE")
    return dict(accounting=ledger, nodes=probes)


def slurm(label, gpu):
    if rows(label):
        raise ValueError("EXACT_LABEL_ALREADY_EXISTS")
    return ["srun", "--jobid="+PARENT, "--job-name="+label, "--exact", "--nodes=2", "--ntasks=2",
            "--ntasks-per-node=1", "--nodelist="+",".join(NODES), "--immediate=5",
            "--cpus-per-task="+("24" if gpu else "1"), "--mem="+("72G" if gpu else "2G"),
            "--gres="+("dcu:8" if gpu else "none"), "--time="+("2" if gpu else "1")]


def one_arm(label, index, mode, epoch):
    directory = ROOT / f"arm-{index}"
    directory.mkdir(mode=0o700)
    save("arm-"+str(index)+"-intent.json",dict(label=label,node_pair=list(NODES),mode=mode,index=index,epoch=epoch))
    command = slurm(label, True) + [
        APPTAINER, "exec", "--rocm", "--no-home", "--bind", str(ROOT)+":"+str(ROOT),
        "--bind", str(PROVIDER)+":/rdma-provider-stack:ro",
        "--bind", str(OPT)+":/candidate-opt:ro",
        "--bind", "/opt/hyhal:/opt/hyhal:ro", str(REF/"containers/dtk26-sandbox-r1"),
        "/bin/bash", "--noprofile", "--norc", "-c", SHELL,
        "transport-arm", str(ROOT/"worker.py"), str(ROOT), str(index), mode,
        str(min(epoch+540, time.time()+103)),
    ]
    with (directory/"step.stdout").open("x") as out,(directory/"step.stderr").open("x") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            child.wait(timeout=max(1,min(115,epoch+575-time.time())))
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=4)
    observed = wait_row(label)
    if child.returncode or observed[2:4] != ["COMPLETED", "0:0"] or "gres/dcu=16" not in observed[-1].split(","):
        raise ValueError("ARM_FAILED_ACTUAL_ACCOUNTING")
    return observed


def main(epoch, request):
    claim = json.loads((ROOT/"CLAIM.json").read_bytes())
    if request != claim["request"] or not 0 <= time.time()-epoch < 38:
        raise ValueError("OWNED_HOST_EPOCH")
    nodes = sorted(n for n,(j,_) in node_map(json.loads((ROOT/"ledger.json").read_bytes())).items() if j==PARENT)
    expand = lambda v: query("scontrol","show","hostnames",v)
    row, = query("scontrol","show","job",PARENT,"-o")
    parent_gate(row,PARENT,os.getuid(),nodes,expand)
    if not set(NODES) <= set(nodes):
        raise ValueError("PAIR_EXACT_MEMBERSHIP")
    step_gate(query("scontrol","show","step",PARENT,"-o"),PARENT,(),set(),expand)
    verify_providers()
    prefix="rccl-transport-4x4-"+request
    before=clean_before(prefix+"-pre",epoch)
    terminal={}
    outcome=None
    try:
        for index,mode in enumerate(("socket","shca","shca","socket")):
            if time.time()>=epoch+430:
                raise TimeoutError("FIXED_ARM_START_DEADLINE")
            terminal[str(index)]=one_arm(prefix+"-"+str(index)+"-"+mode,index,mode,epoch)
    except (ValueError, OSError, subprocess.SubprocessError) as failure:
        outcome=dict(type=type(failure).__name__,code=str(failure) if isinstance(failure,ValueError) else None)
    finally:
        after=None
        active = [line for line in query("scontrol","show","step",PARENT,"-o") if prefix in line]
        if not active:
            after=clean_before(prefix+"-post",epoch)
            if {p["node"]:p["pci"] for p in before["nodes"]} != {p["node"]:p["pci"] for p in after["nodes"]}:
                raise ValueError("SAME_PHYSICAL_DEVICES")
        save("dispatch-terminal.json",dict(request=request,arms=terminal,before=before,after=after,failure=outcome))
    return int(len(terminal)!=4 or outcome is not None or after is None)


if __name__ == "__main__":
    raise SystemExit(main(float(sys.argv[1]),sys.argv[2]))
