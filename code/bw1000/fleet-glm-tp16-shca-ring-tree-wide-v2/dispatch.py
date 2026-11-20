"""One bounded four-arm SHCA Ring/Tree comparison on the same two-node lease."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from admission import parent_gate, fields, need
from qualified_sampler import node_map

SOURCE = Path(__file__).resolve().parent
REF = SOURCE.parents[1]
APPTAINER = "/public/software/apps/apptainer/1.3.4/bin/apptainer"
PROVIDER = REF / "build-deps/rdma-host-stack"
OPT = REF / "candidates/public-shca-layers-r2228/layer3-selected/opt"
SHELL = (
    'set -euo pipefail; source /opt/dtk/env.sh; '
    'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV,TUNING; '
    'case "$4" in ring) export NCCL_ALGO=Ring ;; tree) export NCCL_ALGO=Tree ;; *) exit 28 ;; esac; '
    'export LD_LIBRARY_PATH=/candidate-opt/rccl-shca-net/lib:/candidate-opt/ucx-shca/lib:"$2/verbs-aliases":/rdma-provider-stack:$LD_LIBRARY_PATH '
    'RDMAV_DRIVERS=/rdma-provider-stack/libshca NCCL_NET=IBext_v8 NCCL_NET_PLUGIN=shca NCCL_IB_DISABLE=0 '
    'UCX_MODULE_DIR=/candidate-opt/ucx-shca/lib/ucx UCX_NET_DEVICES=shca_0:1,shca_1:1,shca_2:1,shca_3:1 UCX_TLS=rc,sm,self; '
    'exec python3 -B -u "$1" "$2" "$3" "$4" "$5"'
)
ACCT = "JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160"
POST = (
    'import json, os, socket; from pathlib import Path; '
    'print(json.dumps(dict(node=socket.gethostname().split(".")[0], '
    'step=os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"], '
    'holders=len(list(Path("/sys/class/kfd/kfd/proc").iterdir())), '
    'pci=sorted((v/"device").resolve().name.lower() for v in Path("/sys/class/drm").glob("renderD*")))))'
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
    files = json.loads((SOURCE / "PROVIDERS.json").read_bytes())["files"]
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
    program = POST
    with (ROOT / (label + ".stdout")).open("x") as out, (ROOT / (label + ".stderr")).open("x") as err:
        result = subprocess.run(slurm(label, False) +
            ["/usr/local/python3.12/bin/python3", "-B", "-c", program],
            stdout=out, stderr=err, timeout=min(55, epoch+1080-time.time()))
    ledger = wait_row(label)
    probes = [json.loads(x) for x in (ROOT / (label + ".stdout")).read_text().splitlines() if x.startswith("{")]
    if (result.returncode or ledger[2:4] != ["COMPLETED", "0:0"] or "gres/dcu" in ledger[-1] or
        len(probes) != len(NODES) or {p["node"] for p in probes} != set(NODES) or any(p["holders"] or len(set(p["pci"])) != 8 for p in probes)):
        raise ValueError("INDEPENDENT_KFD_GATE")
    return dict(accounting=ledger, nodes=probes)


def slurm(label, gpu):
    if rows(label):
        raise ValueError("EXACT_LABEL_ALREADY_EXISTS")
    return ["srun", "--jobid="+PARENT, "--job-name="+label, "--exact", "--nodes="+str(len(NODES)), "--ntasks="+str(len(NODES)),
            "--ntasks-per-node=1", "--nodelist="+",".join(NODES), "--immediate=5", "--kill-on-bad-exit=0",
            "--cpus-per-task="+("24" if gpu else "1"), "--mem="+("72G" if gpu else "2G"),
            "--gres="+("dcu:8" if gpu else "none"), "--time="+("3" if gpu else "1")]


def one_arm(label, index, mode, epoch):
    directory = ROOT / f"arm-{index}"
    directory.mkdir(mode=0o700)
    save("arm-"+str(index)+"-intent.json",dict(label=label,pairs=PAIRS,parent=PARENT,port_base=PORT_BASE,mode=mode,index=index,epoch=epoch))
    command = slurm(label, True) + [
        APPTAINER, "exec", "--rocm", "--no-home", "--bind", str(SOURCE)+":"+str(SOURCE),
        "--bind", str(PROVIDER)+":/rdma-provider-stack:ro",
        "--bind", str(OPT)+":/candidate-opt:ro",
        "--bind", "/opt/hyhal:/opt/hyhal:ro", str(REF/"containers/dtk26-sandbox-r1"),
        "/bin/bash", "--noprofile", "--norc", "-c", SHELL,
        "transport-arm", str(SOURCE/"worker.py"), str(ROOT), str(index), mode,
        str(min(epoch+1010, time.time()+150)),
    ]
    with (directory/"step.stdout").open("x") as out,(directory/"step.stderr").open("x") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            child.wait(timeout=max(1,min(165,epoch+1030-time.time())))
        except subprocess.TimeoutExpired:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=4)
    observed = wait_row(label)
    return {"row":observed,"rc":child.returncode,"success":(
        child.returncode==0 and observed[2:4]==["COMPLETED","0:0"]
        and "gres/dcu="+str(len(NODES)*8) in observed[-1].split(","))}


def check_disjoint_steps(lines, parent, nodes, expand):
    need(bool(lines), "CURRENT_STEP_INVENTORY_MISSING")
    for line in lines:
        step = fields(line)
        step_id = step.get("StepId", "")
        need(step_id.startswith(parent + "."), "STEP_PARENT")
        suffix = step_id.partition(".")[2]
        if suffix in ("batch", "extern"):
            continue
        placement = step.get("NodeList")
        need(suffix.isdecimal() and placement not in (None, "(null)"),
             "CURRENT_NUMBERED_OR_UNKNOWN_STEP")
        observed_nodes = set(expand(placement))
        need(bool(observed_nodes) and not observed_nodes.intersection(nodes), "CURRENT_STEP_NODE_CONFLICT")


def main(epoch, request, stage, parent):
    global ROOT, PARENT, NODES, PAIRS, PORT_BASE
    claim = json.loads((SOURCE/"CLAIM.json").read_bytes())
    if request != claim["request"] or not 0 <= time.time()-epoch < (45 if stage=="sample" else 565):
        raise ValueError("OWNED_HOST_EPOCH")
    PARENT=parent
    if stage != "wide" or parent not in ("1087656","1103606"):
        raise ValueError("EXACT_FROZEN_WAVE")
    PORT_BASE = 30620
    PAIRS=json.loads((SOURCE/"PLACEMENT.json").read_bytes())[stage][PARENT]
    NODES=tuple(n for pair in PAIRS for n in pair)
    if len(set(NODES))!=len(NODES) or any(len(pair)!=2 for pair in PAIRS):
        raise ValueError("DISJOINT_OWNED_PAIRS")
    ROOT=SOURCE/f"{stage}-{parent}"
    ROOT.mkdir(mode=0o700)
    nodes = sorted(n for n,(j,_) in node_map(json.loads((SOURCE/"ledger.json").read_bytes())).items() if j==PARENT)
    expand = lambda v: query("scontrol","show","hostnames",v)
    row, = query("scontrol","show","job",PARENT,"-o")
    parent_gate(row,PARENT,os.getuid(),nodes,expand)
    if not set(NODES) <= set(nodes):
        raise ValueError("PAIR_EXACT_MEMBERSHIP")
    current_steps = query("scontrol", "show", "step", PARENT, "-o")
    check_disjoint_steps(current_steps, PARENT, NODES, expand)
    verify_providers()
    prefix=f"glm-tp16-shca-algo2-{stage}-{request}"
    before=clean_before(prefix+"-pre",epoch)
    save("pre-gate.json",before)
    terminal={}
    outcome=None
    try:
        for index,mode in enumerate(("ring","tree","tree","ring")):
            if time.time()>=epoch+870:
                raise TimeoutError("FIXED_ARM_START_DEADLINE")
            item=one_arm(prefix+"-"+str(index)+"-"+mode,index,mode,epoch)
            terminal[str(index)]=item
            if not item["success"]:
                raise ValueError("ARM_FAILED_ACTUAL_ACCOUNTING")
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
    raise SystemExit(main(float(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4]))
