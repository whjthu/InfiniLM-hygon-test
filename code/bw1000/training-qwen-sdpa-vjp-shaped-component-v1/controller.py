"""Execute one distinct Qwen SDPA component and bind exact Slurm accounting."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

D=Path(__file__).resolve().parent
R=D.parent.parent
P=json.loads((D/"PLAN.json").read_bytes())


def need(ok, error):
    if not ok:
        raise ValueError(error)


def query(*args):
    return subprocess.check_output(args,text=True,timeout=8).splitlines()


def save(name,value):
    with (D/name).open("x",encoding="utf-8") as out:
        json.dump(value,out,allow_nan=False,sort_keys=True)
        out.write("\n")


def main():
    context=json.loads((D/"dispatch-context.json").read_bytes()); epoch=context["epoch"]; deadline=epoch+220
    need(time.time()<epoch+28,"ORIGINAL_ADMISSION_WINDOW")
    parent=P["parent"]; node=P["node"]; label=P["label"]
    rows=query("scontrol","show","job",parent,"-o")
    need(len(rows)==1,"EXACT_PARENT")
    row=rows[0]; fields=dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)",row))
    need(fields["JobId"]==parent and fields["JobState"]=="RUNNING" and fields["UserId"].endswith("("+str(os.getuid())+")"),
         "FRESH_OWNED_PARENT")
    need(node in query("scontrol","show","hostnames",fields["NodeList"]),"OWN_QWEN_NODE")
    for x in query("squeue","-h","--steps","-j",parent,"-o","%i|%N"):
        job,nodes=x.split("|")
        if job.rsplit(".",1)[-1] in ("batch","extern"):
            continue
        need(job.rsplit(".",1)[-1].isdecimal() and nodes and node not in query("scontrol","show","hostnames",nodes),
             "NUMBERED_OR_UNKNOWN_STEP_CONFLICT")
    cols="JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%180"
    def accounting():
        return [x.split("|") for x in query("sacct","-n","-P","-j",parent,"--format="+cols) if x.split("|")[1:2]==[label]]
    need(not accounting(),"REPLAY_FORBIDDEN")
    swift=R/"results/training-vlm-swift-offline-runtime-v1"
    processor=R/"results/training-minicpm-hygon-processor-r2/cpu/runtime-site"
    text=(processor/"transformers/integrations/sdpa_attention.py").read_bytes()
    need(len(text)==4851 and hashlib.sha256(text).hexdigest()==P["provider_sha"],"PINNED_PRELAUNCH_PROVIDER")
    image=R/"containers/dtk26-sandbox-r1"
    need(image.is_dir(),"QUALIFIED_DTK_CONTAINER")
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "
           "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
           "PYTHONPATH=/minicpm-transformers:/swift-runtime/site; exec python3 -B "
           +shlex.quote(str(D/"worker.py")))
    args=["srun","--jobid="+parent,"--nodelist="+node,"--nodes=1","--ntasks=1","--cpus-per-task=8",
          "--mem=16G","--gres=dcu:1","--exact","--immediate=4","--time=00:02:00","--job-name="+label,
          "/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--rocm","--no-home","--pwd","/",
          "--bind",f"{D}:{D}:rw","--bind",f"{swift}:/swift-runtime:ro","--bind",f"{processor}:/minicpm-transformers:ro",
          "--bind","/opt/hyhal:/opt/hyhal:ro",str(image),"/bin/bash","--noprofile","--norc","-c",shell]
    save("step-intent.json",dict(context,node=node,expected_GRES=1))
    proc=subprocess.run(args,capture_output=True,timeout=min(150,deadline-time.time()-13))
    acc=accounting()
    for _ in range(2):
        if acc and all(x[2] not in ("RUNNING","PENDING","COMPLETING") for x in acc):break
        time.sleep(1);acc=accounting()
    output=D/"component-result.json"
    received=None
    if output.is_file() and 0<output.stat().st_size<=32768:
        raw=output.read_bytes();result=json.loads(raw)
        need(result["request"]==P["request"],"ORIGINAL_COMPONENT_REQUEST")
        received=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),step=result["step"],
                      cases={n:dict(default_bit_stable=r["default"]["bitwise_stable"],math_bit_stable=r["math"]["bitwise_stable"],
                         default_operations=r["default"]["profile_operator_keys"],math_operations=r["math"]["profile_operator_keys"],
                         default_vs_math=r["default_vs_math"],min_default_seconds=min(x["wall_seconds"] for x in r["default"]["calls"]),
                         min_math_seconds=min(x["wall_seconds"] for x in r["math"]["calls"])) for n,r in result["cases"].items()},
                      peak_allocated_bytes=result["peak_allocated_bytes"],geometry=result["inputs"])
    report=dict(request=P["request"],slurm_rc=proc.returncode,accounting_columns=cols,actual_accounting=acc,
                data=received,stderr_bytes=len(proc.stderr),stderr_sha=hashlib.sha256(proc.stderr).hexdigest(),
                error_classes=[name for name in ("ModuleNotFoundError","ImportError","ValueError","RuntimeError","TimeoutError","PermissionError","TypeError","AssertionError")
                               if name.encode()+b":" in proc.stderr])
    save("terminal.json",report)
    print(json.dumps(report,allow_nan=False))


if __name__=="__main__":
    main()
