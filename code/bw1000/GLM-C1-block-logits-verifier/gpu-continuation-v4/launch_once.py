"""Claim and submit one CPU keeper for the published GLM continuation scope."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import re
import socket
import subprocess
from support import D,R,RUN,LABEL,NODES,OUTPUT,clean_environment,publish,verify

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest-sha",required=True)
    p.add_argument("--publication-commit",required=True)
    a=p.parse_args()
    assert re.fullmatch("[0-9a-f]{40}",a.publication_commit)
    claim=verify(a.manifest_sha)
    prefix=LABEL+"-"+RUN
    assert not OUTPUT.exists() and not list((R/"results").glob(prefix+"*")) and not list((R/"logs").glob(prefix+"-*"))
    for parent,nodes in (("910371",NODES),("910407",["a14r2n20"])):
        job=subprocess.check_output(["scontrol","show","job",parent,"-o"],text=True,timeout=15)
        assert "JobState=RUNNING" in job and "UserId=aclsylqidf(" in job
        members=subprocess.check_output(["scontrol","show","hostnames",
            re.search(r" NodeList=([^ ]+)",job).group(1)],text=True,timeout=10).splitlines()
        assert set(nodes)<=set(members)
    rows=subprocess.check_output(["sacct","-S","2026-11-05","-j","910371,910407","-n","-P",
        "-o","JobIDRaw,JobName%180,State"],text=True,timeout=15)
    assert not any(prefix in row for row in rows.splitlines())
    intent=R/"results"/(prefix+"-intent.json")
    record=dict(publication=a.publication_commit,manifest_SHA256=a.manifest_sha,
        HOST=socket.gethostname(),UTC=datetime.now(timezone.utc).isoformat(),
        invoker_PID=os.getpid(),invoker_startticks=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],
        planned_cards=16,planned_nodes=NODES,keeper_submitted=False)
    publish(intent,record)
    env=clean_environment(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1",HIP_VISIBLE_DEVICES="-1",ROCR_VISIBLE_DEVICES="-1")
    args=["srun","--jobid=910407","--overlap","--exact","--immediate=30","--cpu-bind=none",
        "-N1","-n1","-c2","--gres=none","--mem=8192M","-w","a14r2n20","-t","01:05:00",
        "--job-name="+prefix+"-keeper","--export=ALL","--output="+str(R/"logs"/(prefix+"-keeper-%J.log")),
        "python3","-B",str(D/"controller.py"),"--manifest-sha",a.manifest_sha,
        "--publication-commit",a.publication_commit]
    with (R/"logs"/(prefix+"-launcher.log")).open("x") as log:
        proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(keeper_submitted=True,srun_PID=proc.pid,owned_PGID=proc.pid,
        startticks=Path("/proc/%d/stat"%proc.pid).read_text().rsplit(")",1)[1].split()[19])
    publish(R/"results"/(prefix+"-launch.json"),record)
    print(json.dumps(record))

if __name__=="__main__":
    main()
