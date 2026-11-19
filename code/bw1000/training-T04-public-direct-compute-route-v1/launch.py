"""Dispatch exactly one CPU direct-route check over ordinary project SSH."""

import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import time


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
HELPER=Path("C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py")
HELPER_SHA="3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def need(ok,detail):
    if not ok:
        raise ValueError(detail)


def main(publication):
    rel=HERE.relative_to(REPO).as_posix()
    git=["git","-c","safe.directory="+str(REPO)]
    manifest=(HERE/"SOURCE_FILES.json").read_bytes()
    for name, pin in json.loads(manifest)["files"].items():
        raw=(HERE/name).read_bytes()
        need(dict(bytes=len(raw),SHA256=sha(raw))==pin,"SOURCE_PIN")
    known=subprocess.check_output([*git,"cat-file","blob",publication+":"+rel+"/SOURCE_FILES.json"],cwd=REPO,timeout=10)
    need(manifest==known and sha(HELPER.read_bytes())==HELPER_SHA,"PUBLISHED_PROBE")
    local=REPO/"local-only"/HERE.name
    local.mkdir(mode=0o700)
    epoch=time.time()
    plan=json.loads((HERE/"PLAN.json").read_bytes())
    context={**plan,"epoch":epoch,"deadline":epoch+85,"publication":publication}
    with (local/"intent.json").open("x",encoding="utf-8") as f:json.dump(context,f)
    task=(HERE/"probe.py").read_text(encoding="utf-8")
    wrapper="""import json,os,subprocess,time
p=PAYLOAD
q=subprocess.check_output(["scontrol","show","job","1087656","-o"],text=True,timeout=8)
assert "JobState=RUNNING" in q and "JobId=1087656" in q
nodes=next(x.partition("=")[2] for x in q.split() if x.startswith("NodeList="))
assert "a14r3n14" in subprocess.check_output(["scontrol","show","hostnames",nodes],text=True,timeout=6).splitlines()
cols="JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%120"
def exact():
 raw=subprocess.check_output(["sacct","-n","-P","-j","1087656","--format="+cols],text=True,timeout=9)
 return [x.split("|") for x in raw.splitlines() if x.split("|")[1:2]==[p["label"]]]
assert time.time()<p["epoch"]+25 and not exact()
assert "UserId=" in q and "("+str(os.getuid())+")" in next(x for x in q.split() if x.startswith("UserId="))
args=["srun","--jobid=1087656","--nodelist=a14r3n14","--nodes=1","--ntasks=1","--cpus-per-task=1","--mem=512M",
      "--gres=none","--exact","--immediate=4","--time=00:01:00","--job-name="+p["label"],"/usr/bin/python3","-B","-c",TASK]
r=subprocess.run(args,capture_output=True,timeout=min(48,p["deadline"]-time.time()-10))
data=json.loads(r.stdout) if r.returncode==0 and 0<len(r.stdout)<=8192 else None
rows=exact()
for _ in range(2):
 if rows and not any(row[2] in ("RUNNING","PENDING","COMPLETING") for row in rows):break
 time.sleep(1); rows=exact()
print(json.dumps(dict(request=p["request"],data=data,accounting=rows,accounting_columns=cols,
                      srun_rc=r.returncode,stderr_bytes=len(r.stderr))))
""".replace("PAYLOAD",repr(context)).replace("TASK",repr(task))
    ssh=runpy.run_path(str(HELPER))["ssh"]
    command="/usr/local/python3.12/bin/python3 -I -S -B -c "+shlex.quote(wrapper)
    result=subprocess.run([ssh[0],"-o","StrictHostKeyChecking=yes",*ssh[1:],command],capture_output=True,timeout=90)
    report=dict(request=context["request"],transport_rc=result.returncode,stderr_bytes=len(result.stderr),
                stderr_sha=sha(result.stderr),result=None)
    if result.returncode==0:
        report["result"]=json.loads(result.stdout)
    with (local/"result.json").open("x",encoding="utf-8") as f:json.dump(report,f)
    print(json.dumps(report,allow_nan=False))
    return int(result.returncode!=0)


if __name__=="__main__":
    raise SystemExit(main(sys.argv[1]))
