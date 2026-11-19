"""Stage the pinned component and invoke its single scheduler scope."""
import base64
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time


D=Path(__file__).resolve().parent
G=D.parents[2]
HELPER=Path("C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py")


def sha(x):return hashlib.sha256(x).hexdigest()


def main(publication):
    git=["git","-c","safe.directory="+str(G)]
    m=(D/"SOURCE_FILES.json").read_bytes()
    rel=D.relative_to(G).as_posix()
    if m!=subprocess.check_output([*git,"cat-file","blob",publication+":"+rel+"/SOURCE_FILES.json"],cwd=G,timeout=12):
        raise ValueError("EXACT_PUBLISHED_COMPONENT")
    files={}
    for name,pin in json.loads(m)["files"].items():
        raw=(D/name).read_bytes()
        if len(raw)!=pin["bytes"] or sha(raw)!=pin["SHA256"]:
            raise ValueError("EXACT_SOURCE_MEMBER")
        files[name]=base64.b64encode(raw).decode()
    if sha(HELPER.read_bytes())!="3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1":
        raise ValueError("EXPECTED_SSH_HELPER")
    plan=json.loads((D/"PLAN.json").read_bytes()); epoch=time.time()
    context=dict(publication=publication,epoch=epoch,deadline=epoch+220,request=plan["request"],manifest_sha=sha(m))
    local=G/"local-only"/D.name;local.mkdir(mode=0o700)
    (local/"intent.json").write_text(json.dumps(context),encoding="utf-8")
    root="/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/"+D.name
    script="""import base64,json,pathlib,runpy,sys,time
p=PAYLOAD
assert time.time()<p["context"]["epoch"]+20
root=pathlib.Path(p["root"]);root.mkdir(mode=0o700)
for name,blob in p["files"].items():
 assert pathlib.Path(name).name==name
 with (root/name).open("xb") as f:f.write(base64.b64decode(blob,validate=True))
with (root/"dispatch-context.json").open("x") as f:json.dump(p["context"],f)
sys.path.insert(0,str(root));runpy.run_path(str(root/"controller.py"),run_name="__main__")
""".replace("PAYLOAD",repr(dict(root=root,files=files,context=context)),1)
    ssh=runpy.run_path(str(HELPER))["ssh"]
    r=subprocess.run([ssh[0],"-o","StrictHostKeyChecking=yes",*ssh[1:],"/usr/local/python3.12/bin/python3 -I -S -B -"],input=script.encode(),capture_output=True,timeout=224)
    report=dict(request=plan["request"],transport_rc=r.returncode,stderr_bytes=len(r.stderr),stderr_sha=sha(r.stderr),
                error_classes=[x for x in ("ValueError","TimeoutExpired","PermissionError","KeyError") if x.encode()+b":" in r.stderr],
                result=json.loads(r.stdout) if r.returncode==0 and 0<len(r.stdout)<=32768 else None)
    with (local/"outcome.json").open("x",encoding="utf-8") as out:json.dump(report,out)
    print(json.dumps(report,allow_nan=False))
    return int(r.returncode!=0)


if __name__=="__main__":
    raise SystemExit(main(sys.argv[1]))
