"""Verify publication and start one bounded VGGT training controller."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = "/public/home/aclsylqidf/hygon-inference-refactor-20261014/results"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main(publication, transport):
    git = ["git", "-c", "safe.directory=" + REPO.as_posix(), "-C", str(REPO)]
    def read(*args):
        return subprocess.check_output(git + list(args), timeout=12)
    relative = HERE.relative_to(REPO).as_posix()
    remote = read("ls-remote","--exit-code","origin","refs/heads/bw1000-experiments").decode().split()
    if remote != [publication,"refs/heads/bw1000-experiments"]:
        raise ValueError("Published commit does not match the deployed task")
    if read("cat-file","blob",publication+":"+relative+"/launch.py") != Path(__file__).read_bytes():
        raise ValueError("Launcher differs from its publication")
    data = read("cat-file","blob",publication+":"+relative+"/CLAIM.json")
    if data != (HERE/"CLAIM.json").read_bytes():
        raise ValueError("Claim differs from its publication")
    claim = json.loads(data)
    if (claim["label"],claim["parent"],claim["node"],claim["planned_updates"]) != (HERE.name,"1103606","f13r4n09",45):
        raise ValueError("Unexpected VGGT resource or update scope")
    payload = {"publication":publication,"request":claim["request"],"files":{}}
    frozen = {}
    for filename, pin in claim["sources"].items():
        raw=read("cat-file","blob",publication+":"+pin["path"])
        if digest(raw)!=pin["sha"] or (REPO/pin["path"]).read_bytes()!=raw:
            raise ValueError("Frozen source differs: "+filename)
        frozen[filename]=raw
        payload["files"]["src/"+filename]=dict(sha=pin["sha"],bytes=len(raw),b64=base64.b64encode(raw).decode("ascii"))
    program=("PAYLOAD="+repr(payload)+"\n").encode("ascii")+frozen["run_tum_continuous_once.py"]
    guard=ROOT+"/"+HERE.name+"-bootstrap"
    payload2=dict(body=base64.b64encode(program).decode("ascii"),sha=digest(program),guard=guard,request=claim["request"])
    wrapper="""import base64,hashlib,json,os,pathlib,subprocess,sys
p=PARAMETERS
raw=base64.b64decode(p["body"],validate=True)
assert hashlib.sha256(raw).hexdigest()==p["sha"]
guard=pathlib.Path(p["guard"])
guard.mkdir(mode=0o700)
program=guard/"controller.py"
with program.open("xb") as handle:
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())
with (guard/"controller.log").open("xb") as log:
    proc=subprocess.Popen([sys.executable,"-I","-S","-B",str(program)],stdin=subprocess.DEVNULL,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(request=p["request"],controller_pid=proc.pid,body_SHA256=p["sha"])
with (guard/"launch.json").open("x") as handle:
    json.dump(receipt,handle)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(receipt),flush=True)
""".replace("PARAMETERS",repr(payload2)).encode("ascii")
    record=REPO/"local-only"/(HERE.name+"-first-dispatch")
    record.mkdir()
    (record/"intent.json").write_text(json.dumps(dict(request=claim["request"],publication=publication,
         body_SHA256=digest(program),transport_replay=False,scope=HERE.name)),encoding="utf8")
    with (record/"stdout.private").open("xb") as out,(record/"stderr.private").open("xb") as err:
        result=transport(wrapper,out,err)
    (record/"transport.json").write_text(json.dumps(dict(rc=result.returncode,replay=False)),encoding="utf8")
    if result.returncode==0:
        reply=json.loads((record/"stdout.private").read_bytes())
        if reply["request"]!=claim["request"] or reply["body_SHA256"]!=digest(program):
            raise ValueError("Original launch receipt drift")
        print(json.dumps(reply))
    return result.returncode
