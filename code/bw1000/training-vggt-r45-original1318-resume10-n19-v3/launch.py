"""Create the one-shot target payload with explicitly injected fresh authority."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex

from resume_bindings import ROOT, OLD, LABEL, REQUEST, DISPATCH, NODE, PLACEMENT_HISTORY_SHA, PREVIOUS_FAILURE_SHA, PREVIOUS_V2_FAILURE_SHA, need


PARENT="1103606"
DEST=ROOT/"results"/LABEL
HOST_COMMAND="/usr/local/python3.12/bin/python3 -I -S -B -u -"


def srun_command(phase,payload,wall):
    native=phase=="gpu"
    cmd=["srun","--jobid="+PARENT,"--exact","--immediate=5","--cpu-bind=none","-N1","-n1",
         "-c4" if native else "-c2","--mem=65536M" if native else "--mem=2048M",
         "--gres=dcu:1" if native else "--gres=none","-w",NODE,"-t",wall,"--job-name="+LABEL+"-"+phase]
    return cmd+list(payload)


def container_command(new):
    results=ROOT/"results"
    args=["/public/software/apps/apptainer/1.3.4/bin/apptainer","exec","--no-home","--pwd","/"]
    bindings=((new,"/pilot","rw"),(OLD/"data","/pilot/data","ro"),(OLD/"framework","/pilot/framework","ro"),
        (OLD/"arms/continuous","/reference","ro"),(results/"training-vggt-original-public-model-v1/model","/weights","ro"),
        (results/"training-vggt-native-provider-HIP-CPU-v1/upstream","/vggt-source","ro"),
        (results/"training-t06-sdxl-runtime-v1","/training-runtime","ro"),(Path("/opt/hyhal"),"/opt/hyhal","ro"))
    for source,target,mode in bindings:
        args.extend(("--bind",f"{source}:{target}:{mode}"))
    body=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
          "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "
          "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
          "PYTHONPATH=/training-runtime/site:/pilot/framework/generation/src:/pilot/framework; "
          "exec "+shlex.join(["/usr/bin/timeout","--kill-after=5s","1180s","python3","-B","-u","/pilot/src/owner.py","native"]))
    return args+[str(ROOT/"containers/dtk26-sandbox-r1"),"/bin/bash","--noprofile","--norc","-c",body]


def make_payload(directory,publication,authority):
    need(re.fullmatch(r"[0-9a-f]{40}",publication),"actual published source revision required")
    manifest=(directory/"SOURCE_FILES.json").read_bytes()
    members=json.loads(manifest)["files"]
    need(0<len(members)<=20 and all(Path(name).name==name for name in members),"bounded flat source set")
    files={name:(directory/name).read_bytes() for name in members}
    need(all(dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())==members[name] for name,raw in files.items()),
         "exact source payload bytes")
    claim=json.loads(files["CLAIM.json"])
    need(claim["request"]==REQUEST and claim["dispatch_request"]==DISPATCH and claim["GPU_GO"] is True
         and claim["fresh_root_authority_SHA256"]==sha256(authority).hexdigest() and 0<len(authority)<=16384,
         "independent actual root authority required")
    pin=sha256(manifest).hexdigest()
    data={name:b64encode(raw).decode("ascii") for name,raw in files.items()}
    text=("import base64,json,os,runpy,sys,time\nfrom pathlib import Path\n"
          +"root=Path("+repr(DEST.as_posix())+");os.umask(0o077);root.mkdir(mode=0o700)\n"
          +"src=root/'src';src.mkdir(mode=0o700)\n"
          +"for name in ('data','framework','arms'): (root/name).mkdir(mode=0o700)\n"
          +"files="+repr(data)+"\n"
          +"def put(path,raw):\n with path.open('xb') as f:\n  f.write(raw);f.flush();os.fsync(f.fileno())\n"
          +"for name,content in files.items(): put(src/name,base64.b64decode(content,validate=True))\n"
          +"put(root/'SOURCE_FILES.json',base64.b64decode("+repr(b64encode(manifest).decode())+"))\n"
          +"put(root/'authority.json',base64.b64decode("+repr(b64encode(authority).decode())+"))\n"
          +"epoch=time.time();put(root/'intent.json',json.dumps(dict(request="+repr(REQUEST)+
          ",dispatch="+repr(DISPATCH)+",epoch=epoch,deadline=epoch+1800,publication="+repr(publication)+
          ",manifest_SHA256="+repr(pin)+"),sort_keys=True).encode())\n"
          +"sys.path.insert(0,str(src));runpy.run_path(str(src/'owner.py'),run_name='__main__')\n")
    raw=text.encode()
    need(0<len(raw)<=240000,"bounded source-only payload wire")
    compile(raw,"vg-resume10-target.py","exec")
    return raw


def file_once(path,raw):
    import os
    with path.open("xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def main(publication, transport):
    """Invoke the authorized coordinator's four-argument transport once."""
    import time
    need(callable(transport),"explicit original coordinator transport required")
    here=Path(__file__).resolve().parent
    repo=here.parents[2]
    need(here==repo/"code/bw1000"/LABEL,"canonical published launcher location")
    private=repo/"local-only"
    authority=private/(LABEL+"-authority")/(REQUEST+".json")
    need(authority.resolve(strict=True)==authority and authority.is_file()
         and 0<authority.stat().st_size<=16384,"exact original private authority path")
    raw=authority.read_bytes()
    approval=json.loads(raw)
    need(approval["request"]==REQUEST and approval["label"]==LABEL and approval["parent"]==PARENT
         and approval["node"]==NODE and approval["original_node"]=="f13r4n09"
         and approval["root_sole_dispatcher"] is True
         and approval["ordinary_exclusion_until_independent_post"] is True and approval["approved_one_DCU_once"] is True
         and approval["prior_node_unknown_conflict"] is False
         and approval["placement_history_SHA256"]==PLACEMENT_HISTORY_SHA
         and approval["previous_failed_scope_SHA256"]==PREVIOUS_FAILURE_SHA
         and approval["previous_v2_failed_scope_SHA256"]==PREVIOUS_V2_FAILURE_SHA,"actual new n19 custody required")
    now=time.time()
    need(0<=now-approval["observed_epoch"]<=90 and now+1800<=approval["valid_through_epoch"],
         "fresh original authority and full epoch")
    body=make_payload(here,publication,raw)
    destination=private/(LABEL+"-dispatch-once")
    destination.mkdir(mode=0o700)
    intent={"request":REQUEST,"transport_request":DISPATCH,"publication":publication,"node":NODE,
            "original_node":"f13r4n09","body_SHA256":sha256(body).hexdigest(),"epoch":now,
            "authority_SHA256":sha256(raw).hexdigest(),"replay_allowed":False}
    file_once(destination/"intent.json",(json.dumps(intent,sort_keys=True)+"\n").encode())
    began=time.time()
    need(0<=began-approval["observed_epoch"]<=90 and began+1800<=approval["valid_through_epoch"],
         "authority expired before callback")
    limit=min(1840,max(0,approval["valid_through_epoch"]-began))
    file_once(destination/"attempt.json",(json.dumps({"request":REQUEST,"started_epoch":began,
                  "timeout_seconds":limit,"attempted":True,"replay_allowed":False},sort_keys=True)+"\n").encode())
    rc=None
    error=None
    try:
        with (destination/"stdout.private").open("xb") as out,(destination/"stderr.private").open("xb") as err:
            r=transport(body,out,err,limit)
            rc=r.returncode
    except Exception as e:
        error=type(e).__name__
    file_once(destination/"transport.json",(json.dumps({"request":REQUEST,"returncode":rc,"error_class":error,
               "scope_terminal_verified":False,"replay_allowed":False,
               "state":"UNKNOWN" if rc is None or rc==255 else "RETURNED_AWAITING_INDEPENDENT_READ"},sort_keys=True)+"\n").encode())
    return 1 if rc is None else rc


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication",required=True)
    p.add_argument("--authority",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    payload=make_payload(Path(__file__).resolve().parent,a.publication,a.authority.read_bytes())
    with a.output.open("xb") as f:
        f.write(payload)
    print(json.dumps({"transport_request":DISPATCH,"scope_request":REQUEST,"command":HOST_COMMAND,
                     "payload_bytes":len(payload),"payload_SHA256":sha256(payload).hexdigest(),
                     "transport_invoked":False},sort_keys=True))
