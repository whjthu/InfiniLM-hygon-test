"""Launch the independently staged T09 SFT repeat once through ROOT's injected transport."""
import json
from hashlib import sha256
from pathlib import Path
import re
import stat
import subprocess

from contract import LABEL, REQUEST, RUN, WHOLE, Budget, need, save
from held_child import cleanup_held
from placement import NODE, PARENT, placement_config


HOST_COMMAND="/usr/local/python3.12/bin/python3 -I -S -B -u -"


def load(path):
    before=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode) and 0<before.st_size<=65536,
         "T09_LOCAL_CANONICAL_BOUNDED_INPUT")
    with path.open("rb") as stream:
        raw=stream.read(65537)
    after=path.lstat()
    identity=lambda x:(x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_dev)
    need(identity(before)==identity(after) and len(raw)==before.st_size,"T09_LOCAL_SOURCE_CHANGED")
    return json.loads(raw),{"SHA256":sha256(raw).hexdigest(),"bytes":len(raw)}


def prepare(publication, stub, package):
    claim, _=load(package/"CLAIM.json")
    need(claim["GPU_GO"] is claim["root_GO"] is True and claim["request_id"]==REQUEST,
         "T09_DISABLED_OR_WRONG_REQUEST")
    placement_config()
    need(re.fullmatch("[0-9a-f]{40}",publication) and stub["publication"]==publication
         and stub["request_id"]==REQUEST and stub["node"]==NODE and stub["parent"]==PARENT
         and stub["scope_is_new"] is True,"T09_EXPLICIT_LOCAL_FRESH_SCOPE_BINDING")
    source, manifest=load(package/"SOURCE_FILES.json")
    need(stub["manifest_SHA256"]==manifest["SHA256"]
         and all(re.fullmatch("[0-9a-f]{64}",stub[key]) for key in ("preflight_SHA256","authority_SHA256","route_SHA256")),
         "T09_EXACT_STAGED_SOURCE_AND_PRIVATE_PROOF_PINS")
    for name,record in source["files"].items():
        need(Path(name).name==name,"T09_LOCAL_FLAT_SOURCE_MEMBER")
        path=package/name
        info=path.lstat()
        need(path.resolve(strict=True)==path and stat.S_ISREG(info.st_mode) and info.st_size==record["bytes"]
             and 0<info.st_size<=1<<20,"T09_LOCAL_SOURCE_MEMBER_EXTENT")
        need(sha256(path.read_bytes()).hexdigest()==record["SHA256"],"T09_LOCAL_SOURCE_MEMBER_SHA")
    epoch=stub["epoch"]
    budget=Budget(epoch)
    need(budget.left()>WHOLE-90,"T09_ORIGINAL_ADMISSION_EPOCH_ALREADY_STALE")
    argv=[str(RUN/"src/controller.py"),"--workspace",str(RUN),"--epoch",str(epoch),
          "--source-manifest-sha256",manifest["SHA256"],"--preflight-sha256",stub["preflight_SHA256"],
          "--authority-sha256",stub["authority_SHA256"],"--route-sha256",stub["route_SHA256"],
          "--publication",publication]
    python=("import runpy,sys\n"
            f"sys.path.insert(0,{str(RUN/'src')!r})\n"
            f"sys.argv={argv!r}\n"
            f"runpy.run_path({argv[0]!r},run_name='__main__')\n")
    body=python.encode()
    compile(body,"target-readiness-body","exec")
    return budget,body,{"request_id":REQUEST,"publication":publication,"epoch":epoch,"deadline":epoch+WHOLE,
                        "remote_command":HOST_COMMAND,"argv":argv,"parent":PARENT,"node":NODE,
                        "body_SHA256":sha256(body).hexdigest(),"source_manifest_SHA256":manifest["SHA256"]}


def main(publication, transport):
    need(callable(transport),"T09_INJECTED_EXISTING_ROOT_TRANSPORT_ONLY")
    package=Path(__file__).resolve().parent
    repo=package.parents[2]
    need(package==repo/"code"/"bw1000"/LABEL,"T09_CANONICAL_PUBLISHED_REPOSITORY")
    local=repo/"local-only"
    need(local.is_dir() and local.resolve(strict=True)==local,"T09_CANONICAL_PRIVATE_LOCAL")
    stub,_=load(local/(LABEL+"-placement")/(REQUEST+".json"))
    budget,body,context=prepare(publication,stub,package)
    attempt=local/(LABEL+"-dispatch-"+REQUEST)
    attempt.mkdir(mode=0o700)
    save(attempt/"intent.json",dict(context,already_attempted=False,replay_permitted=False))
    child=None
    held={}
    rc=None
    error=None
    try:
        need(budget.left()>WHOLE-90,"T09_ORIGINAL_EPOCH_CHANGED_BEFORE_SEND")
        save(attempt/"attempt.json",dict(context,dispatch_attempted=True))
        child=transport(context)
        need(isinstance(child,subprocess.Popen) and child.stdin is not None and child.stdout is not None
             and child.stderr is not None,"T09_ACTUAL_HELD_PROCESS")
        stdout,stderr=child.communicate(input=body,timeout=budget.left()+40)
        need(len(stdout)<=2<<20 and len(stderr)<=2<<20,"T09_LOCAL_STDIO_BOUND")
        (attempt/"stdout.private").write_bytes(stdout)
        (attempt/"stderr.private").write_bytes(stderr)
        rc=child.returncode
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        error=type(exc).__name__
    finally:
        try:
            anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True) if child is not None else {}
            held=cleanup_held(child,anchor,context["deadline"]+40)
        except (OSError,ValueError,AttributeError,subprocess.SubprocessError) as exc:
            held={"error_class":type(exc).__name__,"reaped":False}
        receipt=dict(context,returncode=rc,error_class=error,held=held,replay_permitted=False,
                     state="UNKNOWN_NO_REPLAY" if rc in (None,255) else "RETURNED_REQUIRES_INDEPENDENT_TERMINAL")
        save(attempt/"receipt.json",receipt)
    return receipt
