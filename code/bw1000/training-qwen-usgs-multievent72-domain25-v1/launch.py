"""One exact published request over root's existing bounded ordinary SSH callback."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
G=HERE.parents[2]
REQUEST="c933ba7359f84e8b9b80ac40d2d30964"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(publication):
    epoch=time.time()
    mono_end=time.monotonic()+2400
    def left(reserve=10):
        result=mono_end-time.monotonic()-reserve
        if result<=0:
            raise TimeoutError("Original local task epoch exhausted")
        return result
    spec=importlib.util.spec_from_file_location("qwen_exposure25_verify",HERE/"verify_package.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(HERE,adopted=True)
    git=["git","--no-optional-locks","-c","safe.directory="+G.as_posix(),"-C",str(G)]
    def query(*args):
        return subprocess.check_output(git+list(args),timeout=min(12,left()))
    if query("ls-remote","--exit-code","origin","refs/heads/bw1000-experiments").decode().split()!=[publication,"refs/heads/bw1000-experiments"]:
        raise ValueError("Publication is not the ordinary remote head")
    relative=HERE.relative_to(G).as_posix()
    mf=(HERE/"SOURCE_FILES.json").read_bytes()
    if query("cat-file","blob",publication+":"+relative+"/SOURCE_FILES.json")!=mf:
        raise ValueError("Published manifest differs")
    source={"SOURCE_FILES.json":base64.b64encode(mf).decode()}
    for name,pin in json.loads(mf)["files"].items():
        raw=query("cat-file","blob",publication+":"+relative+"/"+name)
        if raw!=(HERE/name).read_bytes() or len(raw)!=pin["bytes"] or sha(raw)!=pin["SHA256"]:
            raise ValueError("Published task member differs: "+name)
        source[name]=base64.b64encode(raw).decode()
    claim_raw=(HERE/"CLAIM.json").read_bytes()
    payload=dict(publication=publication,request=REQUEST,epoch=epoch,deadline=epoch+2400,files=source,
                 source_manifest_SHA256=sha(mf),claim_SHA256=sha(claim_raw))
    body=("PAYLOAD="+repr(payload)+"\n").encode("ascii")+(HERE/"receive.py").read_bytes()
    out=G/"local-only"/(HERE.name+"-original-dispatch")
    out.mkdir(mode=0o700)
    intent=dict(request=REQUEST,publication=publication,epoch=epoch,deadline=epoch+2400,
                source_manifest_SHA256=sha(mf),claim_SHA256=sha(claim_raw),body_SHA256=sha(body),
                max_invocations=1,unknown_replay_allowed=False)
    (out/"intent.json").write_text(json.dumps(intent,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return body,out,intent,left


def main(publication,transport):
    started=time.monotonic()
    body,out,intent,left=prepare(publication)
    (out/"attempt.json").write_text(json.dumps(dict(request=REQUEST,attempted=True,
                      original_epoch=intent["epoch"],attempt_epoch=time.time()))+"\n",encoding="utf8")
    rc=None
    error_type=None
    try:
        with (out/"stdout.private").open("xb") as stdout,(out/"stderr.private").open("xb") as stderr:
            rc=transport(body,stdout,stderr,left()).returncode
    except BaseException as exc:
        error_type=type(exc).__name__
    result=dict(request=REQUEST,publication=publication,epoch=intent["epoch"],deadline=intent["deadline"],
                transport_rc=rc,error_type=error_type,unknown=rc is None or rc==255,
                transport_return_not_GPU_accounting=True,replay_allowed=False,client_wall_seconds=time.monotonic()-started)
    (out/"receipt.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return 1 if rc is None else rc
