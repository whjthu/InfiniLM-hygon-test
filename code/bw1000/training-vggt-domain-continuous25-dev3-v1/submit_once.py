"""Prepare and submit one published development scope after the actual training terminal gate."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys
import uuid

from render_controller import LABEL, DEV_SET_SHA256, controller, scorer
from training_guard import check_binding

G = Path(__file__).resolve().parents[3]
HELPER_SHA256 = "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(raw,expected):
    if sha(raw)!=expected:
        raise ValueError("Published preparation member changed")
    return raw


def prepare(files, claim, manifest_path, binding):
    check_binding(binding)
    if (claim["parent"],claim["node"],claim["device_count"],claim["development_pair_cases"])!=("1087656","f11r4n19",1,3):
        raise ValueError("Fixed dev-set allocation changed")
    body,binding_raw=controller(files["src/run_tum_native_once.py"],binding)
    generated=scorer(files["src/score_vggt_tum_pair.py"])
    pin(body,claim["generated_controller_sha"])
    pin(generated,claim["generated_scorer_sha"])
    pin(binding_raw,claim["training_binding_sha"])
    raw=pin(manifest_path.read_bytes(),DEV_SET_SHA256)
    metadata=json.loads(raw)
    root=manifest_path.resolve().parent
    payload=dict(files)
    payload.update({"src/run_tum_native_once.py":body,"src/score_vggt_tum_pair.py":generated,
                    "src/training-binding.json":binding_raw,"data/pair.json":raw})
    total=len(raw)
    if len(metadata["pairs"])!=3:
        raise ValueError("Fixed development denominator changed")
    for pair in metadata["pairs"]:
        case=pair["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*",case):
            raise ValueError("Invalid development directory")
        directory=root/case
        if directory.is_symlink():
            raise ValueError("Aliased development directory")
        manifest=pin((directory/"pair.json").read_bytes(),pair["manifest_sha256"])
        payload[f"data/{case}/pair.json"]=manifest
        total+=len(manifest)
        for name,digest in json.loads(manifest)["files_SHA256"].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Development member escapes its directory")
            content=pin(path.read_bytes(),digest)
            payload[f"data/{case}/{name}"]=content
            total+=len(content)
    if total!=claim["pair_payload_bytes"] or total>6*1024**2:
        raise ValueError("Frozen development byte budget differs")
    return body,payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication")
    parser.add_argument("ssh_helper",type=Path)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("training_binding",type=Path)
    args=parser.parse_args()
    git=["git","--no-optional-locks","-c","safe.directory="+G.as_posix(),"-C",str(G)]
    def blob(path):
        return subprocess.check_output([*git,"show",args.publication+":"+path])
    remote=subprocess.check_output([*git,"ls-remote","--exit-code","origin","refs/heads/bw1000-experiments"],
                                   timeout=12).decode().split()[0]
    if remote!=args.publication:
        raise ValueError("Published dev claim is not the remote head")
    claim=json.loads(blob("coordination/tasks/"+LABEL+".json"))
    files={"src/"+name:pin(blob(row["path"]),row["sha"]) for name,row in claim["sources"].items()}
    for name in ("training_guard.py","render_controller.py","submit_once.py"):
        pin(Path(__file__).with_name(name).read_bytes(),claim["sources"][name]["sha"])
    # The renderer inherits the already qualified development renderer, from its exact published bytes.
    for name in ("render_tum_dev_controller.py","render_tum_development_set.py"):
        raw=files["src/"+name]
        module=__import__("types").ModuleType(name)
        exec(compile(raw,name,"exec"),module.__dict__)
        sys.modules[name.removesuffix(".py")]=module
    binding=json.loads(args.training_binding.read_bytes())
    body,files=prepare(files,claim,args.manifest,binding)
    pin(args.ssh_helper.read_bytes(),HELPER_SHA256)
    ssh=runpy.run_path(str(args.ssh_helper))["ssh"]
    request=uuid.uuid4().hex
    payload=dict(publication=args.publication,request=request,files={name:dict(
        sha=sha(raw),bytes=len(raw),b64=base64.b64encode(raw).decode("ascii")) for name,raw in files.items()})
    command=("PAYLOAD="+repr(payload)+"\n").encode()+body
    out=G/"local-only"/(LABEL+"-dispatch")
    out.mkdir()
    with (out/"request.json").open("x") as stream:
        json.dump(dict(request=request,publication=args.publication,body_SHA256=sha(command)),stream)
    with (out/"stdout").open("xb") as stdout,(out/"stderr").open("xb") as stderr:
        try:
            proc=subprocess.run([*ssh,"/usr/local/python3.12/bin/python3 -I -S -B -"],input=command,
                                stdout=stdout,stderr=stderr,timeout=700)
        except subprocess.TimeoutExpired:
            proc=None
    print(json.dumps(dict(request=request,ssh_rc=proc.returncode if proc else None,
                          stdout_SHA256=sha((out/"stdout").read_bytes()),stderr_SHA256=sha((out/"stderr").read_bytes()),
                          remote_outcome="UNKNOWN" if proc is None else "inspect original outcome and accounting")))


if __name__=="__main__":
    main()
