"""Render the pinned observed-v2 workflow with the qualified local scratch and owner gates."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import owner_gate as gate


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
BASE=HERE.with_name("training-internvl-gqa56-first2-native-observed-v2")/"render.py"
BASE_SOURCE_SHA="1bb8ac89e51afd3e5891da10577fd80004019d6baaab169a1dd783ce7b679660"
BASE_MANIFEST_SHA="8b57733cff6084ae62752b3459f0afecece85867d3bae660d73fc70455e895eb"
SCRATCH_SOURCE_SHA="ce89f318822dca1e2d8ef02011f517751420adbb4aaf4ec132ec4dcc4bcbb940"
SCRATCH_RESULT_SHA="cc7402fda779ad85a3eab770aa212fa384e026bf03772c886b6c49e70ad4f625"


def sha(raw):
    return sha256(raw).hexdigest()


def encode(data):
    return (json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def edit(files,name,changes):
    text=files[name].decode()
    for old,new in changes:
        gate.need(text.count(old)==1,f"EXACT_{name}_SOURCE_ANCHOR: {old[:55]}")
        text=text.replace(old,new)
    files[name]=text.encode()


def adopt_scratch(files):
    scratch=(REPO/"tools/render_internvl_gqa64_continuous25.py").read_bytes()
    result=(REPO/"results/bw1000/2026-11-27/specialized-internvl-GQA25-4385-continuous-pass-cp25-and-independent-4388-clean.json").read_bytes()
    gate.need(sha(scratch)==SCRATCH_SOURCE_SHA and sha(result)==SCRATCH_RESULT_SHA,"EXACT_ACTUAL_4385_SCRATCH_PROVENANCE")
    edit(files,"node.py",[
        ("import sys", "import sys\nimport tempfile"),
        ("def command(plan,epoch,plan_sha):", "def command(plan,epoch,plan_sha,scratch):"),
        ('f"{RUN / \'tmp\'}:/tmp:rw"', 'f"{scratch}:/tmp:rw"'),
        ('for name in ("home","cache","tmp","capture"):', 'for name in ("home","capture"):'),
        ('''        with (RUN/"container.log").open("xb") as out:
            result=subprocess.run(command(plan,args.epoch,plan_sha),stdin=subprocess.DEVNULL,stdout=out,
                                  stderr=subprocess.STDOUT,timeout=seconds_left(args.epoch,SLURM)-20)
        rc=result.returncode''',
         '''        fs=subprocess.check_output(
            ["findmnt","--noheadings","-o","FSTYPE","--target","/tmp"],text=True,timeout=5).strip()
        need(fs=="ext4","Qualified node-local temporary filesystem differs")
        disk=os.statvfs("/tmp")
        need(disk.f_bavail*disk.f_frsize>=2**31,"Node-local temporary filesystem is full")
        with tempfile.TemporaryDirectory(prefix="internvl-gqa56-",dir="/tmp") as temp:
            st=Path(temp).stat()
            write_once(RUN/"local-scratch.json",dict(request=REQUEST,device=st.st_dev,inode=st.st_ino,fs=fs))
            with (RUN/"container.log").open("xb") as out:
                result=subprocess.run(command(plan,args.epoch,plan_sha,temp),stdin=subprocess.DEVNULL,stdout=out,
                                      stderr=subprocess.STDOUT,timeout=seconds_left(args.epoch,SLURM)-20)
            rc=result.returncode
        need(not Path(temp).exists(),"Owned node-local temporary filesystem contents remain")
        write_once(RUN/"local-scratch-clean.json",dict(request=REQUEST,removed=True))'''),
    ])
    edit(files,"native.py",[
        ('scratch = {**private_tmp(RUN / "tmp"), **private_unix_socket()}',
         '''scratch = json.loads(pin(RUN/"local-scratch.json"))
    tmp=Path("/tmp").stat()
    need(scratch["request"]==REQUEST and scratch["fs"]=="ext4"
         and (scratch["device"],scratch["inode"])==(tmp.st_dev,tmp.st_ino),"Node-local scratch bind differs")
    scratch.update(private_unix_socket())'''),
        ('HF_HOME=str(RUN/"cache"),T03_FIRST2_EPOCH=str(epoch)',
         'HF_HOME="/tmp/hf",HF_DATASETS_CACHE="/tmp/datasets",T03_FIRST2_EPOCH=str(epoch)'),
    ])


def adopt_owner(files):
    files["owner_gate.py"]=(HERE/"owner_gate.py").read_bytes()
    edit(files,"node.py",[
        ('claim["GPU_GO"] is claim["root_GO"] is True','claim["GPU_GO"] is claim["root_GO"] is False'),
        ('"Root adoption of the single engineering diagnostic is required"','"The candidate must require the separate root authority"'),
        ('    args=p.parse_args()','''    p.add_argument("--authority-sha256",required=True)
    p.add_argument("--publication",required=True)
    args=p.parse_args()'''),
        ('    claim=bound_source(args.claim_sha256,args.manifest_sha256)',
         '''    from time import time
    from owner_gate import load as load_owner
    load_owner(HERE,args.authority_sha256,args.publication,args.claim_sha256,args.manifest_sha256,pin,time())
    claim=bound_source(args.claim_sha256,args.manifest_sha256)'''),
    ])
    edit(files,"control_once.py",[
        ('raise SystemExit("DISABLED_UNTIL_FRESH_OWNER_AND_ZERO_GRES_QUALIFICATION")\n',''),
        ("PARENT, NODE, PORT = '1103606', 'f13r1n14', 29967",
         f"PARENT, NODE, PORT = '1103606', 'f13r1n14', {gate.PLACEMENT['port']}"),
        ('def run(launch):','''def validate_launch(launch,verify_accounting=False):
    from owner_gate import CPU_ROW, load, verify_cpu_accounting
    from scope import pin
    need(launch["request"]==REQUEST,"Exact new controller request required")
    load(HERE,launch["owner_SHA256"],launch["publication"],launch["claim_SHA256"],launch["manifest_SHA256"],pin,time.time())
    if verify_accounting:
        return verify_cpu_accounting()
    need(launch["zero_gres_accounting"]==CPU_ROW,"Exact prior .4701 accounting is absent")


def run(launch):'''),
        ('    epoch = launch[\'epoch\']','''    validate_launch(launch)
    epoch = launch['epoch']
    put(OUT/"controller-owned.json",dict(request=REQUEST,epoch=epoch,pid=os.getpid(),owner_SHA256=launch["owner_SHA256"]))'''),
        ('''        placement()
        kwargs = ['--epoch',str(epoch),'--claim-sha256',launch['claim_SHA256'],'--manifest-sha256',launch['manifest_SHA256'],''',
         '''        placement()
        validate_launch(launch)
        kwargs = ['--epoch',str(epoch),'--claim-sha256',launch['claim_SHA256'],'--manifest-sha256',launch['manifest_SHA256'],
                  '--authority-sha256',launch['owner_SHA256'],'--publication',launch['publication'],'''),
        ('''    OUT.mkdir(mode=0o700)
    doc = dict(request=REQUEST,publication=args.publication,epoch=time.time(),claim_SHA256=args.claim_sha256,
               manifest_SHA256=args.manifest_sha256,exports_SHA256=args.exports_sha256)
    put(OUT/'intent.json',doc)''',
         '''    doc = dict(request=REQUEST,publication=args.publication,epoch=time.time(),claim_SHA256=args.claim_sha256,
               manifest_SHA256=args.manifest_sha256,exports_SHA256=args.exports_sha256,owner_SHA256=args.authority_sha256)
    doc["zero_gres_accounting"]=validate_launch(doc,verify_accounting=True)
    OUT.mkdir(mode=0o700)
    put(OUT/'intent.json',doc)
    intent_sha=sha256((OUT/'intent.json').read_bytes()).hexdigest()'''),
        ("[sys.executable,'-B',str(Path(__file__).resolve()),'--run']",
         "[sys.executable,'-B',str(Path(__file__).resolve()),'--run','--intent-sha256',intent_sha]"),
        ("('claim-sha256','manifest-sha256','exports-sha256','publication'):",
         "('claim-sha256','manifest-sha256','exports-sha256','publication','authority-sha256','intent-sha256'):"),
        ("        run(read(OUT/'intent.json'))",'''        from scope import pin
        need(isinstance(args.intent_sha256,str) and HEX.fullmatch(args.intent_sha256),"Exact durable controller intent digest required")
        run(json.loads(pin(OUT/'intent.json',expected=args.intent_sha256)))'''),
    ])


def render():
    raw=BASE.read_bytes()
    gate.need(sha(raw)==BASE_SOURCE_SHA,"EXACT_OBSERVED_V2_RENDERER")
    spec=importlib.util.spec_from_file_location("internvl_observed_v2",BASE)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0,str(BASE.parent))
    exec(compile(raw,str(BASE),"exec"),module.__dict__)
    files=module.render()
    gate.need(sha(files.pop("SOURCE_FILES.json"))==BASE_MANIFEST_SHA,"EXACT_OBSERVED_V2_MANIFEST")
    for name in ("scope.py","control_once.py"):
        edit(files,name,[(module.REQUEST,gate.REQUEST),(module.LABEL,gate.LABEL)])
    edit(files,"scope.py",[(str(module.PORT),str(gate.PLACEMENT["port"]))])
    adopt_scratch(files)
    adopt_owner(files)
    claim=json.loads(files["CLAIM.json"])
    claim.update(request=gate.REQUEST,label=gate.LABEL,placement=gate.PLACEMENT,
                 schema="INTERNVL_GQA56_FIRST2_OBSERVED_HIP_V3",state="EXPLICIT_ROOT_OWNER_AND_4701_QUALIFICATION_REQUIRED",
                 root_GO=False,GPU_GO=False,prior_zero_gres_step=gate.CPU_ROW[0],prior_zero_gres_result_SHA256=gate.CPU_SHA,
                 prior_zero_gres_archive_SHA256=gate.CPU_ARCHIVE_SHA,node_scratch_origin="1103606.4385",
                 node_scratch_source_SHA256=SCRATCH_SOURCE_SHA,node_scratch_result_SHA256=SCRATCH_RESULT_SHA)
    files["CLAIM.json"]=encode(claim)
    for name,data in files.items():
        if name.endswith(".py"):
            compile(data,name,"exec")
    files["SOURCE_FILES.json"]=encode(dict(request=gate.REQUEST,files={
        name:dict(bytes=len(raw),SHA256=sha(raw)) for name,raw in files.items()}))
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    files=render()
    if args.output is not None:
        args.output.mkdir()
        for name,raw in files.items():
            with (args.output/name).open("xb") as stream:
                stream.write(raw)
    summary=dict(request=gate.REQUEST,label=gate.LABEL,files=len(files),bytes=sum(map(len,files.values())),
                 manifest_SHA256=sha(files["SOURCE_FILES.json"]),target_launched=False)
    print(encode(summary).decode(),end="")


if __name__=="__main__":
    main()
