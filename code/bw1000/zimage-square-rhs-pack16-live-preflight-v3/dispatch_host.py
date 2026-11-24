from base64 import b64decode
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

def read_rows(args):
    return subprocess.check_output(args,text=True,timeout=7).splitlines()

def ledger(filter_args, cols):
    return [line.split("|") for line in read_rows(["sacct","-nP",*filter_args,"-o",cols])]

def write_new(path, data):
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def encode(data):
    return (json.dumps(data, sort_keys=True)+"\n").encode()

def require_live_standby(step_lines, step, name, node):
    selected = []
    for line in step_lines:
        fields = dict(field.split("=", 1) for field in line.split() if "=" in field)
        if fields.get("StepId") == step:
            selected.append(fields)
    if len(selected) != 1:
        raise ValueError("expected one exact standby step in the controller inventory")
    row = selected[0]
    if (row.get("Name"), row.get("NodeList"), row.get("State")) != (name, node, "RUNNING"):
        raise ValueError("standby label, node, or controller state differs")
    if "gres/dcu=8" not in row.get("TRES", "").split(","):
        raise ValueError("expected an eight-DCU standby step")


def main():
    packet=json.load(sys.stdin)
    request,b=packet["scope"],packet["binding"]
    assert re.fullmatch("[0-9a-f]{32}",request)
    assert b["precheck_request"]==request
    parent,node=b["parent"],b["node"]
    assert (parent,node)==("1103606","f12r2n10")
    label="O01-pack16-deps-json-"+request[:12]
    base=Path("/public/home/aclsylqidf/hygon-inference-refactor-20261014")
    stage,out=base/"incoming"/(label+"-reviewed"),base/"results"/label
    standby=b["standby_step"]
    columns="JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End"
    prior=[r for r in ledger(["-j",standby],columns) if r[0]==standby]
    assert len(prior)==1 and prior[0][1:3]==[b["standby_label"],"RUNNING"]
    assert prior[0][5]==node and "gres/dcu=8" in prior[0][6].split(",")
    stepinfo=read_rows(["scontrol","show","step",parent,"--oneliner"])
    require_live_standby(stepinfo, standby, b["standby_label"], node)
    assert not stage.exists() and not out.exists()
    assert not any(r[1]==label for r in ledger(["-j",parent,"-S","2026-11-24T00:00:00"],columns) if len(r)==9)
    files={name:b64decode(raw,validate=True) for name,raw in packet["files"].items()}
    assert set(files)=={"precheck.py","SCOPED_FILES.json","worker.py","candidate.py","baseline.py","contract.py","hipprof_trace.py"}
    assert sha256(files["precheck.py"]).hexdigest()==packet["compiled_python_source_sha256"]
    manifest=files["SCOPED_FILES.json"]
    assert sha256(manifest).hexdigest()==packet["frozen_runtime_manifest_sha256"]
    pins=json.loads(manifest)["files"]
    assert set(pins)==set(files)-{"precheck.py","SCOPED_FILES.json"}
    for name,spec in pins.items():
        assert len(files[name])==spec["bytes"] and sha256(files[name]).hexdigest()==spec["SHA256"]
    os.umask(0o077)
    out.mkdir()
    stage.mkdir()
    (stage/"runtime").mkdir()
    source=stage/"precheck.py"
    for name,raw in files.items():
        write_new(source if name=="precheck.py" else stage/"runtime"/name,raw)
    ns={"__name__":"published_o01_precheck"}
    exec(compile(files["precheck.py"],str(source),"exec"),ns)
    assert (ns["STAGE"],ns["OUT"],ns["REQUEST"])==(stage,out,request)
    argv=ns["srun_argv"](source,packet["compiled_python_source_sha256"])
    write_new(out/"submission-intent.json",encode({"scope":request,"label":label,"standby":prior[0],"claim_sha256":packet["claim_sha256"],"argv":argv}))
    try:
        child=subprocess.run(argv,capture_output=True,timeout=73,check=False)
        rc=child.returncode
        rawout,rawerr=child.stdout,child.stderr
    except subprocess.TimeoutExpired as error:
        rc=None
        rawout,rawerr=error.stdout or b"",error.stderr or b""
    write_new(out/"srun.stdout",rawout)
    write_new(out/"srun.stderr",rawerr)
    rows=[]
    for _ in range(6):
        rows=[r for r in ledger(["-j",parent,"-S","2026-11-24T00:00:00"],columns) if len(r)==9 and r[1]==label]
        if len(rows)==1 and rows[0][8]!="Unknown":
            break
        time.sleep(.5)
    step_result=None
    step_path=out/"step.stdout"
    if step_path.is_file():
        with step_path.open("rb") as fd:
            raw=fd.read(16385)
        if 0<len(raw)<=16384:
            try:
                result=json.loads(raw)
            except (ValueError,UnicodeError):
                result=None
            if isinstance(result,dict) and result.get("request")==request:
                step_result=result
    final={"request":request,"scope":request,"srun_rc":rc,"accounting":rows,
           "standby_current":[r for r in ledger(["-j",standby],columns) if r[0]==standby],
           "step_result":step_result,"srun_stdout_sha256":sha256(rawout).hexdigest(),
           "srun_stderr_sha256":sha256(rawerr).hexdigest(),"GPU_GO":False}
    write_new(out/"launch-and-accounting.json",encode(final))
    print(json.dumps(final,sort_keys=True))


if __name__ == "__main__":
    main()
