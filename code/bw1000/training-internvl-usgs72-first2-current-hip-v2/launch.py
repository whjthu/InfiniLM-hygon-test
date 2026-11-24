"""Stage and hand the exact owned standby to the current InternVL engineering run."""

import argparse
from base64 import b64encode
import gzip
import json
from pathlib import Path
import re
import shlex
import subprocess
from types import ModuleType

from render import HERE, ROOT_REPO, canonical, digest, render


HOST = r'''
from base64 import b64decode
import fcntl, gzip, json, os, re, subprocess, sys, time
from hashlib import sha256
from pathlib import Path

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def digest(raw):
    return sha256(raw).hexdigest()

def document(path, pin=None, cap=160000):
    need(path.resolve(strict=True)==path and path.is_file(), 'missing exact input')
    with path.open('rb') as f:
        data=f.read(cap+1)
    need(0<len(data)<=cap and (pin is None or digest(data)==pin), 'original evidence identity differs')
    return json.loads(data)

def save(path, obj):
    with path.open('x') as stream:
        json.dump(obj,stream,sort_keys=True,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=8).splitlines()

raw=sys.stdin.buffer.read(270001)
need(0<len(raw)<=270000,'bounded source packet required')
packet=json.loads(gzip.decompress(b64decode(raw,validate=True)))
files={name:b64decode(record,validate=True) for name,record in packet['files'].items()}
manifest=json.loads(files['SOURCE_FILES.json'])
claim=json.loads(files['CLAIM.json'])
request=claim['request']
scope=claim['placement']
parent,node=scope['parent'],scope['node']
need(re.fullmatch(r'[a-f0-9]{40}',packet['publication']) and re.fullmatch(r'[a-f0-9]{32}',request)
     and (parent,node,scope['dcu'],scope['cpus'],scope['mem_mib'],scope['port'])==('1103606','f13r1n14',8,64,224000,29967)
     and claim['root_GO'] is claim['GPU_GO'] is True and claim['formal17_qualified'] is claim['replay_allowed'] is False,
     'unsafe launch binding')
need(all(Path(name).name==name for name in files) and set(files)-{'SOURCE_FILES.json'}==set(manifest['files'])
     and manifest['request']==request and all(len(files[name])==rec['bytes'] and digest(files[name])==rec['SHA256']
        for name,rec in manifest['files'].items()),'rendered package source identity differs')
results=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
stage=results/('training-internvl-usgs72-v2-stage-'+request)
target=results/claim['label']
ctrl=results/(claim['label']+'-controller')
prior=claim['previous']; prior_step=prior['previous']; prior_request=prior['request']
prior_label='training80-dcu-standby-'+prior_request[:12]+'-'+node
prior_dir=results/('training80-dcu-standby-'+prior_request)/node
resources='cpu=8,gres/dcu=8,mem=16G,node=1'
old_labels={claim['label']+'-'+s for s in ('pre','gpu','post')}
columns='JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'

def account(job):
    lines=[x.split('|') for x in query('sacct','-nP','-j',job,'-o',columns)]
    need(all(len(row)==9 for row in lines),'unexpected accounting format')
    return lines

def fields(line):
    return dict(x.split('=',1) for x in line.split() if '=' in x)

def observe():
    row,=map(fields,query('scontrol','show','job','-o',parent))
    need(row['JobId']==parent and row['JobState']=='RUNNING' and row['UserId'].endswith(f'({os.getuid()})')
         and 'gres/dcu=320' in row['AllocTRES'].split(',') and node in query('scontrol','show','hostnames',row['NodeList']),
         'parent ownership changed')
    cache={}
    def members(spec):
        need(spec and spec not in ('(null)','N/A','None','Unknown'),'unresolved node membership')
        if spec not in cache:
            cache[spec]=[spec] if re.fullmatch(r'f\d+r\d+n\d+',spec) else query('scontrol','show','hostnames',spec)
            need(cache[spec] and len(cache[spec])==len(set(cache[spec])) and all(re.fullmatch(r'f\d+r\d+n\d+',x) for x in cache[spec]),
                 'ambiguous host list')
        return cache[spec]
    active={}
    for step in map(fields,query('scontrol','show','step',parent,'--oneliner')):
        key=step['StepId']
        if key in {parent+'.batch',parent+'.extern'}:
            continue
        need(re.fullmatch(parent+r'\.\d+',key),'unknown controller step')
        placements=members(step['NodeList'])
        if node in placements:
            need(placements==[node] and key not in active,'ambiguous scoped controller step')
            active[key]=[step['Name'],step['State'],step['TRES']]
    queued=set()
    for line in query('squeue','--steps','-h','-j',parent,'-o','%i|%N'):
        step,hosts=line.split('|')
        if step in {parent+'.batch',parent+'.extern'}:
            continue
        need(re.fullmatch(parent+r'\.\d+',step),'unknown queue step')
        placements=members(hosts)
        if node in placements:
            need(placements==[node],'ambiguous scoped queue step')
            queued.add(step)
    need(set(active)<={prior_step} and queued<={prior_step},'another step uses the training node')
    [original]=[row for row in account(prior_step) if row[0]==prior_step]
    need(original[1]==prior_label and original[5:7]==[node,resources],'standby accounting identity changed')
    return original,active,queued

def zero_card_port_probe():
    name='training-internvl-v2-port-'+request[:12]
    need(not any(r[1]==name for r in account(parent)), 'port probe already exists')
    program='import os,socket; assert os.environ.get("SLURM_STEP_GPUS") in (None,"","none","N/A","(null)"); s=socket.socket(); s.bind(("127.0.0.1",29967)); s.close(); print(os.environ["SLURM_JOB_ID"]+"."+os.environ["SLURM_STEP_ID"])'
    r=subprocess.run(['srun','--jobid='+parent,'--job-name='+name,'--exact','--overlap','--immediate=5','--nodes=1','--ntasks=1',
        '--nodelist='+node,'--cpus-per-task=1','--mem=256M','--gres=none','--time=00:01:00',sys.executable,'-I','-S','-B','-c',program],
        check=True,capture_output=True,text=True,timeout=15)
    step=r.stdout.strip()
    need(re.fullmatch(parent+r'\.\d+',step),'unbound port probe')
    for _ in range(8):
        rows=[row for row in account(step) if row[0]==step]
        if len(rows)==1 and rows[0][1:4]==[name,'COMPLETED','0:0'] and rows[0][5:7]==[node,'cpu=1,mem=256M,node=1']:
            return rows[0]
        time.sleep(0.4)
    raise ValueError('port probe accounting not terminal')

os.umask(0o077)
need(not target.exists() and not ctrl.exists() and not any(row[1] in old_labels for row in account(parent)),
     'new training scope has an existing result or label')
stage.mkdir(mode=0o700)
for name, data in files.items():
    with (stage/name).open('xb') as stream:
        stream.write(data)
save(stage/'stage-receipt.json',dict(request=request,publication=packet['publication'],source_files=len(files),
     manifest_SHA256=digest(files['SOURCE_FILES.json']),data_written=False))
sys.path.insert(0,str(stage))
from node import actual_model_plan, bound_source
import native
bound_source(digest(files['CLAIM.json']),digest(files['SOURCE_FILES.json']))
asset, plan, view = actual_model_plan(claim['exports_SHA256'])
asset.verify_originals(plan)
need(len(view)==20 and len(native.binding()['trainable_manifest'])==504,'restored source asset preflight differs')
expected=document(prior_dir/'expected.json',prior['expected_SHA256'])
worker=document(prior_dir/'worker.json',prior['worker_SHA256'])
need(expected['request']==prior_request and expected['parent']==parent and expected['node']==node and expected['label']==prior_label
     and expected['seconds']==prior['worker_seconds'] and worker['node']==node and worker['step']==prior_step and worker['synthetic'] is True
     and (worker['worker_pid'],worker['epoch'])==(prior['worker_PID'],prior['worker_start_epoch']) and sorted(worker['pci'])==expected['pci']
     and len(set(expected['pci']))==8,'standby worker and physical source binding differ')
port=zero_card_port_probe()
save(stage/'port.json',port)
with (results/'.training80-dcu-standby.lock').open('ab') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original,active,queued=observe()
    if original[2:4]==['RUNNING','0:0']:
        need(active=={prior_step:[prior_label,'RUNNING',resources]} and not (prior_dir/'terminal.json').exists(),
             'predecessor no longer eligible for signal')
        for index,pci in enumerate(worker['pci']):
            card=document(prior_dir/f'card-{index}.json')
            need(card['pci']==pci and card['synthetic'] is card['active'] is card['ready'] is True and card['calls']>0
                 and 0<=time.time()-card['epoch']<25,'standby card receipt is stale or not active')
        save(stage/'signal-intent.json',dict(request=request,target=prior_step,original=original))
        signal=subprocess.run(['scancel','--signal=TERM',prior_step],stdin=subprocess.DEVNULL,capture_output=True,timeout=8,check=False)
        save(stage/'signal-result.json',dict(rc=signal.returncode,stderr_SHA256=digest(signal.stderr)))
    end=time.monotonic()+82
    while True:
        original,active,queued=observe()
        if not active and not queued and original[2] in ('COMPLETED','CANCELLED') and original[3]=='0:0' and original[-1]!='Unknown':
            term=document(prior_dir/'terminal.json')
            need(term['state']=='WORKER_RETURNED' and term['step']==prior_step and term['node']==node,'standby did not return normally')
            break
        need(time.monotonic()<end,'original standby terminal unavailable')
        time.sleep(1.5)
    save(stage/'standby-released.json',dict(original=original,terminal=term,request=request))
    args=[sys.executable,'-B',str(stage/'control_once.py'),'--publication',packet['publication'],
          '--claim-sha256',digest(files['CLAIM.json']),'--manifest-sha256',digest(files['SOURCE_FILES.json']),
          '--exports-sha256',claim['exports_SHA256']]
    launch=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,timeout=45,check=False)
    need(launch.returncode==0 and 0<len(launch.stdout)<=8192,'training controller submission not confirmed')
    launched=json.loads(launch.stdout)
    need(launched['request']==request and launched['state']=='CONTROLLER_STARTED','controller identity differs')
    save(stage/'launch-controller.json',launched)
print(json.dumps(dict(request=request,publication=packet['publication'],stage_manifest_SHA256=digest(files['SOURCE_FILES.json']),
     predecessor_terminal=original,port_probe=port,controller=launched),sort_keys=True))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    args = p.parse_args()
    compile(HOST, "<handoff controller>", "exec")
    claim_raw = (HERE / "CLAIM.json").read_bytes()
    claim = json.loads(claim_raw)
    for proof in claim["proof"].values():
        if digest((ROOT_REPO / proof["path"]).read_bytes()) != proof["sha256"]:
            raise ValueError("pinned public proof differs")
    files = render(claim)
    if claim_raw != files["CLAIM.json"]:
        raise ValueError("claim must use deterministic bytes")
    if args.check:
        print(json.dumps({"request": claim["request"], "rendered": len(files),
                          "manifest_SHA256": digest(files["SOURCE_FILES.json"]), "remote_calls": 0}))
        return 0
    if not re.fullmatch(r"[a-f0-9]{40}", args.publication):
        raise ValueError("exact public commit required")
    for f in (HERE / "CLAIM.json", HERE / "render.py", HERE / "launch.py"):
        actual = subprocess.check_output(["git", "-c", f"safe.directory={ROOT_REPO.as_posix()}", "show",
                                         args.publication + ":" + f.relative_to(ROOT_REPO).as_posix()], cwd=ROOT_REPO)
        if actual != f.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("worktree and publication differ")
    import sys

    sys.path.insert(0, str(ROOT_REPO / "tools"))
    import report_training80_backlog as backlog

    local = ROOT_REPO / "local-only"
    holds = backlog.read_queue_holds(local / "tuning/queue.sqlite3")
    rows = json.loads((ROOT_REPO / "coordination/training/training80-backlog-v1.json").read_bytes())["items"]
    names = set(backlog.allocation(backlog.TRAINING80_LEDGER.read_bytes())[2])
    names.update(node for row in rows for node in row["placement"])
    original, _, _ = backlog.read_original_holds(ROOT_REPO, rows, names, holds)
    node = claim["placement"]["node"]
    if any(node in row["nodes"] for row in [*holds, *original.values()]):
        raise ValueError("another local job holds the training node")
    helper = local / "run-minicpm-sdpa-target-asset-preflight-original.py"
    raw = helper.read_bytes()
    if digest(raw) != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("existing strict project transport differs")
    transport = ModuleType("existing_transport")
    exec(compile(raw, str(helper), "exec"), vars(transport))
    payload = b64encode(gzip.compress(canonical({"publication": args.publication,
                        "files": {name: b64encode(data).decode() for name, data in files.items()}}), mtime=0))
    argv = [*transport.ssh(ROOT_REPO.parent), shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", HOST])]
    request = claim["request"]
    with (local / f"training-internvl-v2-{request}-send.json").open("xb") as f:
        f.write(canonical({"request": request, "payload_SHA256": digest(payload), "publication": args.publication}))
    try:
        result = subprocess.run(argv, input=payload, capture_output=True, timeout=145, check=False)
        rc, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        rc, stdout, stderr = None, e.stdout or b"", e.stderr or b""
    try:
        observed = json.loads(stdout) if rc == 0 and 0 < len(stdout) <= 16384 else None
    except (ValueError, UnicodeError):
        observed = None
    record = {"rc": rc, "observed": observed, "stdout_SHA256": digest(stdout), "stderr_SHA256": digest(stderr),
              "request": request, "publication": args.publication, "replay_allowed": False}
    if observed is None:
        record["failure_class"] = next((x.decode() for x in (b"Permission denied", b"Timeout", b"CalledProcessError",
                                        b"ValueError", b"FileNotFoundError", b"KeyError") if x in stderr), None)
    out = local / f"training-internvl-v2-{request}-result.json"
    data = canonical(record)
    with out.open("xb") as f:
        f.write(data)
    print(json.dumps({"file": out.name, "sha256": digest(data), **record}))
    return int(observed is None)


if __name__ == "__main__":
    raise SystemExit(main())
