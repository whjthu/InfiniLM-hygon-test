"""Stage one published pack16 package and observe its exact zero-DCU dependency step."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import subprocess
from uuid import uuid4

from run_hygon_bound_payload import document, send


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "local-only"
REQUEST = "30f9d5dd39bc49c1b8d52312ba2951d6"
CANDIDATE_COMMIT = "c4da359b16d2eea35300547f42e9967dd3199b86"
PRE = "code/bw1000/zimage-square-rhs-pack16-target-preflight-v1/"
CANDIDATE = "code/bw1000/zimage-square-rhs-pack16-v1/"

HOST = r'''
from base64 import b64decode
from hashlib import sha256
import fcntl,json,os,runpy,subprocess,sys,time
from pathlib import Path

def need(ok,reason):
    if not ok: raise ValueError(reason)

def read(command):
    raw=subprocess.check_output(command,timeout=8)
    need(len(raw)<1048576,'SCHEDULER_RESPONSE_CAP')
    return raw.decode().splitlines()

def fields(line):
    return dict(field.split('=',1) for field in line.split() if '=' in field)

def save(path,raw):
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())

def encoded(x):
    return (json.dumps(x,sort_keys=True,allow_nan=False)+'\n').encode()

packet=json.loads(sys.stdin.buffer.read(200000))
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
request,stage_label='30f9d5dd39bc49c1b8d52312ba2951d6','O01-pack16-deps-30f9d5dd39bc'
parent,node='1103606','f12r2n10'
stage=root/'incoming'/(stage_label+'-reviewed')
out=root/'results'/stage_label
files={name:b64decode(raw,validate=True) for name,raw in packet['files'].items()}
expected={'precheck.py', 'SCOPED_FILES.json', 'baseline.py', 'candidate.py', 'contract.py', 'hipprof_trace.py', 'worker.py'}
need(packet['request']==request and set(files)==expected,'DEPLOYED_SCOPE')
source_pin='2d6152e99c3401dbd05c2c63d64212807df855e98f01bfe030af08e6e51928a0'
need(sha256(files['precheck.py']).hexdigest()==source_pin,'PRECHECK_SOURCE')
manifest=files['SCOPED_FILES.json']
need(sha256(manifest).hexdigest()=='999e985cfe5066e6520c6a713d45d0240851a07a04b712246e39d0bce006b74f','MANIFEST_SHA')
members=json.loads(manifest)['files']
need(set(members)==expected-{'precheck.py','SCOPED_FILES.json'}
     and all(len(files[n])==m['bytes'] and sha256(files[n]).hexdigest()==m['SHA256'] for n,m in members.items()),'MANIFEST_MEMBERS')

def accounting():
    args=['sacct','-nP','-j',parent,'-S','2026-11-24T15:00:00',
          '-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES']
    return [row for line in read(args) if len(row:=line.split('|'))==7 and row[1]==stage_label]

os.umask(0o077)
with (root/'results'/'.training80-dcu-standby.lock').open('ab') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    [job]=list(map(fields,read(['scontrol','-o','show','job',parent])))
    need(job.get('JobState')=='RUNNING' and job['UserId'].endswith(f'({os.getuid()})')
         and node in read(['scontrol','show','hostnames',job['NodeList']]),'OWNED_PARENT_NODE')
    steps=list(map(fields,read(['scontrol','-o','show','step',parent])))
    own=[row for row in steps if row.get('StepId')=='1103606.2480']
    need(len(own)==1 and own[0].get('State')=='RUNNING' and own[0].get('NodeList')==node
         and own[0].get('Name')=='training80-dcu-standby-9454531ca7ef-f12r2n10'
         and 'gres/dcu=8' in own[0].get('TRES','').split(','),'ORIGINAL_STANDBY')
    need(not accounting() and not os.path.lexists(stage) and not os.path.lexists(out),'ORIGINAL_PRECHECK_EXISTS')
    stage.mkdir(mode=0o700)
    out.mkdir(mode=0o700)
    (stage/'runtime').mkdir(mode=0o700)
    for name,raw in files.items():
        save(stage/'precheck.py' if name=='precheck.py' else stage/'runtime'/name,raw)
    runner=runpy.run_path(str(stage/'precheck.py'))['srun_argv']
    argv=runner(stage/'precheck.py',source_pin)
    need('--gres=none' in argv and '--nodelist='+node in argv and '--time=00:01:00' in argv,'ZERO_GRES_COMMAND')
    save(out/'intent.json',encoded({'request':request,'publication':packet['publication'],'source_SHA256':source_pin,
                                   'GRES':0,'epoch':time.time(),'standby':own[0]['StepId'],'replay_allowed':False}))
    process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

try:
    rc=process.wait(timeout=70)
except subprocess.TimeoutExpired:
    rc=None
rows=accounting()
answer=out/'step.stdout'
observed=None
if answer.is_file():
    with answer.open('rb') as f: raw=f.read(8193)
    if 0<len(raw)<=8192:
        observed=json.loads(raw)
        observed={'step':observed.get('step'),'precheck_passed':observed.get('precheck_passed'),
                  'gate':observed.get('gate'),'same_standby_unchanged':observed.get('same_standby_unchanged'),
                  'step_stdout_SHA256':sha256(raw).hexdigest()}
result={'request':request,'precheck_child_rc':rc,'accounting':rows,'observed':observed,'requested_GRES':0}
save(out/'controller-result.json',encoded(result))
print(json.dumps(result,sort_keys=True))
'''


def git_show(commit, path):
    return subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}", "show", f"{commit}:{path}"])


def prepare():
    renderer = runpy.run_path(str(REPO / CANDIDATE / "prepare.py"))
    files, manifest = renderer["build"]()
    files["SCOPED_FILES.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    files["precheck.py"] = (REPO / PRE / "precheck.py").read_bytes()
    for name, actual in files.items():
        if name in ("candidate.py", "worker.py"):
            assert actual == git_show(CANDIDATE_COMMIT, CANDIDATE + name)
    assert sha256(files["SCOPED_FILES.json"]).hexdigest() == "999e985cfe5066e6520c6a713d45d0240851a07a04b712246e39d0bce006b74f"
    assert sha256(files["precheck.py"]).hexdigest() == "2d6152e99c3401dbd05c2c63d64212807df855e98f01bfe030af08e6e51928a0"
    compile(HOST, "<zero-DCU launcher>", "exec")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    args = parser.parse_args()
    files = prepare()
    if args.check:
        print(json.dumps({"request": REQUEST, "source_files": sorted(files), "remote_calls": 0}))
        return
    if not re.fullmatch("[0-9a-f]{40}", args.publication):
        raise ValueError("expected exact published source commit")
    subprocess.run(["git", "-c", f"safe.directory={REPO.as_posix()}", "merge-base", "--is-ancestor",
                    args.publication, "refs/remotes/origin/bw1000-experiments"], cwd=REPO, check=True)
    for path in (PRE + "precheck.py", PRE + "CLAIM.json", "tools/run_zimage_pack16_preflight_once.py"):
        if (REPO / path).read_bytes().replace(b"\r\n", b"\n") != git_show(args.publication, path):
            raise ValueError("published source differs")
    helper = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
    if sha256(helper.read_bytes()).hexdigest() != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("existing strict connection helper differs")
    data = document({"request": REQUEST, "publication": args.publication,
                     "files": {name: b64encode(raw).decode() for name, raw in files.items()}})
    result = send(runpy.run_path(str(helper))["ssh"](REPO.parent), request=REQUEST, program=HOST, data=data,
                  send_token=LOCAL / f"o01-pack16-{REQUEST}-send.json", outcome_seconds=90)
    path = LOCAL / f"o01-pack16-{REQUEST}-{uuid4().hex}.json"
    raw = document(result)
    with path.open("xb") as f:
        f.write(raw)
    print(json.dumps({"file": path.name, "SHA256": sha256(raw).hexdigest(), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
