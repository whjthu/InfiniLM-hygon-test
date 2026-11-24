"""Publish-bound one-shot target CPU check of the existing InternVL domain assets."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
import subprocess
from types import ModuleType


HERE = Path(__file__).resolve().parent
G = HERE.parents[2]
LOCAL = G / "local-only"
ORIGINAL = HERE.parent / "training-internvl-usgs72-native-first2-hip-v1"
FILES = {"scope.py", "domain_checks.py", "owned_processes.py", "native.py", "NATIVE_BINDING.json", "TARGET_MATH_SOURCES.json"}
HOST = r'''
from base64 import b64decode
from hashlib import sha256
import json, os, re, subprocess, sys, time
from pathlib import Path

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

raw = sys.stdin.buffer.read(310001)
require(0 < len(raw) <= 310000, 'source packet extent')
packet = json.loads(raw)
claim = json.loads(b64decode(packet['claim'], validate=True))
request = claim['request']
require(re.fullmatch(r'[a-f0-9]{32}', request) and re.fullmatch(r'[a-f0-9]{40}', packet['publication']), 'request identity')
parent, node = claim['parent'], claim['node']
require((parent, node, claim['cpus'], claim['memory_MiB'], claim['GPU'], claim['slurm_seconds']) ==
        ('1103606', 'f13r1n14', 2, 4096, 0, 120), 'resource scope differs')
files = {name: b64decode(value, validate=True) for name, value in packet['files'].items()}
wanted = {'scope.py','domain_checks.py','owned_processes.py','native.py','NATIVE_BINDING.json','TARGET_MATH_SOURCES.json','check_assets.py'}
require(set(files) == wanted and {name:sha256(data).hexdigest() for name,data in files.items()} == claim['source_SHA256'], 'source pins differ')

def query(*argv):
    return subprocess.check_output(argv, text=True, timeout=8).splitlines()

rows = query('scontrol','show','job','-o',parent)
require(len(rows) == 1, 'ambiguous parent')
job = dict(item.split('=',1) for item in rows[0].split() if '=' in item)
require(job['JobId'] == parent and job['JobState'] == 'RUNNING' and job['UserId'].endswith(f'({os.getuid()})')
        and 'gres/dcu=320' in job['AllocTRES'].split(',') and node in query('scontrol','show','hostnames',job['NodeList']), 'parent membership differs')
label = 'training-internvl-usgs72-cpu-' + request[:12]
columns = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'

def accounting(job_id):
    return [parts for line in query('sacct','-nP','-j',job_id,'-o',columns)
            if len(parts:=line.split('|'))==9 and parts[1]==label]

require(not accounting(parent), 'request label already visible')
root = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
results = root / 'results'
out = results / ('training-internvl-usgs72-current-cpu-' + request)
os.umask(0o077)
out.mkdir()

def save(name, data):
    with (out / name).open('xb') as stream:
        stream.write(data)

def encode(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()

epoch = time.time()
save('intent.json', encode(dict(request=request, publication=packet['publication'], node=node, parent=parent,
                                label=label, epoch=epoch, deadline=epoch+120, replay_allowed=False)))
for name, data in files.items():
    save(name, data)

framework_parent = results / 'training-qwen-a68952c-swift-initial-CPU-v1'
domain = results / 'training-internvl-USGS72-native-CPU-v1'
reference = results / 'training-internvl-rank00-reference-provision-v1' / 'payload'
runtime = results / 'training-vlm-swift-offline-runtime-v1'
binds = [(path,path) for path in (out,framework_parent/'framework',framework_parent/'source', reference,
    domain/'data',domain/'source',domain/'native-caller-rows.json',domain/'native-PEFT-structure.json')]
binds.extend(((runtime,'/swift-runtime'),('/usr/local/hyhal','/opt/hyhal')))
cmd = ['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--contain','--no-home','--pwd','/']
for host,target in binds:
    cmd.extend(('--bind', f'{host}:{target}:ro'))
shell = ('set -euo pipefail; source /opt/dtk/env.sh; export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 '
         'PYTHONPATH=/swift-runtime/site CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= '
         'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; '
         'exec /usr/bin/python3 -B -u "$1"')
argv = ['srun','--jobid='+parent,'--job-name='+label,'--exact','--overlap','--immediate=7','--nodes=1','--ntasks=1',
        '--nodelist='+node,'--cpus-per-task=2','--mem=4096M','--gres=none','--time=00:02:00', *cmd,
        str(root/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell,'asset-check',str(out/'check_assets.py')]
try:
    p = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=96, check=False)
    rc, stdout, stderr = p.returncode, p.stdout, p.stderr
except subprocess.TimeoutExpired as e:
    rc, stdout, stderr = None, e.stdout or b'', e.stderr or b''
save('step.stdout',stdout)
save('step.stderr',stderr)
rows = accounting(parent)
for _ in range(6):
    if len(rows)==1 and rows[0][2] not in {'RUNNING','PENDING','COMPLETING'}:
        break
    time.sleep(0.5)
    rows=accounting(parent)
prefix=b'RESULT_JSON '
structured=[line.removeprefix(prefix) for line in stdout.splitlines() if line.startswith(prefix)]
result = json.loads(structured[0]) if len(structured)==1 and len(structured[0])<=12000 else None
passed = (rc==0 and isinstance(result,dict) and len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0']
          and rows[0][5:7]==[node,'cpu=2,mem=4G,node=1'] and result.get('step')==rows[0][0]
          and result.get('node')==node and result.get('actual_GPU_steps')==0)
observation = dict(request=request, published=packet['publication'], runner_rc=rc, accounting=rows, asset_check=result,
                   passed=passed, target_result=None if passed else next((e.decode() for e in
                    (b'FileNotFoundError',b'ValueError',b'PermissionError',b'ModuleNotFoundError',b'Timeout') if e in stderr),None),
                   private_stdout_SHA256=sha256(stdout).hexdigest(), private_stderr_SHA256=sha256(stderr).hexdigest())
save('outcome.json',encode(observation))
print(json.dumps(observation,sort_keys=True))
'''


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--publication")
    args = cli.parse_args()
    compile(HOST, "<target controller>", "exec")
    claim_path = HERE / "CLAIM.json"
    claim_raw = claim_path.read_bytes()
    claim = json.loads(claim_raw)
    mount = claim["prior_mount_evidence"]
    if sha256((G / mount["path"]).read_bytes()).hexdigest() != mount["sha256"]:
        raise ValueError("current-node original mount evidence differs")
    files = {name:(ORIGINAL / name).read_bytes() for name in FILES}
    files["check_assets.py"] = (HERE / "check_assets.py").read_bytes()
    if {name:sha256(raw).hexdigest() for name,raw in files.items()} != claim["source_SHA256"]:
        raise ValueError("source pins differ")
    for name, raw in files.items():
        if name.endswith(".py"):
            compile(raw, name, "exec")
    if args.check:
        print(json.dumps({"checked_files": len(files), "request": claim["request"], "remote_calls": 0}))
        return 0
    if not re.fullmatch(r"[a-f0-9]{40}", args.publication):
        raise ValueError("exact public commit required")
    for f in (claim_path, HERE / "check_assets.py", HERE / "launch.py"):
        actual = subprocess.check_output(["git","-c",f"safe.directory={G.as_posix()}","show",
                                          args.publication + ":" + f.relative_to(G).as_posix()],cwd=G)
        if actual != f.read_bytes().replace(b"\r\n",b"\n"):
            raise ValueError("worktree source differs from the published bytes")
    source_path = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
    helper = source_path.read_bytes()
    if sha256(helper).hexdigest() != "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd":
        raise ValueError("existing strict SSH helper differs")
    transport = ModuleType("existing_transport")
    exec(compile(helper, str(source_path), "exec"), vars(transport))
    argv = [*transport.ssh(G.parent), shlex.join(["/usr/local/python3.12/bin/python3","-I","-S","-B","-c",HOST])]
    raw = json.dumps({"publication":args.publication,"claim":b64encode(claim_raw).decode(),
                      "files":{name:b64encode(data).decode() for name,data in files.items()}},sort_keys=True).encode()
    if len(raw) > 310000:
        raise ValueError("published source payload exceeds its cap")
    request = claim["request"]
    with (LOCAL / f"internvl-usgs72-current-input-cpu-{request}-send.json").open("x",encoding="utf-8") as out:
        json.dump({"request":request,"payload_SHA256":sha256(raw).hexdigest(),"publication":args.publication},out,sort_keys=True)
    try:
        p = subprocess.run(argv,input=raw,capture_output=True,timeout=130,check=False)
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, stdout, stderr = None, e.stdout or b"", e.stderr or b""
    try:
        observed = json.loads(stdout) if rc==0 and 0 < len(stdout) <= 16384 else None
    except (ValueError, UnicodeError):
        observed = None
    record = {"request":request,"rc":rc,"observed":observed,"output_SHA256":sha256(stdout).hexdigest(),
              "transport_stderr_SHA256":sha256(stderr).hexdigest()}
    if observed is None:
        record["failure_class"] = next((s.decode() for s in (b"Permission denied", b"Connection closed", b"Timeout",
            b"CalledProcessError", b"ValueError", b"FileNotFoundError", b"KeyError") if s in stderr), None)
    name = f"internvl-usgs72-current-input-cpu-{request}-result.json"
    data = (json.dumps(record,indent=2,sort_keys=True)+"\n").encode()
    with (LOCAL / name).open("xb") as out:
        out.write(data)
    print(json.dumps({"file":name,"SHA256":sha256(data).hexdigest(),**record},sort_keys=True))
    return int(observed is None or observed["passed"] is not True)


if __name__ == "__main__":
    raise SystemExit(main())
