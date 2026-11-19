"""Run one source-pinned CPU integration check inside the allocated target container."""
import base64
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HELPER = Path("C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py")
HELPER_SHA = "3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1"
REQUEST = "a731065879bb4e4294ed50b938396963"

REMOTE = """import base64, hashlib, json, os, pathlib, re, subprocess, time
request, files = PAYLOAD
epoch=time.monotonic()
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
results=base/'results'
root=results/('training-qwen-grouped-adamw-swift-factory-CPU-v1-'+request)
parent, node, label = '1103606', 'f13r1n10', 'training-qwen-grouped-adamw-swift-factory-CPU-v1-cpu'
def cli(*args):
 return subprocess.check_output(args, text=True, timeout=8).splitlines()
row,=cli('scontrol','show','job',parent,'-o')
fields=dict(re.findall(r'(?:^|\\s)([A-Za-z][A-Za-z0-9_]*)=(\\S+)',row))
assert fields['JobState']=='RUNNING' and fields['UserId'].endswith('('+str(os.getuid())+')')
assert node in cli('scontrol','show','hostnames',fields['NodeList'])
for line in cli('squeue','--steps','-h','-j',parent,'-o','%i|%N'):
 handle, names=line.split('|',1)
 if handle in {parent+'.batch',parent+'.extern'}:continue
 assert re.fullmatch(parent+r'\\.\\d+',handle) and names not in ('', '(null)')
 assert node not in cli('scontrol','show','hostnames',names)
columns='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'
def accounting():
 return [line.split('|') for line in cli('sacct','-n','-P','-j',parent,'--format='+columns)
         if line.split('|')[1]==label]
assert not accounting()
root.mkdir(mode=0o700)
for name, item in files.items():
 p=root/name
 p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 with p.open('xb') as stream:stream.write(base64.b64decode(item,validate=True))
def save(path,value):
 with (root/path).open('x',encoding='utf-8') as stream:json.dump(value,stream)
save('intent.json',dict(request=request, GPU=0, started=time.time()))
cmd=['srun','--jobid='+parent,'--job-name='+label,'--nodelist='+node,'--exact','--nodes=1','--ntasks=1',
 '--cpus-per-task=4','--mem=8192M','--gres=none','--immediate=5','--time=2',
 '/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home',
 '--bind',str(root)+':/candidate:ro','--bind','/opt/hyhal:/opt/hyhal:ro',
 '--bind',str(results/'training-vlm-swift-offline-runtime-v1/site')+':/swift-site:ro',
 '--bind',str(results/'training-minicpm-hygon-processor-r2/cpu/runtime-site')+':/transformers-site:ro',
 str(base/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',
 'set -eo pipefail; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= PYTHONPATH=/candidate/framework:/transformers-site:/swift-site; python3 -B /candidate/checks/test_native_swift_factory.py -v; exec python3 -B -c "$1"',
 'candidate','from swift.optimizers import optimizers_map; from vlm.plugins.grouped_adamw import KEY, GroupedAdamWCallback, verify_providers; verify_providers(); assert optimizers_map[KEY] is GroupedAdamWCallback; print("ACTUAL_SWIFT_REGISTRY_AND_PROVIDER_PINS")']
rc=None
try:
 with (root/'worker.log').open('xb') as log:
  rc=subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=max(1,145-(time.monotonic()-epoch))).returncode
except subprocess.TimeoutExpired:pass
for _ in range(4):
 rows=accounting()
 if len(rows)==1 and rows[0][2] not in ('RUNNING','PENDING','COMPLETING'):break
 time.sleep(1)
log=(root/'worker.log').read_bytes()
assert len(log)<=262144
tests=[line.decode().split(' (')[0] for line in log.splitlines()
       if line.startswith(b'test_') and b' ... ok' in line]
result=dict(request=request, rc=rc, accounting_columns=columns.split(','), accounting=rows,
   elapsed_seconds=time.monotonic()-epoch, tests=tests, registry_pass=b'ACTUAL_SWIFT_REGISTRY_AND_PROVIDER_PINS' in log,
   log_SHA256=hashlib.sha256(log).hexdigest(), GPU_cards=0,
   errors=[line.decode(errors='replace')[:240] for line in log.splitlines()
       if re.match(rb'^(?:RuntimeError|ImportError|ModuleNotFoundError|ValueError|TypeError|AssertionError):',line)][-4:])
save('terminal.json',result)
print(json.dumps(result),flush=True)
"""


def main(publication):
    relative = HERE.relative_to(REPO).as_posix()
    raw = (HERE / "SOURCE_FILES.json").read_bytes()
    base = ["git", "-c", f"safe.directory={REPO}"]
    if subprocess.check_output([*base, "show", f"{publication}:{relative}/SOURCE_FILES.json"], cwd=REPO, timeout=12) != raw:
        raise ValueError("published CPU source manifest differs")
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError("existing connection helper differs")
    files = {}
    for key, pin in json.loads(raw)["files"].items():
        item = (HERE / pin["relative"]).read_bytes()
        if (len(item), hashlib.sha256(item).hexdigest()) != (pin["bytes"], pin["SHA256"]):
            raise ValueError("frozen CPU source differs")
        files[key] = base64.b64encode(item).decode()
    ssh = runpy.run_path(str(HELPER))["ssh"]
    target = REPO / "local-only" / HERE.name
    target.mkdir()
    body = REMOTE.replace("PAYLOAD", repr((REQUEST, files)), 1).encode()
    receipt = dict(request=REQUEST, publication=publication,
                   payload_SHA256=hashlib.sha256(body).hexdigest())
    (target/"intent.json").write_text(json.dumps(receipt),encoding="utf-8")
    try:
        p = subprocess.run([ssh[0], "-o", "StrictHostKeyChecking=yes", *ssh[1:],
                            "/usr/local/python3.12/bin/python3 -I -S -B -"],
                           input=body, capture_output=True, timeout=185)
        receipt.update(transport_rc=p.returncode,
                       stderr_SHA256=hashlib.sha256(p.stderr).hexdigest())
        if p.returncode==0:
            receipt["actual"]=json.loads(p.stdout)
    except subprocess.TimeoutExpired:
        receipt["transport_rc"]=None
    (target/"observed.json").write_text(json.dumps(receipt),encoding="utf-8")
    print(json.dumps(receipt))
    return int(receipt.get("transport_rc")!=0)


if __name__=="__main__":
    raise SystemExit(main(sys.argv[1]))
