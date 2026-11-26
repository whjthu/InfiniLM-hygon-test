"""Stage one exact public-derived clip and inspect it under the original zero-GRES image."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import shlex
import subprocess


REPO = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
LOCAL = REPO / "local-only"
MEDIA = LOCAL / "hunyuan-abo-b07m6phm8c-orbit-9frames-v1"
ROUTE = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
ROUTE_SHA = "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
LABEL = "training-hunyuanvideo-abo-vae-inspect-v1"
CLAIM = REPO / "coordination/tasks" / (LABEL + ".json")
PARENT, NODE = "1103606", "f13r1n12"
MEMBERS = {
    "binding.json": REPO / "code/bw1000/training-hunyuanvideo-cached-native25-target-v3/NATIVE_BINDING.json",
    "vae-metadata.json": LOCAL / "hunyuan-abo-separate-verified-vae-current-metadata-v1.json",
    **{f"src/{name}": PACKAGE / name for name in ("worker.py", "contract.py", "readback.py")},
    **{f"media/{name}": MEDIA / name for name in ("render.json", "frames.jsonl", *(f"frame-{i:02d}.private.png" for i in range(9)))},
}


REMOTE = r'''
from base64 import b64decode
from hashlib import sha256
import json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

project=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
results=project/'results'
runtime=results/'training-t06-sdxl-runtime-v1'
native=results/'training-hunyuanvideo-original-source-v1/source'
vae=results/'training-hunyuanvideo-public-vae-v1/vae'
image=project/'containers/dtk26-sandbox-r1'
apptainer='/public/software/apps/apptainer/1.3.4/bin/apptainer'
label='training-hunyuanvideo-abo-vae-inspect-v1'
parent,node='1103606','f13r1n12'

def pin(raw):
    return dict(SHA256=sha256(raw).hexdigest(),bytes=len(raw))

def save(path,raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())

def encode(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()

value=json.loads(sys.stdin.buffer.readline(7000000))
request=value['request']
assert re.fullmatch('[0-9a-f]{32}',request) and re.fullmatch('[0-9a-f]{40}',value['publication'])
pins=value['members']
names={'binding.json','vae-metadata.json','src/worker.py','src/contract.py','src/readback.py','media/render.json','media/frames.jsonl'}
names.update(f'media/frame-{i:02d}.private.png' for i in range(9))
assert set(pins)==set(value['payload'])==names and sum(x['bytes'] for x in pins.values())==3021024
assert sha256(encode(dict(actual_frames=9,shape_THWC=[9,480,848,3],member_count=16,bytes=3021024,files=pins))).hexdigest()=='063b973a0e9a9c6703c8e1d3bcd40048d20ed45de469bc719b603bb49bfbe2ed'
files={name:b64decode(value['payload'][name],validate=True) for name in sorted(names)}
assert all(pin(raw)==pins[name] for name,raw in files.items())
job=label+'-'+request[:12]
namespace=results/label
root=namespace/request
start=datetime.fromtimestamp(time.time()-1800,timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
columns='JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'

def accounting():
    q=subprocess.run(['env','TZ=UTC','LC_ALL=C','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S',
        'sacct','-S',start,'-nP','-j',parent,'-o',columns],capture_output=True,timeout=9,check=True)
    rows=[line.split('|') for line in q.stdout.decode().splitlines()]
    return [row for row in rows if len(row)==9 and row[1]==job]

assert not accounting() and not os.path.lexists(root)
assert all(d.is_dir() for d in (runtime,native,vae,image)) and os.access(apptainer,os.X_OK)
os.umask(0o077)
namespace.mkdir(mode=0o700,exist_ok=True)
assert namespace.is_dir() and not namespace.is_symlink()
root.mkdir(mode=0o700)
for name in ('media','src','home','cache','tmp'):
    (root/name).mkdir(mode=0o700)
for name,raw in files.items():
    save(root/name,raw)
env=('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
     'export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
     'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; '
     'export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=/training-runtime/site; '
     'export HOME="$1/home" XDG_CACHE_HOME="$1/cache" TMPDIR="$1/tmp"; shift; exec python3 -B -u "$@"')
binds=[(root,root,'rw'),(root/'src',root/'src','ro'),(root/'media',root/'media','ro'),
       (root/'binding.json',root/'binding.json','ro'),(root/'vae-metadata.json',root/'vae-metadata.json','ro'),
       (runtime,Path('/training-runtime'),'ro'),(native,native,'ro'),(vae,vae,'ro'),
       (Path('/opt/hyhal'),Path('/opt/hyhal'),'ro')]
command=[apptainer,'exec','--no-home','--no-mount','cwd','--pwd','/']
for source,target,mode in binds:
    command+=['--bind',f'{source}:{target}:{mode}']
epoch=time.time()
authority={
    'request':request,'label':'training-hunyuanvideo-abo-vae-encode-v1','mode':'inspect',
    'recipe':'ABO_RGB9_NCTHW_BF16_NATIVE_POSTERIOR_MODE_FP32_SCALED_V1','parent':parent,'node':node,
    'root_authorized':True,'network_GO':False,'training_GO':False,'text_GO':False,
    'zero_GRES_GO':True,'GPU_GO':False,'weight_payload_GO':False,'allocatable_DCU':0,'epoch':epoch,'deadline':epoch+120,
    'media_receipt_SHA256':pins['media/render.json']['SHA256'],'upstream_binding_SHA256':pins['binding.json']['SHA256'],
    'vae_metadata_SHA256':pins['vae-metadata.json']['SHA256'],
    'original_full_weight_SHA256':'2609ed7033c052fdf164afcac71e8a7c82afccdbb001d673444a72f194fbb918',
}
raw=encode(authority)
save(root/'authority.json',raw)
save(root/'intent.json',encode(dict(request=request,publication=value['publication'],claim_SHA256=value['claim_SHA256'],
         job=job,epoch=epoch,deadline=epoch+120,authority_SHA256=pin(raw)['SHA256'],staged_member_pins=pins)))
command += [str(image),'/bin/bash','--noprofile','--norc','-c',env,label,str(root),
            str(root/'src/worker.py'),'inspect','--authority',str(root/'authority.json'),
            '--authority-sha256',pin(raw)['SHA256'],'--request',request,'--media',str(root/'media'),
            '--upstream-binding',str(root/'binding.json'),'--vae-metadata',str(root/'vae-metadata.json'),
            '--out',str(root/'worker')]
gate=r"""import json,os,sys,time
from hashlib import sha256
from pathlib import Path
root=Path(sys.argv[1]);end=time.monotonic()+12
while True:
    try:
        intent=json.loads((root/'intent.json').read_bytes())
        assert sha256((root/'authority.json').read_bytes()).hexdigest()==intent['authority_SHA256']
        for name,expected in intent['staged_member_pins'].items():
            p=root/name
            assert not p.is_symlink()
            with p.open('rb') as stream: value=stream.read(expected['bytes']+1)
            assert len(value)==expected['bytes'] and sha256(value).hexdigest()==expected['SHA256']
        break
    except FileNotFoundError:
        if time.monotonic()>=end: raise
        time.sleep(.15)
os.execv(sys.argv[2],sys.argv[2:])
"""
args=['srun','--jobid='+parent,'--job-name='+job,'--exact','--immediate=6','--nodes=1','--ntasks=1',
      '--nodelist='+node,'--cpus-per-task=2','--mem=4G','--gres=none','--time=00:02:00','--kill-on-bad-exit=1',
      '/usr/local/python3.12/bin/python3','-I','-S','-B','-c',gate,str(root),*command]
rc=None
with (root/'step.stdout').open('xb') as stdout,(root/'step.stderr').open('xb') as stderr:
    try:
        child=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,timeout=137)
        rc=child.returncode
    except subprocess.TimeoutExpired:
        pass
rows=accounting()
for _ in range(3):
    if len(rows)==1 and rows[0][2] not in {'RUNNING','PENDING','COMPLETING'}: break
    time.sleep(1)
    rows=accounting()
worker=root/'worker/worker.json'
summary=None
if worker.is_file():
    with worker.open('rb') as stream: raw=stream.read(16385)
    assert 0<len(raw)<=16384
    d=json.loads(raw)
    assert d['request']==request and d['actual_GPU']==d['optimizer_updates']==0
    summary=dict(identity=pin(raw),**{k:d[k] for k in ('passed','phase','actual_step','validated_frames','actual_candidate_latents')},failure=d.get('failure'))
with (root/'step.stderr').open('rb') as f:
    error=f.read(65536)
reply=dict(request=request,child_rc=rc,accounting=rows,worker=summary,stderr_pin=pin(error),
           stage_members=len(files),stage_bytes=sum(len(x) for x in files.values()),
           outcome='OBSERVED_TERMINAL' if len(rows)==1 and rows[0][2] not in {'RUNNING','PENDING','COMPLETING'} else 'UNKNOWN')
save(root/'launcher-result.json',encode(reply))
print(json.dumps(reply,sort_keys=True))
'''


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def fresh(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", required=True)
    args = parser.parse_args()
    pub = args.publication
    assert re.fullmatch(r"[0-9a-f]{40}", pub)
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]

    def query(*parts):
        return subprocess.check_output([*git, *parts], cwd=REPO, stdin=subprocess.DEVNULL, timeout=18)

    assert query("remote", "get-url", "origin").strip() == b"git@github.com:whjthu/InfiniLM-hygon-test.git"
    assert query("ls-remote", "--exit-code", "origin", "refs/heads/bw1000-experiments").split()[0] == pub.encode()
    claim_bytes = CLAIM.read_bytes()
    assert query("show", pub + ":" + CLAIM.relative_to(REPO).as_posix()) == claim_bytes
    claim = json.loads(claim_bytes)
    assert claim["target"] == dict(parent=PARENT, node=NODE, cpus=2, memory_MiB=4096, GPUs=0, epoch_seconds=120)
    assert pin(Path(__file__).read_bytes()) == claim["runner"] and not claim["replay_allowed"]
    request = claim["request"]
    pins = claim["members"]
    assert re.fullmatch(r"[0-9a-f]{32}", request) and set(pins) == set(MEMBERS)
    files = {}
    for name, path in MEMBERS.items():
        with path.open("rb") as stream:
            raw = stream.read(pins[name]["bytes"] + 1)
        assert pin(raw) == pins[name]
        if name.startswith("src/") or name == "binding.json":
            assert query("show", pub + ":" + path.relative_to(REPO).as_posix()) == raw
        files[name] = b64encode(raw).decode()
    assert sha256(ROUTE.read_bytes()).hexdigest() == ROUTE_SHA
    ssh = runpy.run_path(str(ROUTE))["ssh"](REPO.parent)
    assert ssh[1] == "-v"
    ssh[1:2] = ["-o", "LogLevel=ERROR"]
    payload = encoded(dict(request=request, publication=pub, claim_SHA256=pin(claim_bytes)["SHA256"], members=pins, payload=files))
    compile(REMOTE, "<ABO inspect login carrier>", "exec")
    base = LOCAL / (LABEL + "-" + request)
    base.mkdir(mode=0o700)
    fresh(base / "intent.json", dict(request=request, claim=pin(claim_bytes), publication=pub, wire=pin(payload), replay_allowed=False))
    argv = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", REMOTE])
    try:
        run = subprocess.run([*ssh, argv], input=payload, capture_output=True, timeout=170)
        value = json.loads(run.stdout) if run.returncode == 0 and 0 < len(run.stdout) <= 8000 else None
        if value:
            assert value["request"] == request
        result = dict(request=request, rc=run.returncode, stderr=pin(run.stderr), response=value,
                      outcome=value["outcome"] if value else "UNKNOWN")
    except subprocess.TimeoutExpired:
        result = dict(request=request, rc=None, outcome="UNKNOWN")
    fresh(base / "result.json", result)
    print(json.dumps(dict(receipt=pin(encoded(result)), **result), sort_keys=True))
    return int(result["outcome"] != "OBSERVED_TERMINAL")


if __name__ == "__main__":
    raise SystemExit(main())
