"""Run one published Gemma zero-GRES IPC fixture with the existing project transport."""

import argparse
from base64 import b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import shlex
import subprocess


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
LOCAL = REPO / "local-only"
LABEL = "gemma-r120-posix-ipc-fixture-v2"
CLAIM = REPO / "coordination/tasks" / (LABEL + ".json")


REMOTE = r'''
from base64 import b64decode
from datetime import datetime,timezone
from hashlib import sha256
import json,os,re,subprocess,sys,time
from pathlib import Path

program=json.loads(sys.stdin.buffer.read(150000))
claim=program['claim'];request=claim['request']
assert re.fullmatch('[0-9a-f]{32}',request) and os.getuid()==154459
assert claim['target']==dict(parent='1103606',node='f13r1n12',CPU=4,DCU=0,mem_MiB=8192,epoch_seconds=120)
assert claim['root_authorized'] and not claim['replay_allowed'] and not claim['training_GO']
assert re.fullmatch('/tmp/gr[0-9a-f]{8}',claim['short_alias'])
members={k:b64decode(v,validate=True) for k,v in program['members'].items()}
mf='SOURCE_FILES.json'
def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())
assert pin(members[mf])==claim['source_manifest']
names={'fixture.py','native_worker_off.py','readback.py','short_lifecycle.py','short_mount.py','cohort.py','provision.py','PROVIDER_BINDING.json'}
assert set(members)==names|{mf}
assert {n:pin(members[n]) for n in names}==json.loads(members[mf])['files']
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/incoming/gemma-r120-posix-ipc-fixture-v2')/request
job='gemma-r120-posix-ipc-fixture-v2-'+request[:12]
argv=claim['argv_template']
assert argv[:14]==['srun','--jobid=1103606','--job-name='+job,'--exact','--immediate=6','--nodes=1','--ntasks=1',
    '--nodelist=f13r1n12','--cpus-per-task=4','--mem=8G','--gres=none','--time=00:02:00','--kill-on-bad-exit=1','/usr/local/python3.12/bin/python3']
assert argv[-2:]==['--authority-sha256','0'*64] and argv.count('0'*64)==1
start=datetime.fromtimestamp(time.time()-180,timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
def account():
 q=subprocess.run(['env','TZ=UTC','LC_ALL=C','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S','sacct','-S',start,'-nP','-j','1103606',
  '-o','JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'],capture_output=True,check=True,timeout=9)
 rows=[r.split('|') for r in q.stdout.decode().splitlines()]
 return [r for r in rows if len(r)==9 and r[1]==job]
assert not os.path.lexists(root) and not account()
def save(path,raw):
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def encoded(d):return (json.dumps(d,sort_keys=True,allow_nan=False)+'\n').encode()
os.umask(0o077)
root.parent.mkdir(mode=0o700,exist_ok=True)
assert not root.parent.is_symlink()
root.mkdir(mode=0o700)
src=root/'src';src.mkdir(mode=0o700)
for name,raw in sorted(members.items()):save(src/name,raw)
now=time.time()
a={k:claim[k] for k in ('request','label','short_alias','original_4177_archive_SHA256','root_authorized','zero_GRES_GO','GPU_GO','training_GO','weights_GO')}
a.update(parent='1103606',node='f13r1n12',epoch=now,deadline=now+120,source_manifest_SHA256=pin(members[mf])['SHA256'])
b=encoded(a);save(root/'approval.json',b)
argv[-1]=pin(b)['SHA256']
# Keep the fixture's create-once operation empty until it owns the numbered step.
log=root.parent/(request+'.carrierlog');log.mkdir(mode=0o700)
rc=None
with (log/'stdout').open('xb') as out,(log/'stderr').open('xb') as err:
 try:rc=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=137).returncode
 except subprocess.TimeoutExpired:pass
rows=account()
for _ in range(3):
 if len(rows)==1 and rows[0][2] not in {'RUNNING','PENDING','COMPLETING'}:break
 time.sleep(1);rows=account()
r=dict(request=request,rc=rc,accounting=rows,metadata={},outcome='OBSERVED_TERMINAL' if len(rows)==1 and rows[0][2] not in {'RUNNING','PENDING','COMPLETING'} else 'UNKNOWN')
fields={'host.json':('passed','failure','carrier_attempted','container_rc','guest_passed','host_marker_only_after','host_managers_absent','step'),
 'guest.json':('passed','failure','step','actual_dataset_map_rows','manager_queue_roundtrip','remaining_child_count','registered_cleanup'),
 'owner-gate.json':(), 'short-private-tmp-creation.json':(), 'short-private-tmp-cleanup.json':(), 'short-private-tmp-lifecycle.json':()}
for name,keys in fields.items():
 p=root/name
 if not p.is_file():continue
 with p.open('rb') as f:raw=f.read(200001)
 assert 0<len(raw)<=200000
 d=json.loads(raw);r['metadata'][name]={'pin':pin(raw),'projection':{k:d[k] for k in keys if k in d}}
for name in ('stdout','stderr'):
 with (log/name).open('rb') as f:raw=f.read(200001)
 if len(raw)<=200000:r[name]=pin(raw)
save(log/'result.json',encoded(r))
print(json.dumps(r,sort_keys=True))
'''


def pin(raw):
    return dict(bytes=len(raw), SHA256=sha256(raw).hexdigest())


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication", required=True)
    pub = p.parse_args().publication
    assert re.fullmatch(r"[0-9a-f]{40}", pub)
    git = ["git", "--no-optional-locks", "-c", "safe.directory=" + REPO.as_posix()]

    def q(*a):
        return subprocess.check_output([*git, *a], cwd=REPO, stdin=subprocess.DEVNULL, timeout=18)

    assert q("remote", "get-url", "origin").strip() == b"git@github.com:whjthu/InfiniLM-hygon-test.git"
    assert q("ls-remote", "--exit-code", "origin", "refs/heads/bw1000-experiments").split()[0] == pub.encode()
    claim = json.loads(CLAIM.read_bytes())
    assert q("show", pub + ":" + CLAIM.relative_to(REPO).as_posix()) == CLAIM.read_bytes()
    assert claim["launcher"] == pin(Path(__file__).read_bytes())
    mf = json.loads((PACKAGE / "SOURCE_FILES.json").read_bytes())
    members = {}
    for name, expected in {**mf["files"], "SOURCE_FILES.json": claim["source_manifest"]}.items():
        path = PACKAGE / name
        raw = path.read_bytes()
        assert pin(raw) == expected and q("show", pub + ":" + path.relative_to(REPO).as_posix()) == raw
        members[name] = b64encode(raw).decode()
    route = LOCAL / "run-minicpm-sdpa-target-asset-preflight-original.py"
    assert pin(route.read_bytes())["SHA256"] == "530f4d14b026abf79ed82887ad95490242ac436c7f8e05ed6f32908553e106bd"
    ssh = runpy.run_path(str(route))["ssh"](REPO.parent)
    assert ssh[1] == "-v"
    ssh[1:2] = ["-o", "LogLevel=ERROR"]
    wire = encoded(dict(claim=claim, members=members))
    compile(REMOTE, "<Gemma POSIX IPC remote carrier>", "exec")
    root = LOCAL / (LABEL + "-" + claim["request"])
    root.mkdir(mode=0o700)
    with (root / "intent.json").open("xb") as f:
        f.write(encoded(dict(request=claim["request"], publication=pub, wire=pin(wire), replay_allowed=False)))
    command = shlex.join(["/usr/local/python3.12/bin/python3", "-I", "-S", "-B", "-c", REMOTE])
    try:
        r = subprocess.run([*ssh, command], input=wire, capture_output=True, timeout=166)
        data = json.loads(r.stdout) if r.returncode == 0 and 0 < len(r.stdout) <= 11000 else None
        assert data is None or data["request"] == claim["request"]
        receipt = dict(rc=r.returncode, stderr=pin(r.stderr), response=data, outcome=data["outcome"] if data else "UNKNOWN")
    except subprocess.TimeoutExpired:
        receipt = dict(rc=None, outcome="UNKNOWN")
    raw = encoded(receipt)
    with (root / "result.json").open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(receipt=pin(raw), **receipt), sort_keys=True))
    return int(receipt["outcome"] == "UNKNOWN")


if __name__ == "__main__":
    raise SystemExit(main())
