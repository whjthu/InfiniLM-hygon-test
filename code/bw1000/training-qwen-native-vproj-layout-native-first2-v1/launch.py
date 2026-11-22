"""Install the published candidate and start only its finite Slurm controller."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
G = D.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main(commit, transport):
    manifest_raw = (D/'SOURCE_FILES.json').read_bytes()
    relative = D.relative_to(G).as_posix()
    git = ['git', '-c', 'safe.directory='+str(G)]
    published = subprocess.check_output([*git, 'cat-file', 'blob', commit+':'+relative+'/SOURCE_FILES.json'],
                                        cwd=G, timeout=15)
    if (published != manifest_raw
            or subprocess.check_output([*git,'status','--short','--',relative],cwd=G,timeout=12)):
        raise ValueError('Published source manifest or SSH transport differs')
    remote = subprocess.check_output([*git, 'ls-remote', 'origin',
                                      'refs/heads/bw1000-experiments'], cwd=G, timeout=15).split()
    if not remote or remote[0].decode('ascii') != commit:
        raise ValueError('Current published commit differs from the claim')
    files = {}
    for name, pin in json.loads(manifest_raw)['files'].items():
        raw = subprocess.check_output([*git,'cat-file','blob',commit+':'+relative+'/'+name],
                                      cwd=G,timeout=15)
        if raw != (D/name).read_bytes():
            raise ValueError('Published source bytes differ: '+name)
        if Path(name).name != name or (len(raw), sha(raw)) != (pin['bytes'], pin['SHA256']):
            raise ValueError('Published package member differs')
        files[name] = base64.b64encode(raw).decode()
    files['SOURCE_FILES.json'] = base64.b64encode(manifest_raw).decode()
    claim = json.loads((D/'CLAIM.json').read_bytes())
    local = G/'local-only'/D.name
    local.mkdir(mode=0o700)
    epoch = time.time()
    intent = dict(claim_commit=commit, request_id=claim['request_id'], epoch=epoch,
                  deadline=epoch+claim['whole_epoch_seconds'], source_manifest_SHA256=sha(manifest_raw),
                  resource={k:claim[k] for k in ('parent','node','DCU','CPU','memory_MiB')},
                  formal_performance=False, planned_phases=['native_update_one','native_update_two'])
    write(local/'intent.json', intent)
    bootstrap = """import base64,hashlib,json,os,pathlib,subprocess,time
p=PAYLOAD
root=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')/p['label']
intent=p['intent']
assert abs(time.time()-intent['epoch'])<30
intent['epoch']=time.time();intent['deadline']=intent['epoch']+p['whole_seconds']
assert not root.exists()
parent=intent['resource']['parent']
account=subprocess.check_output(['sacct','-n','-P','-j',parent,'--format=JobIDRaw,JobName%100'],capture_output=False,text=True,timeout=12)
labels={p['label']+suffix for suffix in ('','-pre','-post')}
assert not any(row.split('|')[1] in labels for row in account.splitlines())
root.mkdir(mode=0o700)
src=root/'src';src.mkdir(mode=0o700)
for name,data in p['files'].items():
 assert pathlib.Path(name).name==name
 with (src/name).open('xb') as f:f.write(base64.b64decode(data,validate=True));f.flush();os.fsync(f.fileno())
layout=root/'upstream-layout';layout.mkdir(mode=0o700)
os.link(src/'UPSTREAM_SOURCE_FILES.json',layout/'SOURCE_FILES.json')
os.link(src/'upstream_layout.py',layout/'upstream_layout.py')
with (root/'intent.json').open('x') as f:json.dump(intent,f);f.flush();os.fsync(f.fileno())
with (root/'keeper.log').open('xb') as log:
 child=subprocess.Popen(['/usr/local/python3.12/bin/python3','-B','-u',str(src/'controller.py')],
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=pathlib.Path('/proc',str(child.pid),'stat').read_text().split(') ',1)[1].split()
owner=dict(PID=child.pid,PGID=int(v[2]),startticks=int(v[19]),request_id=intent['request_id'],epoch=intent['epoch'],deadline=intent['deadline'],
           boot_id_SHA256=hashlib.sha256(pathlib.Path('/proc/sys/kernel/random/boot_id').read_bytes()).hexdigest())
with (root/'keeper-owner.json').open('x') as f:json.dump(owner,f);f.flush();os.fsync(f.fileno())
print(json.dumps(owner),flush=True)
""".replace('PAYLOAD', repr(dict(label=claim['label'], files=files, intent=intent, whole_seconds=claim['whole_epoch_seconds'])), 1)
    if len(bootstrap.encode()) > 300000:
        raise ValueError('Source-only payload exceeds its bound')
    result = dict(request=intent['request_id'], rc=None, outcome='UNKNOWN')
    try:
        p = transport(bootstrap.encode())
        result.update(rc=p.returncode, stderr_SHA256=sha(p.stderr), stdout_SHA256=sha(p.stdout))
        if p.returncode == 0:
            owner = json.loads(p.stdout)
            if (owner['request_id'] != intent['request_id'] or abs(owner['epoch'] - epoch) >= 30
                    or owner['deadline'] - owner['epoch'] != claim['whole_epoch_seconds']):
                raise ValueError('Original controller owner identity differs')
            result.update(outcome='CONTROLLER_OWNER_REPORTED', owner=owner)
    except subprocess.TimeoutExpired:
        pass
    write(local/'dispatch.json', result)
    print(json.dumps(result))
    return int(result['outcome'] != 'CONTROLLER_OWNER_REPORTED')
