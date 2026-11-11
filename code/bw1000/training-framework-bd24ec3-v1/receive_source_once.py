"""Install the complete SHA-bound source tree once and inspect its normal entry."""
import base64,hashlib,io,json,shlex,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
epoch=time.time();deadline=epoch+90
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');ROOT=R/'results/training-framework-bd24ec3-v1'
def query(*args):return subprocess.check_output(args,text=True,timeout=min(12,max(1,deadline-time.time()-5))).strip()
assert not ROOT.exists()
parent=query('squeue','-h','-j','1087656','-o','%T|%D|%N').split('|')
nodes=query('scontrol','show','hostnames',parent[-1]).splitlines()
assert parent[:2]==['RUNNING','40'] and len(nodes)==len(set(nodes))==40 and sorted(nodes)==EXPECTED_NODES
manifest_raw=base64.b64decode(MANIFEST_B64,validate=True);manifest=json.loads(manifest_raw)
assert hashlib.sha256(manifest_raw).hexdigest()==MANIFEST_SHA
payload=sys.stdin.buffer.read(8000001);assert len(payload)==manifest['archive_bytes']<8000000
assert hashlib.sha256(payload).hexdigest()==manifest['archive_SHA256']
ROOT.mkdir(mode=0o700);(ROOT/'helpers').mkdir();(ROOT/'evidence').mkdir()
with (ROOT/'intent.json').open('x') as f:json.dump({'claim_commit':CLAIM,'epoch':epoch,'deadline':deadline,'parent':'1087656','actual_parent_state':parent[0],'actual_nodes':nodes,'framework_commit':manifest['framework_commit'],'framework_tree':manifest['framework_tree'],'manifest_SHA256':MANIFEST_SHA,'archive_SHA256':manifest['archive_SHA256'],'GPU_allocation':0},f)
with (ROOT/'FRAMEWORK_FILES.json').open('xb') as f:f.write(manifest_raw)
seen=set()
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
    for item in archive:
        path=PurePosixPath(item.name);assert not path.is_absolute() and '..' not in path.parts and path.parts[0]=='framework'
        if item.isdir():continue
        assert item.isfile() and item.name in manifest['files'] and item.name not in seen
        row=manifest['files'][item.name];data=archive.extractfile(item).read()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['SHA256']
        dest=ROOT/item.name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(data)
        dest.chmod(row['mode']);seen.add(item.name)
assert seen==set(manifest['files'])
for name,row in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==row['SHA256']
for name,row in HELPERS.items():
    assert name=='verify_entry.py';data=base64.b64decode(row['b64'],validate=True);assert hashlib.sha256(data).hexdigest()==row['SHA256']
    with (ROOT/'helpers'/name).open('xb') as f:f.write(data)
shell=('set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
       'export FRAMEWORK_EPOCH='+shlex.quote(str(epoch))+' FRAMEWORK_DEADLINE='+shlex.quote(str(deadline))+'; exec python3 -B /stage-tools/verify_entry.py')
args=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/','--bind',str(ROOT/'framework')+':/shared-training:ro','--bind',str(ROOT/'helpers')+':/stage-tools:ro','--bind',str(ROOT/'evidence')+':/evidence:rw',str(R/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',shell]
r=subprocess.run(args,capture_output=True,timeout=max(1,deadline-time.time()-5))
with (ROOT/'evidence/entry.stdout').open('xb') as f:f.write(r.stdout)
with (ROOT/'evidence/entry.stderr').open('xb') as f:f.write(r.stderr)
out={'claim_commit':CLAIM,'epoch':epoch,'deadline':deadline,'framework_commit':manifest['framework_commit'],'framework_tree':manifest['framework_tree'],'manifest_SHA256':MANIFEST_SHA,'files_verified':len(seen),'bytes_verified':sum(x['bytes'] for x in manifest['files'].values()),'entry_returncode':r.returncode,'stdout_SHA256':hashlib.sha256(r.stdout).hexdigest(),'stderr_SHA256':hashlib.sha256(r.stderr).hexdigest(),'actual_parent_state':parent[0],'actual_nodes':nodes,'GPU_allocation':0}
if (ROOT/'evidence/entry-verification.json').is_file():
    data=(ROOT/'evidence/entry-verification.json').read_bytes();out['entry_verification']={'SHA256':hashlib.sha256(data).hexdigest(),'value':json.loads(data)}
else:out['error_tail']=r.stderr.decode(errors='replace')[-1000:]
with (ROOT/'receipt.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
