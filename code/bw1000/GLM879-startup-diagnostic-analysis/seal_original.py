"""Terminal-only original metadata archive. No Slurm submission or signals."""
from pathlib import Path
import base64,datetime,hashlib,io,json,subprocess,tarfile
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
label='GLM-r444-startup-diagnostic-v4-startupdiag-20261104-v4-s0'
folder=R/'results/GLM-W4-fewshot-quality-r444/startupdiag-20261104-v4-s0'
out=R/'results/GLM879-original-terminal-v1.tar.gz'
if out.exists():
    blob=out.read_bytes();print(json.dumps(dict(path=str(out),bytes=len(blob),SHA256=hashlib.sha256(blob).hexdigest(),existing_immutable_archive=True,base64=base64.b64encode(blob).decode())));raise SystemExit(0)
watch=R/'results'/(label+'-watch.json');w=json.loads(watch.read_bytes())
assert w['completed']and not w['passed']and w['actual_GPU']=='910376.879'and w['CPU_keeper']=='910407.941'and w['actual_client']is None
assert w['startup_STOP_drain']['normal_GPU_STOP_drain']and not w['forced_cleanup']
items={}
def add(name,path):
    raw=path.read_bytes();assert len(raw)<=16*1024**2;items[name]=raw
add('watch.json',watch);add('intent.json',R/'results'/(label+'-intent.json'))
add('startup-snapshot.json',Path(w['startup_original_snapshot']['path']))
for path in sorted(folder.iterdir()):
    if path.is_file()and path.suffix in ['.json','.jsonl','.txt']and'.writing-'not in path.name:add('run/'+path.name,path)
for path in sorted((R/'logs').glob(label+'*')):
    if not path.is_file():continue
    with path.open('rb')as f:f.seek(0,2);f.seek(max(0,f.tell()-16384));items['log-tail/'+path.name]=f.read(16384)
a=subprocess.run(['sacct','-S','2026-11-03','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,ElapsedRaw,NodeList%100'],capture_output=True,text=True,timeout=30);assert a.returncode==0
rows=[line.split('|')for line in a.stdout.splitlines()if label in line]
accounts={r[0]:r for r in rows};assert accounts['910376.879'][2:4]==['COMPLETED','0:0']and accounts['910407.941'][2:4]==['FAILED','1:0']and accounts['910376.880'][2:4]==['COMPLETED','0:0']
items['accounting.json']=(json.dumps(dict(argv=a.args,rows=rows),indent=2)+'\n').encode()
assert sum(map(len,items.values()))<=256*1024**2
manifest=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Original879 zero-request startup timeout, real Python stacks, normal GPU STOP and failure keeper; no weights or activations',files={n:dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())for n,raw in items.items()})
items['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode();memory=io.BytesIO()
with tarfile.open(fileobj=memory,mode='w:gz')as tf:
    for name,raw in items.items():
        info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0;tf.addfile(info,io.BytesIO(raw))
blob=memory.getvalue()
with out.open('xb')as f:f.write(blob)
print(json.dumps(dict(path=str(out),bytes=len(blob),SHA256=hashlib.sha256(blob).hexdigest(),members=len(items),base64=base64.b64encode(blob).decode())))
