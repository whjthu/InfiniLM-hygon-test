"""Offline verification; never execute imported experimental source."""
import hashlib,json,pathlib,tarfile
r=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'manifests/bw1000/recent-source-files.json').read_text(encoding='utf8'))
for x in m['files']:
 p=r/x['path'];assert p.is_file()and p.stat().st_size==x['bytes']and sha(p)==x['SHA256'],x['path']
proof=json.loads((r/'manifests/bw1000/profile-evidence.json').read_text(encoding='utf8'))
for x in proof['files']:
 assert sha(r/x['path'])==x['SHA256'],x['path']
a=json.loads((r/'archives/bw1000/source-archive-manifest.json').read_text(encoding='utf8'));p=r/'archives/bw1000'/a['archive'];assert sha(p)==a['SHA256']and p.stat().st_size==a['compressed_bytes']
with tarfile.open(p)as t:
 members=t.getmembers();assert len(members)==a['files']
 for x in members:assert x.isfile()and not x.name.startswith('/')and '..'not in pathlib.PurePosixPath(x.name).parts,x.name
results=list((r/'results').rglob('*.json'))
for p in results:
 d=json.loads(p.read_text(encoding='utf8'));assert d['hardware']=='BW1000'and d['level']and d['status']
for p in(r/'coordination/tasks').glob('*.json'):
 d=json.loads(p.read_text(encoding='utf8'));assert d['owner']and d['state']and d['work_item']==p.stem
for p in r.rglob('*'):
 if p.is_file()and '.git'not in p.parts:
  assert p.suffix not in ['.so','.o','.a','.safetensors','.gguf','.pem','.key'],p
print(json.dumps({'passed':True,'recent_source_files':len(m['files']),'historical_archive_files':a['files'],'result_files':len(results),'scope':'Offline integrity/structure only; no GPU/model/online performance claim.'}))
