"""Exact original7 terminal seals only; oldpilot never recollected; nojob/HTTP."""
from pathlib import Path
import argparse,base64,datetime,gzip,hashlib,json,os,socket,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');H=lambda b:hashlib.sha256(b).hexdigest()
BINDINGS=[{'shard': 0, 'GPU': '910374.1478', 'client': '910374.1484', 'keeper': '910407.1036', 'post': '910374.1490', 'node': 'a02r4n01', 'epoch': 1793863417.6511638, 'config': 'd3157a21d4c2dff11a428c7622204b1277d63fdb867c407f1079d1912c2222a9', 'prefix': 'Deep-public-C4-remaining-quality-remaining-c4-20261105-b-s0'}, {'shard': 1, 'GPU': '910374.1480', 'client': '910374.1486', 'keeper': '910407.1037', 'post': '910374.1488', 'node': 'a02r4n05', 'epoch': 1793863420.3360229, 'config': '6c5a336860d0c3d230964f3758349cd2808711590055e79e8694aa7a5d135779', 'prefix': 'Deep-public-C4-remaining-quality-remaining-c4-20261105-b-s1'}, {'shard': 2, 'GPU': '910374.1482', 'client': '910374.1487', 'keeper': '910407.1038', 'post': '910374.1489', 'node': 'a02r4n07', 'epoch': 1793863422.2359815, 'config': 'f3f3dfad9dbb9c21b0b22d0bcd893b69466ad3ea2090c170ac94e7761a5a4138', 'prefix': 'Deep-public-C4-remaining-quality-remaining-c4-20261105-b-s2'}, {'shard': 3, 'GPU': '910376.940', 'client': '910376.951', 'keeper': '910407.1039', 'post': '910376.953', 'node': 'a02r4n12', 'epoch': 1793863427.1743076, 'config': 'b923e3a0378fd796eae386650301c53654b30a9e9e587c741c3735f4bf943cb2', 'prefix': 'Deep-public-C4-remaining-quality-remaining-c4-20261105-b-s3'}, {'shard': 4, 'GPU': '910376.944', 'client': '910376.949', 'keeper': '910407.1040', 'post': '910376.952', 'node': 'a02r4n13', 'epoch': 1793863433.3619704, 'config': '626229cbac525cb29835850856f3dcbdd5667da0280bb9d5d9a16e778d1ff8cf', 'prefix': 'Deep-public-C4-remaining-quality-remaining-c4-20261105-b-s4'}, {'shard': 5, 'GPU': '910376.943', 'client': '910376.950', 'keeper': '910407.1041', 'post': '910376.954', 'node': 'a02r4n14', 'epoch': 1793863437.27808, 'config': '39645ddfc5883650c0948fff850ba1de8b54ad687c98dee26cef8a08953f656a', 'prefix': 'Deep-public-C4-remaining-quality-remaining-c4-20261105-b-s5'}, {'shard': 6, 'GPU': '910376.946', 'client': '910376.947', 'keeper': '910407.1042', 'post': '910376.955', 'node': 'a02r4n15', 'epoch': 1793863444.3030794, 'config': '132da99ea14018a5f40d810dc1f1821fe8d47f47da0bb807c33f2ac89c5a106d', 'prefix': 'Deep-public-C4-remaining-quality-remaining-c4-20261105-b-s6'}]
def collect(j):
 b=BINDINGS[j];prefix=b['prefix'];target=R/'results'/('Deep-C4-remaining-quality-shard%d-terminal-original-sealed-v1.json.gz'%j);intent=target.with_suffix(target.suffix+'.collection-intent.json')
 if target.exists():
  raw=target.read_bytes();q=json.loads(gzip.decompress(raw));assert q['binding']==b and q['new_jobs']==q['new_HTTP']==0;return dict(shard=j,path=str(target),bytes=len(raw),SHA256=H(raw),existing=True,files=len(q['files']))
 assert not intent.exists(),'Originalsealunknown/inprogress; reconcilePID/startticks; never reseal'
 watch=R/'results'/(prefix+'-watch.json');w=json.loads(watch.read_bytes());assert w['completed']and w['passed']and w['execution_passed']and not w['forced_cleanup']and not w.get('error');assert(w['actual_GPU'],w['actual_client'],w['CPU_keeper'],w['original_GPU_start_unix'])==(b['GPU'],b['client'],b['keeper'],b['epoch'])
 z=subprocess.run(['sacct','-S','2026-11-05','-j','910374,910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,ElapsedRaw,NodeList%100'],capture_output=True,text=True,timeout=30);assert z.returncode==0;rows=[s for s in z.stdout.splitlines()if'|'in s and s.split('|')[1].startswith(prefix)]
 for step in[b['GPU'],b['client'],b['keeper'],b['post']]:assert len([s for s in rows if s.split('|')[0]==step and s.split('|')[2:4]==['COMPLETED','0:0']])==1,step
 who=dict(origin_HOST=socket.gethostname(),PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_jobs=0,new_HTTP=0)
 with intent.open('x')as f:json.dump(who,f);f.flush();os.fsync(f.fileno())
 O=R/'results/Deep-public-fixed-cohort-r465'/('remaining-c4-20261105-b-s'+str(j));C=R/'results'/(prefix+'-client');B=R/'incoming/Deep-public-C4-remaining-quality';F=B/'final-claim-v2';D=B/'deployment-v3';P=R/'incoming/Deep-public-fixed-cohort-profile-r465/source-v1'
 paths={p for folder in[O,C]for p in folder.rglob('*')if p.is_file()};paths|={watch,intent,R/'results'/(prefix+'-intent.json'),C.with_name(C.name+'-identity.json')};paths|={p for p in(R/'logs').glob(prefix+'*.log')if p.is_file()}
 assert H((D/'SOURCE_MANIFEST.json').read_bytes())=='95af7817339c78db12142501e536515d79199c3288ba1baf1d8492693e1462bf';cpu=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());paths.add(D/'SOURCE_MANIFEST.json')
 final=json.loads((F/'GPU_RESOURCE_CLAIM.json').read_bytes());assert H((F/'GPU_RESOURCE_CLAIM.json').read_bytes())=='cc6f3f7d8ceb0e1f8c3dcd5232427a882b96a1b9f771b2a4b0626753cc1e6d79';paths|={p for p in F.iterdir()if p.is_file()}
 for name,h in cpu['files_SHA256'].items():
  p=D/name
  if name.endswith('/SOURCE_MANIFEST.json'):
   k=int(name.split('/')[0][5:]);backup=F/('original-false-shard%d.json'%k);assert H(backup.read_bytes())==h;assert H(p.read_bytes())==final['shards'][k]['final_SHA256'];paths.add(backup)
  else:assert H(p.read_bytes())==h,name
  paths.add(p)
 assert H((P/'SOURCE_MANIFEST.json').read_bytes())=='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec';pm=json.loads((P/'SOURCE_MANIFEST.json').read_bytes());paths.add(P/'SOURCE_MANIFEST.json')
 for name,h in pm['files_SHA256'].items():p=P/name;assert H(p.read_bytes())==h;paths.add(p)
 for folder,h in pm['parents_manifest_SHA256'].items():p=R/Path(folder).relative_to('/refactor')/'SOURCE_MANIFEST.json';assert H(p.read_bytes())==h;paths.add(p)
 plan=R/'candidates/Deep-public-C4-remaining-quality-CPU-v2'/('shard%d-private.json'%j);assert H(plan.read_bytes())==final['shards'][j]['private_plan']['SHA256'];private=json.loads(plan.read_bytes());paths.add(plan)
 encpath=R/'candidates/Deep-broad-fewshot-fullpre-quality-r450-CPU-v1/encoding.json';assert H(encpath.read_bytes())=='04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a';enc=json.loads(encpath.read_bytes());paths|={encpath,R/'incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/cases.json',R/'candidates/Deep-public-C4-remaining-quality-CPU-v2/result.json',R/'results/Deep-public-C4-remaining-quality-HOST-v3-preflight-result.json'}
 for ordinal in private['ordinals']:paths.add(R/Path(enc['cases'][ordinal]['private_path']).relative_to('/refactor'))
 files=[];total=0
 for p in sorted(paths):
  assert p.is_file()and not p.is_symlink();raw=p.read_bytes();total+=len(raw);assert len(raw)<512*1024**2 and total<1024*1024**2
  files.append(dict(path=str(p),bytes=len(raw),SHA256=H(raw),gzip_base64=base64.b64encode(gzip.compress(raw,compresslevel=1,mtime=0)).decode()))
 seal=dict(schema='Deep-C4-remaining-original-terminal-v1',binding=b,origin=who,files=files,accounting=dict(rc=z.returncode,rows=rows,stderr=z.stderr),new_jobs=0,new_HTTP=0,private_not_Git=True)
 raw=gzip.compress(json.dumps(seal,separators=(',',':')).encode(),compresslevel=3,mtime=0)
 with target.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(shard=j,path=str(target),bytes=len(raw),SHA256=H(raw),files=len(files),decoded_file_bytes=total,existing=False)
def main():
 results=[]
 for j in range(7):
  try:results.append(collect(j))
  except BaseException as e:results.append(dict(shard=j,error=repr(e),preserve_original_intent=True));break
 print(json.dumps(dict(new_jobs=0,new_HTTP=0,seals=results)))
if __name__=='__main__':main()
