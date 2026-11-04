from pathlib import Path
import base64,datetime,gzip,hashlib,json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');RUN='public-kimi-wide-20261104-a';LABEL='Kimi-r448-wide-quality-'+RUN
ARCHIVE=R/'results/Kimi448-585-terminal-sealed-v1.json.gz';assert not ARCHIVE.exists(),'Append-only archive exists; consume it, never recapture or rerun'
files={};total=0;H=lambda b:hashlib.sha256(b).hexdigest()
def add(name,path):
 global total
 path=Path(path);assert name not in files and path.is_file(),(name,str(path));raw=path.read_bytes();assert len(raw)<=512*1024**2,(name,len(raw));total+=len(raw);assert total<=1024**3
 files[name]=dict(path=str(path),bytes=len(raw),SHA256=H(raw),base64=base64.b64encode(raw).decode())
def add_tree(name,root):
 root=Path(root);add(name+'/SOURCE_MANIFEST.json',root/'SOURCE_MANIFEST.json');m=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
 for relative,digest in m['files_SHA256'].items():
  p=root/relative;assert p.resolve().is_relative_to(root.resolve())and H(p.read_bytes())==digest,(name,relative);add(name+'/'+relative,p)
watch=R/'results'/(LABEL+'-watch.json');w=json.loads(watch.read_bytes());assert w['completed']and w['passed']and w['actual_GPU']=='910380.585'and w['CPU_keeper']=='910407.944'
add('watch.json',watch)
client=R/'results'/(LABEL+'-client');add('client/result.json',client/'result.json');c=json.loads((client/'result.json').read_bytes());assert c['completed']and c['execution_passed']and c['completed_HTTP']==256
for p in sorted(client.rglob('*')):
 if p.is_file()and p.name!='result.json':
  assert p.suffix in('.json','.wire'),str(p);add('client/'+p.relative_to(client).as_posix(),p)
identity=client.with_name(client.name+'-identity.json');add('client-identity.json',identity)
runtime=R/'results/public-Kimi-text-provider-r434'
for p in sorted(runtime.glob('*'+RUN+'*')):
 if p.is_file():
  assert p.suffix in('.json','.jsonl','.txt'),str(p);add('runtime/'+p.name,p)
add('inputs/cases.json',R/'incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/cases.json')
add('inputs/original-requests-and-pairs.json',R/'incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/requests-and-pairs.json')
B=R/'candidates/Kimi-wide-fewshot-quality-r448-CPU-v1'
add('inputs/encoding.json',B/'encoding.json');add('inputs/requests-private.json',B/'requests-private.json')
enc=json.loads((B/'encoding.json').read_bytes())
for row in enc['cases']:
 p=Path(row['private_path']);p=R/str(p.relative_to('/refactor'))if str(p).startswith('/refactor/')else p
 assert p.resolve().is_relative_to(B.resolve())and H(p.read_bytes())==row['private_SHA256'];add('inputs/private/'+row['case_id']+'.json',p)
for ident,path in [('CPU942','Kimi-wide-fewshot-quality-r448-client-CPU-v1/CPU-result.json'),('CPU943','Kimi-wide-fewshot-quality-r448-client-CPU-repair-v2/CPU-result.json')]:add(ident+'.json',R/'candidates'/path)
for name,path in [('source448',R/'incoming/Kimi-wide-fewshot-quality-r448'),('deployment',R/'incoming/Kimi-wide-fewshot-quality-r448/client-v1'),('final',R/'incoming/Kimi-wide-fewshot-quality-r448/client-v1/gpu-final-v1'),('provider434',R/'incoming/Kimi-public-incremental-text-r434'),('runtime428',R/'incoming/public-Kimi-native-provider-r428/runtime-v2'),('public425',R/'incoming/public-GLM-native-provider-r425')]:add_tree(name,path)
add('final/GPU_MANIFEST.json',R/'incoming/Kimi-wide-fewshot-quality-r448/client-v1/gpu-final-v1/GPU_MANIFEST.json')
for p in sorted((R/'logs').glob(LABEL+'*')):
 if p.is_file():add('logs/'+p.name,p)
account=subprocess.check_output(['sacct','-S','2026-11-04','-j','910380,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30)
rows=[s.split('|')for s in account.splitlines()if'|'in s and s.split('|')[1].startswith(LABEL)]
assert {r[0]for r in rows}=={'910380.584','910380.585','910380.586','910380.587','910407.944'}and all(r[2:4]==['COMPLETED','0:0']for r in rows)
q=dict(schema='Kimi448-585-original-terminal-seal-v1',run=RUN,actual_GPU='910380.585',actual_client='910380.586',actual_keeper='910407.944',post='910380.587',sealed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),accounting=rows,new_jobs=0,new_HTTP=0,raw_input_output_private=True,files=files)
raw=(json.dumps(q,ensure_ascii=False,separators=(',',':'))+'\n').encode();packed=gzip.compress(raw,compresslevel=9,mtime=0)
with ARCHIVE.open('xb')as f:f.write(packed)
print(json.dumps(dict(archive_path=str(ARCHIVE),archive_bytes=len(packed),archive_SHA256=H(packed),uncompressed_bytes=len(raw),uncompressed_SHA256=H(raw),file_count=len(files),raw_file_bytes=total,archive_base64=base64.b64encode(packed).decode(),no_new_jobs=True)))
