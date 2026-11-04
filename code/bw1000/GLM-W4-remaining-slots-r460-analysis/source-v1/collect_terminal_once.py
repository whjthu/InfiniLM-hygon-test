"""Single read-only terminal seal per original slot1/6; no loop/job/control."""
from pathlib import Path
import argparse,base64,datetime,gzip,hashlib,json,os,socket,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
BINDINGS={'0': {'slot': 0, 'GPU': '910376.886', 'client': '910376.887', 'keeper': '910407.950', 'epoch': 1793755914.8956137, 'first_ready': 1793756804.8594055, 'run': 'verifiedready-20261104-socket-v1-s0', 'label': 'GLM-r444-verified-ready-socket-v1', 'nodes': ['a02r4n14', 'a02r4n15'], 'deployment': '56133dff89d8579b070f64d31f1adad2aa0582cea0cd9168ed164065162df6d9', 'legacy_identity_not_archived': True}, '1': {'slot': 1, 'GPU': '910376.894', 'client': '910376.897', 'keeper': '910407.969', 'epoch': 1793769182.9887497, 'first_ready': 1793769997.6175935, 'run': 'remaining-20261104-a-s1', 'label': 'GLM-r460-remaining', 'nodes': ['a02r4n14', 'a02r4n15'], 'deployment': 'db0166fbab6d5562984957874d9def544ab5aa0bcefa84eedfc1bcf8f0eb3f06', 'legacy_identity_not_archived': False, 'config_SHA256': '30dcd3ca566d51853ce48ea213626d055ba4182b6a69fd3e57a37a8376a0138f', 'client_source_SHA256': '5952de95257088d02bad7e732f914aec1df09cce4383d5ec6053a33fa30399a1', 'GPU_manifest_SHA256': '1848614c09796409fc60cc6b53b4914a0a55d089bcd713ef5a1d246c93ec6312'}, '6': {'slot': 6, 'GPU': '910376.896', 'client': '910376.898', 'keeper': '910407.970', 'epoch': 1793769192.9743097, 'first_ready': 1793770203.711196, 'run': 'remaining-20261104-a-s6', 'label': 'GLM-r460-remaining', 'nodes': ['a02r4n12', 'a02r4n13'], 'deployment': 'db0166fbab6d5562984957874d9def544ab5aa0bcefa84eedfc1bcf8f0eb3f06', 'legacy_identity_not_archived': False, 'config_SHA256': 'bd326f8e0f337a0e7eb9b7c19950cfe147f852d75dd60a7d4c5d8fe9737fecaf', 'client_source_SHA256': '5952de95257088d02bad7e732f914aec1df09cce4383d5ec6053a33fa30399a1', 'GPU_manifest_SHA256': '1848614c09796409fc60cc6b53b4914a0a55d089bcd713ef5a1d246c93ec6312'}}
DEPLOY=R/'incoming/GLM-W4-remaining-slots-r460/deployment-v1'
GPU_MF='1848614c09796409fc60cc6b53b4914a0a55d089bcd713ef5a1d246c93ec6312'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def collect(slot):
 b=BINDINGS[str(slot)];prefix=b['label']+'-'+b['run'];watch=R/'results'/(prefix+'-watch.json');w=json.loads(watch.read_bytes())
 if not w['completed']:return dict(slot=slot,collected=False,reason='Original watch live: no sleep, accounting query, owner reads or newpoller',actual_GPU=w['actual_GPU'],actual_client=w['actual_client'],actual_keeper=w['CPU_keeper'])
 assert w['slot']==slot and w['resource_claim']['run']==b['run']and w['deployment_source_SHA256']==b['deployment']and(w['actual_GPU'],w['actual_client'],w['CPU_keeper'])==(b['GPU'],b['client'],b['keeper'])
 output=R/'results'/('GLM460-slot%d-terminal-original-sealed-v1.json.gz'%slot)
 if output.exists():
  raw=output.read_bytes();seal=json.loads(gzip.decompress(raw));assert seal['slot']==slot and seal['read_only']and seal['new_jobs']==0
  return dict(slot=slot,collected=True,existing_original=True,path=str(output),bytes=len(raw),SHA256=sha(raw),files=len(seal['files']),slot_index=seal['slot_index'])
 argv=['sacct','-S','2026-11-04','-j','910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,ElapsedRaw,NodeList%100'];p=subprocess.run(argv,capture_output=True,text=True,timeout=30);assert p.returncode==0
 rows=[line for line in p.stdout.splitlines()if'|'in line and line.split('|')[1].startswith(prefix)]
 for step in[b['GPU'],b['client'],b['keeper']]:
  matches=[row.split('|')for row in rows if row.split('|')[0]==step]
  if len(matches)!=1 or matches[0][2].split()[0].rstrip('+')not in TERMINAL:return dict(slot=slot,collected=False,reason='Original accounting not yet terminal; no wait/retry',rows=rows)
 intent=output.with_suffix(output.suffix+'.collection-intent.json')
 if intent.exists():return dict(slot=slot,collected=False,reason='Original collection already started but final archive not present; reconcile exact original PID/startticks, do not reseal',intent=str(intent))
 with intent.open('x')as stream:json.dump(dict(slot=slot,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),origin_HOST=socket.gethostname(),PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],new_jobs=0,new_requests=0),stream)
 folder=R/'results/GLM-W4-fewshot-quality-r444'/b['run'];client=R/'results'/(prefix+'-client');identity=R/'results'/(prefix+'-client-identity.json');files={}
 def add(path):
  raw=path.read_bytes();files[str(path)]=dict(bytes=len(raw),SHA256=sha(raw),gzip_base64=base64.b64encode(gzip.compress(raw,compresslevel=1,mtime=0)).decode())
 add(watch)
 for root in[folder,client]:
  for path in sorted(root.rglob('*')):
   if path.is_file()and path.suffix in['.json','.jsonl','.wire']:add(path)
 for path in[identity,R/'results'/(prefix+'-intent.json'),DEPLOY/'SOURCE_MANIFEST.json',DEPLOY/'GPU_MANIFEST-initial-two.json']:
  if path.is_file():add(path)
 manifest=json.loads((DEPLOY/'SOURCE_MANIFEST.json').read_bytes());assert sha((DEPLOY/'SOURCE_MANIFEST.json').read_bytes())==b['deployment']and sha((DEPLOY/'GPU_MANIFEST-initial-two.json').read_bytes())==GPU_MF
 for name,digest in manifest['files_SHA256'].items():
  path=DEPLOY/name;assert sha(path.read_bytes())==digest;add(path)
 for foldername in['GLM444-verified-ready-socket-CPU-v1/provider','GLM-W4-fewshot-quality-r444-client-v1/client-v1']:
  path=R/'incoming'/foldername/'SOURCE_MANIFEST.json';add(path)
 accounting_path=str(R/'results'/('GLM460-slot%d-terminal-sacct-original.txt'%slot));raw=(chr(10).join(rows)+chr(10)).encode();files[accounting_path]=dict(bytes=len(raw),SHA256=sha(raw),base64=base64.b64encode(raw).decode())
 index=dict(slot=slot,watch_path=str(watch),client_path=str(client/'result.json'),run_directory=str(folder),accounting_path=accounting_path,client_identity_path=str(identity))
 seal=dict(schema='GLM460-original-terminal-only-v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),slot=slot,read_only=True,new_jobs=0,new_requests=0,original_sacct_command=argv,slot_index=index,files=files,identity_missing=not identity.exists(),raw_weights_or_activations=False)
 raw=gzip.compress((json.dumps(seal,ensure_ascii=False)+chr(10)).encode(),compresslevel=3,mtime=0)
 with output.open('xb')as stream:stream.write(raw)
 return dict(slot=slot,collected=True,existing_original=False,path=str(output),bytes=len(raw),SHA256=sha(raw),files=len(files),slot_index=index)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--slot',type=int,choices=[1,6],action='append');args=parser.parse_args();slots=args.slot or[1,6];assert len(set(slots))==len(slots)
 results=[]
 for slot in slots:
  try:results.append(collect(slot))
  except Exception as error:results.append(dict(slot=slot,collected=False,error=repr(error),reason='Preserve original collection/intent; diagnose only, no retry or newjobs'))
 print(json.dumps(dict(read_only=True,new_jobs=0,new_requests=0,slots=results)))
if __name__=='__main__':main()
