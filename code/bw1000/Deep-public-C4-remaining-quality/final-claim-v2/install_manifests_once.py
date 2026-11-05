"""One explicit7 childmanifest install; noSlurm/HTTP/Native or retry."""
from pathlib import Path
import argparse,hashlib,json,os,socket,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');F=Path(__file__).resolve().parent;D=F.parent/'deployment-v3';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--claim-sha',required=True);a.add_argument('--publication-commit',required=True);args=a.parse_args();assert H(F/'GPU_RESOURCE_CLAIM.json')==args.claim_sha and len(args.publication_commit)==40
 claim=json.loads((F/'GPU_RESOURCE_CLAIM.json').read_bytes());assert claim['cards']==56 and claim['HTTP']==448 and len(claim['shards'])==7
 assert H(D/'SOURCE_MANIFEST.json')==claim['preexecution_source_MF'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in m['files_SHA256'].items():assert H(D/name)==h,name
 ref=claim['CPU_receipt'];p=R/Path(ref['path']).relative_to('/refactor');assert H(p)==ref['SHA256'];q=json.loads(p.read_bytes());assert q['actual_CPU']=='910407.1034'and q['passed']and q['completed']and len(q['shards'])==7 and q['source_manifest_SHA256']==claim['CPU_receipt_source_MF']and q['project_maps']==[]
 assert not any(q[k]for k in['GPU','Native_dlopen','SDK','weights','new_compile','reencode'])
 host=claim['actual_HOST_preflight'];p=Path(host['path']);assert H(p)==host['SHA256'];hq=json.loads(p.read_bytes());assert hq['passed']and hq['source_MF']==claim['preexecution_source_MF']and len(hq['shards'])==7 and all(r['passed']and r['rc']==0 and r['proof']['Slurm_submitted']==0 and not r['proof']['intent_created']for r in hq['shards'])
 rows=subprocess.check_output(['sacct','-S','2026-11-05','-j','910374,910376,910407','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode'],text=True,timeout=30).splitlines()
 actual_CPU=[r.split('|')for r in rows if'|'in r and r.split('|')[0]=='910407.1034'];assert len(actual_CPU)==1 and actual_CPU[0][1:4]==['Deep-public-C4-remaining-quality-CPU-v2','COMPLETED','0:0'],'OriginalCPU accounting must be normal'
 for item in claim['shards']:
  slot=item['shard'];target=D/item['target'];assert H(target)==item['original_false_SHA256']and H(F/item['final_name'])==item['final_SHA256']
  final=json.loads((F/item['final_name']).read_bytes());old=json.loads(target.read_bytes());assert final['files_SHA256']==old['files_SHA256']and not old['GPU_GO']and final['GPU_GO']and final['resource_claim']['GPU_GO']
  ref=item['private_plan'];p=R/Path(ref['path']).relative_to('/refactor');assert H(p)==ref['SHA256'];assert q['shards'][slot]['private_SHA256']==ref['SHA256']
  scope=old['scope'];prefix=scope['label']+'-'+scope['run'];assert not any('|'in r and r.split('|')[1].startswith(prefix)for r in rows),'ExistingSlurmhandle followoriginal'
  for suffix in['-intent.json','-launch.json','-watch.json']:assert not(R/'results'/(prefix+suffix)).exists(),'Existingorunknown originalstate'
  assert not(R/'results/Deep-public-fixed-cohort-r465'/scope['run']).exists();assert not list((R/'logs').glob(prefix+'*'))
 intent=F/'manifest-install-intent.json';result=F/'manifest-install-result.json';assert not intent.exists()and not result.exists()
 with intent.open('x')as f:json.dump(dict(origin_HOST=socket.gethostname(),PID=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],publication_commit=args.publication_commit,claim_SHA256=args.claim_sha,new_jobs=0),f)
 for item in claim['shards']:
  target=D/item['target'];raw=target.read_bytes();backup=F/('original-false-shard%d.json'%item['shard'])
  with backup.open('xb')as f:f.write(raw)
  temp=target.with_name('SOURCE_MANIFEST.execution-tmp.json')
  with temp.open('xb')as f:f.write((F/item['final_name']).read_bytes());f.flush();os.fsync(f.fileno())
  assert H(target)==item['original_false_SHA256'];os.replace(temp,target)
  assert H(target)==item['final_SHA256']
 out=dict(completed=True,new_jobs=0,new_HTTP=0,shards=7,publication_commit=args.publication_commit,claim_SHA256=args.claim_sha,top_CPU_manifest_preserved=H(D/'SOURCE_MANIFEST.json')==claim['preexecution_source_MF'],GPU_not_launched=True)
 with result.open('x')as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
