"""Consume the single CPU-only16marker execution; no producer imports."""
import argparse,base64,gzip,hashlib,json,statistics
from pathlib import Path
H=lambda b:hashlib.sha256(b).hexdigest()
def audit(p):
 b=Path(p).read_bytes();assert H(b)=='db75ed304caf1d26c04aabb26b10faaeee9a9a2d0f3ffd44ed6bc0476ad9f636';a=json.loads(gzip.decompress(b));f={}
 for v in a['files']:
  d=gzip.decompress(base64.b64decode(v['gzip_base64'],validate=True));assert len(d)==v['bytes']and H(d)==v['SHA256']and v['path']not in f;f[v['path']]=d
 assert len(f)==62 and sum(map(len,f.values()))==98006
 R='/public/home/aclsylqidf/hygon-inference-refactor-20261014';D=R+'/incoming/GLM952-global-TP16-component-r470/barrier-CPU-v1/';O=R+'/results/GLM-r470-marker-barrier-CPU-v1/';j=lambda p:json.loads(f[p]);assert H(f[D+'SOURCE_MANIFEST.json'])=='9bfff936085d1d225c3c7bbca1e685f0368c76d79f2cda55eb13c5583943f49c'
 for n,h in j(D+'SOURCE_MANIFEST.json')['files_SHA256'].items():assert H(f[D+n])==h
 assert H(f[D+'CPU_RESOURCE_CLAIM.json'])=='afd00159312ef5a05d80253cc1532cf11f7b913fd34bac89ca166ec00c114622';assert H(f[D.rsplit('barrier-CPU-v1/',1)[0]+'device-v1/shared.py'])=='4c14563023e21292dcadad5402a5d68c9f2c5551551f6db43e2c6aad665b872c'
 assert a['accounting']['rc']==0 and '910371.872|GLM-r470-marker-barrier-CPU-v1|COMPLETED|0:0|33' in a['accounting']['stdout']
 owners=[j(O+'owner-%02d.json'%i)for i in range(16)];workers=[j(O+'worker-%d.json'%i)for i in range(2)];epoch=1793893499.0150878;result=[]
 for node,w in enumerate(workers):
  assert w['passed']and w['completed']and w['actual_CPU']=='910371.872'and w['epoch']==epoch and len(w['owners'])==8 and not w.get('forced_TERM')and not w.get('forced_KILL')
  for v in w['owners']:
   i=v['rank'];r=owners[i];assert r['rank']==i==node*8+r['device']and r['PID']==v['PID']and r['startticks']==v['startticks']and r['epoch']==epoch and r['actual_CPU']=='910371.872'and r['node']==w['node']and r['completed']and r['passed']and r['phase']=='all16_read'and r['CPU_only']and not r['SDK_initialized']and not r['GPU'];assert r['source_shared_SHA256']=='4c14563023e21292dcadad5402a5d68c9f2c5551551f6db43e2c6aad665b872c';assert len(r['waits'])==16
   waits=[]
   for k,x in enumerate(r['waits']):
    assert x['rank']==k and x['path']=='/refactor/results/GLM-r470-marker-barrier-CPU-v1/created-%02d.json'%k and x['exists_after']and x['seconds']>=0
    raw=f[O+'created-%02d.json'%k];assert len(raw)==x['bytes']and H(raw)==x['SHA256']and json.loads(raw)==dict(global_rank=k,epoch=epoch);waits.append(x['seconds'])
   assert f[O+'owner-%02d.log'%i]==b'';result.append(dict(rank=i,node=r['node'],save_seconds=r['save_seconds'],wait_total_seconds=sum(waits),wait_rank0_seconds=waits[0],largest_wait_seconds=max(waits)))
 return dict(audit_passed=True,actual_CPU='910371.872',state='COMPLETED0',seconds=33,archive=dict(path=Path(p).name,SHA256=H(b),files=62,decoded_bytes=98006),source_manifest_SHA256=H(f[D+'SOURCE_MANIFEST.json']),owners=16,wait_calls=256,normal_workers=2,all16_read=True,per_rank=result,node1_wait_rank0_range_seconds=[min(x['wait_rank0_seconds']for x in result[8:]),max(x['wait_rank0_seconds']for x in result[8:])],node0_total_wait_range_seconds=[min(x['wait_total_seconds']for x in result[:8]),max(x['wait_total_seconds']for x in result[:8])],no_SDK_Runtime_GPU=True,old870_qualification=False,causal_boundary='Current identical save/wait control path succeeds but has node1->rank0 wait30.043..30.446s. This includes retries/read/scheduling; no errno or kernel cache trace. Not a historical870 NFS diagnosis or evidence that collective was healthy. Does motivate avoiding shared-file polling as a crossnode readiness gate.')
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);b=(json.dumps(q,indent=2)+'\n').encode();Path(a.output).write_bytes(b);print(json.dumps(dict(passed=True,actual_CPU=q['actual_CPU'],owners=16,output_SHA256=H(b))))
if __name__=='__main__':main()
