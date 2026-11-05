"""Read only original637 metadata. No producer imports, jobs, Native or SDK."""
import argparse,hashlib,json,tarfile
from pathlib import Path
H=lambda b:hashlib.sha256(b).hexdigest()
def read(archive):
 raw=Path(archive).read_bytes();assert H(raw)=='78206dc20170c0afcdf59d0b05d77eaadca2c28da021466d6c7a2e9732a6c27b'
 label='Kimi-r461-full-model-tp32-full-cohort-20261105-c';files={}
 with tarfile.open(archive)as t:
  for m in t:
   assert m.isfile()and not m.name.startswith('/')and '..'not in Path(m.name).parts
   b=t.extractfile(m).read();files[m.name]=b
 assert len(files)==31
 watch=json.loads(files['results/'+label+'-watch.json']);assert watch['completed']and not watch['passed']and watch['forced_cleanup']
 assert watch['actual_GPU']=='910380.637'and watch['CPU_keeper']=='910407.1047'
 out=[]
 for n in range(4):
  p='results/'+label+'/node%d.json'%n;v=json.loads(files[p]);assert v['node']==n and v['epoch']==1793870246.1336355 and v['source_manifest_SHA256']=='1a817f0a57fba50d8f68287e1ca2438293e8965036f63e72027845e1d1fbbd4a'
  assert v['fullModel_forward']==0 and not v.get('Model_READY')and v['create_calls']==1
  co=next(e['payload']for e in v['events']if e['event']=='constructor_observation')
  assert len(co['affinity']['owners'])==8 and all(o['changes']==0 and o['verified_mask']==''for o in co['affinity']['owners'])
  assert all(x['SHA256']=='b8a64e2d7fbdf19150a4ef6d6aadda051fd96fe276a2bd233f00c07f1aed12f5'for x in v['dladdr'].values())
  last=v['events'][-1];row=dict(node=n,PID=v['PID'],completed=v['completed'],destroy_calls=v['destroy_calls'],weight_keys=v['weight_keys'],last_event=last['event'],last_payload=last['payload'],report_SHA256=H(files[p]),error=v.get('error'))
  if n==2:
   assert v['completed']and v['destroy_calls']==1 and last['event']=='destroy_returned'and last['payload']['code']==0
   assert 'node.py", line 134'in v['traceback']and 'changes\': 0'in v['error']
   assert v['full_loader']['loaded_keys']==497052 and all(o['postload_rows_verified']for o in v['head_after_finish']['owners'])
   row.update(loader_flushes=v['full_loader']['loader_flushes'],sampled_min_free=min(x[0]for x in v['memory_sampler_final']['memory']),floor_scope='node2 eight devices only; no all32 finish/floor qualification',after_finish_affinity_scope='verify threw at device0; remaining7 raw post-finish values not retained')
  else:assert not v['completed']and v['destroy_calls']==0
  out.append(row)
 close=watch['owned_cleanup'];assert close['closed']and not close['PGID_still_live']and close['srun_returncode']==143
 post=next(v for v in watch['CPU_idle_observations']if v['phase']=='failure-post');vals=[json.loads(x)for x in post['stdout'].splitlines()if x.startswith('{')]
 assert len(vals)==4 and post['returncode']==1
 assert all(not v['kfd_pids']and not v['matching']and not v['expected_PIDs_still_exist']for v in vals)
 assert next(v for v in vals if v['node']=='a14r2n08')['ports_bindable']['26743']is False
 return dict(audit_PASS=True,Model_execution_PASS=False,quality_denominator=256,completed_requests=0,missing_requests=256,normal32destroy_proven=False,original_post_PASS=False,nodes=out,original_post=vals,archive_SHA256=H(raw),archive_files=[dict(name=n,bytes=len(b),SHA256=H(b))for n,b in files.items()],new_jobs=0,Native_calls=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args();print(json.dumps(read(a.archive),indent=2,sort_keys=True))
