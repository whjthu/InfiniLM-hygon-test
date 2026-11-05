"""Offline exact49-file original643 consumption, no producer/SDK/Native imports."""
from pathlib import Path
import json,tarfile,hashlib,argparse
H=lambda b:hashlib.sha256(b).hexdigest()
def read(archive,manifest):
 raw=Path(archive).read_bytes();assert H(raw)=='e4a8195a1a944bce76230870cbd90e27a45ccf156d818e6f0d24c0afa4eaa59b'
 mraw=Path(manifest).read_bytes();assert H(mraw)=='420ae05e199b10050339fe4a7491d277180257de6ca07563b63c782169fbf95d';m=json.loads(mraw)
 files={}
 with tarfile.open(archive)as t:
  for f in t:
   assert f.isfile()and not f.name.startswith('/')and '..'not in Path(f.name).parts
   files[f.name]=t.extractfile(f).read()
 assert len(files)==49 and set(files)=={f['name']for f in m['files']}
 for f in m['files']:assert len(files[f['name']])==f['bytes']and H(files[f['name']])==f['SHA256']
 L='results/Kimi-r461-full-model-tp32-full-cohort-20261105-d';w=json.loads(files[L+'-watch.json'])
 assert w['actual_GPU']=='910380.643'and w['CPU_keeper']=='910407.1052'and w['epoch']==1793877331.6412237
 assert w['completed']and not w['passed']and w['forced_cleanup']
 rows=[];minimum=[];latest=[];finish=[];floor=4294967296
 for n in range(4):
  v=json.loads(files[L+'/node%d.json'%n]);assert v['node']==n and v['source_manifest_SHA256']=='0c22352b83f7ccd1ab773b0a2c91094c306c0442fcaf15504f0c8508fb5fab9d'
  assert v['completed']and v['Model_READY']and v['weight_keys']==497052 and v['full_loader']['loaded_keys']==497052
  assert v['error']=="AssertionError('Original sampled all32 floor')"
  assert v['create_calls']==v['destroy_calls']==1 and any(e['event']=='destroy_returned'and e['payload']['code']==0 for e in v['events'])
  assert v['fullModel_forward']==0 and not v.get('completed_requests')
  assert v['dladdr']['modern_owner_affinity_report_r2392']['SHA256']=='b8a64e2d7fbdf19150a4ef6d6aadda051fd96fe276a2bd233f00c07f1aed12f5'
  events=v['events'];rawguard=next(e['payload']for e in events if e['event']=='first_prefill_affinity_raw');guard=next(e['payload']for e in events if e['event']=='first_prefill_affinity_all32')
  assert len(guard['peers'])==4 and all(p['passed']for p in guard['peers'])
  owners=rawguard['report']['owners'];entry=sorted(v['entry_mask']);masks=[]
  assert {o['device']for o in owners}==set(range(8))
  for o in owners:
   mask=[int(x)for x in o['verified_mask'].split(',')];assert o['changes']==1 and mask==entry[o['device']*4:o['device']*4+4];masks+=mask
  assert len(masks)==len(set(masks))==32
  progress=[json.loads(s)for s in files[L+'/node%d-progress.jsonl'%n].splitlines()if s]
  assert [p['event']for p in progress]==['forward_before','forward_completed']
  call=progress[1]['payload'];assert (call['kind'],call['past'],call['width'])==(0,0,512)and len(call['per_node'])==4
  assert len({p['IDs_SHA256']for p in call['per_node']})==len({p['logits_SHA256']for p in call['per_node']})==1
  assert progress[1]['unix']<=next(e['unix']for e in events if e['event']=='first_prefill_affinity_raw')
  sampled=v['memory_sampler_final'];assert not sampled['sampler_error'];free=[a[0]for a in sampled['memory']];now=[a[0]for a in sampled['latest_memory']];after=[a[0]for a in v['memory_after_finish']]
  minimum+=free;latest+=now;finish+=after
  case=json.loads(files[L+'/node%d-case-000.json'%n]);assert not case['completed']and case['error']==v['error']
  rows.append(dict(node=n,PID=v['PID'],Ready=True,loaded_keys=v['weight_keys'],loading_seconds=v['full_loader']['loading_seconds'],loader_phase=v['full_loader']['host_loader_phase_profile'],strict8owner_binding_PASS=True,firstprefill=call,minimum_free=free,latest_pre_destroy_free=now,after_finish_free=after,perdevice_floor_deficit=[max(0,floor-x)for x in free],CAPI_destroy_returncode=0,completed_requests=0,attempted_question_ordinal=0,case_SHA256=H(files[L+'/node%d-case-000.json'%n])))
 assert len(minimum)==32 and min(minimum)<floor<min(finish)and min(latest)>floor
 post=next(p for p in w['CPU_idle_observations']if p['phase']=='failure-post');pv=[json.loads(s)for s in post['stdout'].splitlines()if s.startswith('{')]
 assert post['returncode']==1 and len(pv)==4 and all(not p['kfd_pids']and not p['matching']and not p['expected_PIDs_still_exist']for p in pv)
 assert next(p for p in pv if p['node']=='a14r2n08')['ports_bindable']['26743']is False
 return dict(audit_PASS=True,execution_PASS=False,Model_qualified=False,archive_files=49,archive_SHA256=H(raw),manifest_SHA256=H(mraw),actual_GPU=w['actual_GPU'],CPU_keeper=w['CPU_keeper'],epoch=w['epoch'],GPU_terminal='CANCELLED by0/0:15',normal_GPU_exit=False,strict32owner_firstprefill_binding_PASS=True,actual_firstprefill_per_node=1,firstprefill_width=512,firstprefill_CAPI_max_s=rows[0]['firstprefill']['CAPI_maxhost_s'],finite_scope='Actual source finite check passed before forward_completed; raw logits not retained, no independent numeric-value audit',outputs='all4 logit SHA equal diagnostic only; no emit/EOS/completed answer',sampled_min_free=min(minimum),floor=floor,worst_floor_deficit=max(floor-x for x in minimum),latest_pre_destroy_min=min(latest),after_finish_min=min(finish),floor_decision='real cumulative sampled floor FAIL; afterfinish sampledmin passed then failed before firstprefill postguard memorycheck. Window includes codec/reset/512forward/guard wait, not a specific workspace attribution; later free does not erase history',nodes=rows,CAPI_destroy_success_nodes=4,individual32owner_destructor_census_retained=False,original_post_PASS=False,original_post=pv,planned_questions=256,completed_questions=0,missing_questions=256,new_jobs=0,Native_calls_by_reader=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('manifest');a=p.parse_args();print(json.dumps(read(a.archive,a.manifest),indent=2,sort_keys=True))
