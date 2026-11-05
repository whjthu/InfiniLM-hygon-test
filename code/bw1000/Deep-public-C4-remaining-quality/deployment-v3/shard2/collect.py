"""Pure terminal public64/8owner/counter/cancellation/quality retention checks."""
from pathlib import Path
import hashlib,json,time
def collect(root,run,actual,source_sha,client_result,deadline):
 root=Path(root);folder=root/'results/Deep-public-fixed-cohort-r465'/run
 def read(p):
  while time.time()<deadline:
   try:return json.loads(Path(p).read_bytes())
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
  raise TimeoutError('Original terminal visibility deadline:'+str(p))
 client=read(client_result);assert client['completed']and client['execution_passed']and len(client['rows'])==64
 assert len({(r['case_id'],r['mode'])for r in client['rows']})==64 and {r['mode']for r in client['rows']}=={'serial','cohort'}
 front=read(folder/'FRONTEND.json');ready=read(folder/'MODEL_READY.json');closed=read(folder/'closed-engine.json');q=read(folder/'owner.json')
 identity=ready['identity'];assert all(v['identity']==identity for v in[front,closed,q])and identity['actual_step']==actual and identity['source_manifest_SHA256']==source_sha
 assert ready['rows']==4 and ready['capacity_per_row']==32768 and ready['standard_public_C1']is False and ready['Native_internal_owner_count']==8
 assert q['completed']and q['normal_Native_shutdown']and q['engine_worker_shutdown_completed']and q['observed_all8_threads_exited']and q['loaded_key_count']==67569
 assert q['Native_SHA256']=='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'and q['precision_policy']['policy']==0
 assert closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited']
 masks=[];old=None
 for key in['owner_affinity_before_weight_load','owner_affinity_after_weight_load']:
  owners=q[key]['actual_owner_report']['owners'];assert len(owners)==8 and sorted(v['device']for v in owners)==list(range(8));masks=[]
  for v in owners:assert v['changes']==1;m=[int(x)for x in v['verified_mask'].split(',')];assert len(set(m))==4;masks+=m
  assert len(set(masks))==32;signature=[v['verified_mask']for v in owners];assert old is None or old==signature;old=signature
 assert q['owner_thread_observation']['status']=='OBSERVED'and len({v['TID']for v in q['owner_thread_observation']['owners']})==8
 assert all(f>=16*1024**3 for f,t in q['GPU_memory_after_load'])
 assert len(q['requests'])==64 and len(q['cohorts'])==16;by={r['request_id']:r for r in q['requests']};assert len(by)==64
 for c in client['rows']:
  r=by[c['request_id']];m=r['final_metadata'];assert r['normal_reset_after']and m['cohort_normal_reset_after']and m['real_cache_rows']==4 and m['fixed_cohort_id']==c['cohort']and m['fixed_cohort_mode']==c['mode']
  ids=r['generated_token_ids'];assert 0<len(ids)<=4096 and len(ids)==c['usage']['completion_tokens']and all(type(x)is int and 0<=x<129280 for x in ids)
  assert r['native_finish_reason']in['EOS','cap']
  assert(ids[-1]==1 and ids.count(1)==1)if r['native_finish_reason']=='EOS'else(1 not in ids and len(ids)==4096)
 calls=0;trueC4=0
 for c in q['cohorts']:
  t=c['trial'];assert c['completed']and t['normal']and t['Native_reset_before']and t['Native_reset_after']and not t['failed']
  assert len(t['states'])==4 and all(s['finish']in['EOS','cap']for s in t['states'])
  assert all(v['rc']==0 and v['finite']and len(v['rows'])==v['output_rows']for v in t['calls'])
  if c['mode']=='cohort':trueC4+=sum(v['kind']=='decode'and len(v['rows'])==4 for v in t['calls'])
  else:assert all(len(v['rows'])==1 for v in t['calls'])
  calls+=len(t['calls'])
 assert trueC4>0 and q['HC_dispatch_checked_forwards']==q['layout_dispatch_checked_forwards']==sum(q['dispatch_checks_r4842'].values())==calls
 refs=[]
 for name in['FRONTEND.json','MODEL_READY.json','owner.json','closed-engine.json','STOP.json']:
  p=folder/name;read(p);raw=p.read_bytes();refs.append(dict(path=str(p),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()))
 identity_file=Path(client_result).parent.with_name(Path(client_result).parent.name+'-identity.json');client_identity=read(identity_file)
 return dict(reports=refs,actual_owner_count=8,Native_engines=1,external_requests=64,Native_CAPI_calls=calls,true_C4_decode_calls=trueC4,quality_summary=client['quality_summary'],quality_passed=None,execution_passed=True,normal8internalowner_destroy=True,expected_node_PIDs={'a02r4n07':[front['PID'],client_identity['PID']]},Model_promotion=False,online_SLO=False)
