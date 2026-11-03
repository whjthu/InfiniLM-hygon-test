"""Read-only original scalar/byte audit; no reconstructed numerical oracle."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,math
MF='d03ad3175d074f8ec6bf3f47824c7b056b91ed1413fb34dd5fc5d72954295601'
LIB='74a8342812034bff5a5b7fa3426cf61ff4d23218b995f47ff7e774552d80ecd8'
def audit(path):
 raw=Path(path).read_bytes();e=json.loads(gzip.decompress(raw));files={}
 for name,v in e['files'].items():
  b=base64.b64decode(v['bytes']);assert hashlib.sha256(b).hexdigest()==v['SHA256'];files[name]=b
 def get(suffix):
  found=[b for p,b in files.items()if p.endswith(suffix)];assert len(found)==1,suffix;return found[0]
 assert hashlib.sha256(get('/GPU_SOURCE_MANIFEST.json')).hexdigest()==MF
 w=json.loads(get('/GLM431-watch-v2.json'));assert w['completed']and not w['passed']and w['actual_step']=='910374.1361'and w['GPU_source_manifest_SHA256']==MF
 accounts={r[1]:r for r in e['account_rows']};label='GLM-r431-value-latent-layer38-v2'
 assert accounts[label][0]=='910374.1361'and accounts[label][2:4]==['FAILED','1:0']and accounts[label+'-keeper'][0]=='910407.868'and accounts[label+'-keeper'][2:4]==['FAILED','1:0']
 post=w['failure_post'];assert post['actual_step']=='910374.1362'and post['passed']and not post['KFD_pids']and not post['matching_GPU_driver_PIDs']and not post['expected_PIDs_still_exist']
 assert accounts[label+'-failure-post'][2:4]==['COMPLETED','0:0']
 ranks=[];masks=[];entry=None;hidden=set();pcis=set();warm=[[],[]];load=[]
 for rank in range(8):
  q=json.loads(get('/GLM431-910374.1361-rank%d.json'%rank));assert q['rank']==rank and q['actual_step']=='910374.1361'and q['GPU_source_manifest_SHA256']==MF and q['library_SHA256']==LIB
  assert not q['normal']and not q['completed']and q['destroy_attempted_once']and q['destroy_RC']==0 and q['Runtime_expired']
  assert 'all8 diagnostic stop'in q['error']and len(q['records'])==2
  assert q['load']['keys']==12 and q['load']['checkpoint_load_passes']==1 and q['load']['payload_referenced_bytes']==165065152
  assert q['free_after_weightload']>=4*1024**3
  a=q['first_successful_seed_affinity'];wide=q['CPU_affinity_entry'];assert len(wide)>=32 and wide==sorted(set(wide))and a['actual_mask']==a['expected_mask']==wide[4*rank:4*rank+4]
  if entry is None:entry=wide
  assert wide==entry;masks+=a['actual_mask'];owners=a['actual_owner_report']['owners'];assert len(owners)==8
  for v in owners:
   if v['device']==rank:assert v['changes']==1 and list(map(int,v['verified_mask'].split(',')))==a['actual_mask']
   else:assert v['changes']==0 and not v['verified_mask']
  hidden.add(q['synthetic_hidden_SHA256']);pcis.add(q['physical_PCI']);load.append(q['load_wall_s']);numeric=[]
  for mode,r in enumerate(q['records']):
   assert r['case']=='decode-U2048-periodic'and r['index']==mode and r['mode']==mode and r['warm']and r['NoV_fixed']and r['Native_CAPI_attempted']and r['Native_CAPI_completed']and r['qualification_passed']==(mode==0)
   x=r['result'];n=x['native'];assert x['identity_valid']and x['canary']and n['mode']is True and n['value_reassociate']==bool(mode)and n['persistent_V_bytes']==0
   assert [n[k]for k in ['count','past','seeded','union_U']]==[1,32767,32768,2048]
   assert[n[k]for k in ['r431_chosen','r431_aggregation_Gemm_submits','r431_M8_expand_completed','r431_diagonal_copies','selectedV_HOST_completed_expand_calls']]==[mode,8*mode,mode,8*mode,1-mode]
   assert x['FP8_runtime_runs_delta']==mode and x['FP8_runtime_after']['invalid']==0
   if mode:assert x['FP8_runtime_after']['created']==1 and x['FP8_runtime_after']['bytes']==4136960
   for v in x['numerics']:
    assert v['finite']and all(isinstance(v[k],(int,float))and math.isfinite(v[k])for k in ['NRMSE','maxabs','reference_RMS'])
    if mode:numeric.append(dict(**v,error_RMS_derived=v['NRMSE']*v['reference_RMS']))
    else:assert v['NRMSE']==v['maxabs']==0 # firstOFF defines this diagnostic reference, no independentoracle
   warm[mode].append(n['complete_attention_CAPI_ms'])
  assert any(v['NRMSE']>.02 for v in numeric)
  ranks.append(dict(rank=rank,physical_PCI=q['physical_PCI'],first_successful_seed_CPUmask=a['actual_mask'],destroy_RC=q['destroy_RC'],Runtime_expired=q['Runtime_expired'],ON_numerics=numeric,warm_fullCAPI_ms=[r['result']['native']['complete_attention_CAPI_ms']for r in q['records']],warm_seed_s=[r['seed_wall_s']for r in q['records']]))
 assert len(set(masks))==32 and len(hidden)==1 and len(pcis)==8
 attention=[v for r in ranks for v in r['ON_numerics']if v['name']=='attention'];h=[v for r in ranks for v in r['ON_numerics']if v['name']=='hidden']
 return dict(terminal_evidence_audit_passed=True,qualification_passed=False,state='DIAGNOSTIC_STOP_IN_FIRST_ON_WARM_FULL_PLAN_INCOMPLETE',seal_SHA256=hashlib.sha256(raw).hexdigest(),original_files=len(files),source_MF=MF,library_SHA256=LIB,actual_GPU='910374.1361',actual_keeper='910407.868',GPU_elapsed=accounts[label][4],keeper_elapsed=accounts[label+'-keeper'][4],post=post,normal_full_plan_qualified=False,all8_failure_path_destroy_RC0_Runtime_expired=True,query_CAPI_per_owner_completed=2,total_query_CAPI_completed=16,formal_queries=0,tail_fixture_not_reached=True,all8_chosen_M8_8aggregate_8diagonal=True,all8_finite_canary=True,readonly_full_hidden_final_hash_not_reached=True,first_successful_seed_masks_32logical_disjoint=True,final_mask_strict_not_reached=True,
  hidden_NRMS=[min(v['NRMSE']for v in h),max(v['NRMSE']for v in h)],hidden_maxabs=max(v['maxabs']for v in h),attention_NRMS=[min(v['NRMSE']for v in attention),max(v['NRMSE']for v in attention)],attention_maxabs=max(v['maxabs']for v in attention),ranks=ranks,layer_checkpoint_load_s_range=[min(load),max(load)],cold_warm_same_trial_max8_fullCAPI_ms=[max(v)for v in warm],performance_scope='Two first/coldwarm observations only; no ABBA/formal/OFFdrift/performancegain qualification',
  raw_state_available=dict(BF16_hidden_outputs=False,BF16_attention_outputs=False,probabilities=False,selected_latent=False,activation_FP8_scales=False,reason='D2H CPU references existed in process only; persisted metric+SHA, processdestroyed. Cannot derive rawvalues/oracle fromhash.'),
  implementation_vs_rounding='UNKNOWN. Actual positive globalrow128scale/address map and originalM8expand path inherited; branch/canary/finite passes do not prove numerical implementation. Baseline perrowlatentFP8quant->W->BF16V->PV vs aggregate rawBF16->BF16round->newM8FP8quant->W changesquantization association. No real p/latent/scales saved to partition error.',
  task_quality_failure=False,Model_answers_quality='UNKNOWN',Model_speed='UNKNOWN',route_permanently_rejected_by_NRMS=False,next='PARK currentboundedcomponent after predeclared diagnosticSTOP; no thresholdchange/GPUretry/Modelload. Any next numericdiagnosis requires explicit newscope and actualtyped intermediates, not CPUproxy approval.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('sealed');p.add_argument('--output');a=p.parse_args();q=audit(a.sealed);text=json.dumps(q,indent=2,allow_nan=False)+'\n'
 if a.output:Path(a.output).write_text(text,encoding='utf8')
 else:print(text)
