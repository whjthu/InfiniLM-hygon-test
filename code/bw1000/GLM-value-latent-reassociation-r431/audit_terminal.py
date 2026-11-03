"""Offline original-byte audit. No SSH/Slurm/GPU submissions or imports."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,math,statistics
MF='8ec1360c3ded961b549cb9c123f10857c1b3ea7ef8fadf826fc6e2a63d291c57'
LIB='74a8342812034bff5a5b7fa3426cf61ff4d23218b995f47ff7e774552d80ecd8'
def audit(path):
 raw=Path(path).read_bytes();envelope=json.loads(gzip.decompress(raw));files={}
 for p,v in envelope['files'].items():
  b=base64.b64decode(v['bytes']);assert hashlib.sha256(b).hexdigest()==v['SHA256'],p;files[p]=b
 get=lambda suffix:next(b for p,b in files.items()if p.endswith(suffix))
 assert hashlib.sha256(get('/GPU_SOURCE_MANIFEST.json')).hexdigest()==MF
 watch=json.loads(get('/GLM431-watch-v1.json'));assert watch['completed']
 if not watch['passed']:
  assert watch['actual_step']=='910374.1358'and watch['GPU_source_manifest_SHA256']==MF
  accounts=envelope['account_rows'];labels={r[1]:r for r in accounts};assert labels['GLM-r431-value-latent-layer38-v1'][0:4:2]==['910374.1358','FAILED']and labels['GLM-r431-value-latent-layer38-v1-keeper'][2]=='FAILED'
  post=watch['failure_post'];assert post['passed']and not post['KFD_pids']and not post['matching_GPU_driver_PIDs']and not post['expected_PIDs_still_exist']
  ranks=[]
  for rank in range(8):
   q=json.loads(get('/GLM431-910374.1358-rank%d.json'%rank));assert q['rank']==rank and not q['normal']and not q['completed']and q['records']==[]
   assert '2392 owner mask too narrow for eight disjoint requested subsets'in q['error']and 'glm_attention_create_r398'in q['traceback']
   assert q['library_SHA256']==LIB and q['GPU_source_manifest_SHA256']==MF and 'load'not in q
   ranks.append(dict(rank=rank,PID=q['PID'],error=q['error'],normal=False,records=0))
  return dict(passed=False,terminal_evidence_audit_passed=True,seal_SHA256=hashlib.sha256(raw).hexdigest(),original_files=len(files),actual_GPU='910374.1358',actual_keeper='910407.867',failure_post=post,ranks=ranks,stage='factory returnedNULL before checkpointload/0query; runtimepartial constructor not normaldestroy qualification',cause='Driver prepinPython4CPU +newRuntimeownerwidth4 requires8disjoint masks/32CPU; caller mask too narrow',algorithm_numeric_negative=False,automatic_retry=False)
 assert watch['normal8']and watch['srun_exit0']and watch['Slurm_COMPLETED0']
 assert watch['GPU_source_manifest_SHA256']==MF and watch['actual_step']=='910374.1358'
 post=watch['post'];assert post['passed']and not post['KFD_pids']and not post['matching_GPU_driver_PIDs']and not post['expected_PIDs_still_exist']
 accounts=envelope['account_rows'];bylabel={r[1]:r for r in accounts}
 for label,step in [('GLM-r431-value-latent-layer38-v1','910374.1358'),('GLM-r431-value-latent-layer38-v1-keeper','910407.867')]:
  row=bylabel[label];assert row[0]==step and row[2:4]==['COMPLETED','0:0']
 assert all(r[2:4]==['COMPLETED','0:0']for r in accounts if r[1].endswith('-post'))
 owners=[];masks=[];allmetrics=[];inputs=set();pcis=set();cases=['decode-U2048-periodic','decode-U2048-tail']
 for rank in range(8):
  b=get('/GLM431-910374.1358-rank%d.json'%rank);q=json.loads(b);owners.append(q)
  assert q['completed']and q['normal']and q['normal_destroy']and q['Runtime_weak_expired']and q['Torch_after_destroy']and q['policy_closed']and q['destroy_RC']==0 and q['Runtime_expired']
  assert q['rank']==rank and q['actual_step']==watch['actual_step']and q['GPU_source_manifest_SHA256']==MF and q['library_SHA256']==LIB
  assert q['measurement_calls']==8 and q['warm_calls']==4 and q['query_calls']==12 and q['seed_calls']==780 and q['NoV_fixed']and q['value_reassociation_only_variable']and q['input_immutable']
  stats=q['FP8_runtime_final_stats'];assert stats['created']==stats['destroyed']and stats['runs']==6 and stats['entries']==stats['bytes']==stats['invalid']==0
  before=q['owner_affinity_before_queries']['owners'];after=q['owner_affinity_after_queries']['owners'];fields=lambda a:sorted((v['device'],v['changes'],v['verified_mask'])for v in a);assert fields(before)==fields(after)
  mine=[v for v in after if v['device']==rank];assert len(mine)==1 and mine[0]['changes']==1;mask=list(map(int,mine[0]['verified_mask'].split(',')));assert mask==q['CPU_affinity']and len(mask)==4;masks+=mask
  inputs.add(q['synthetic_hidden_SHA256']);pcis.add(q['physical_PCI']);assert len(q['records'])==12
  for i,r in enumerate(q['records']):
   mode=[0,1,0,1,1,0][i%6];assert r['case']==cases[i//6]and r['index']==i%6 and r['mode']==mode and r['warm']==(i%6<2)
   assert r['Native_CAPI_attempted']and r['Native_CAPI_completed']and r['qualification_passed']and r['NoV_fixed']
   x=r['result'];n=x['native'];assert x['identity_valid']and x['canary']and n['mode']is True and n['value_reassociate']==bool(mode)
   assert n['past']==32767 and n['count']==1 and n['union_U']==2048 and n['seeded']==32768 and n['persistent_V_bytes']==0
   assert[n[k]for k in ['r431_chosen','r431_aggregation_Gemm_submits','r431_M8_expand_completed','r431_diagonal_copies']]==[mode,8*mode,mode,8*mode]
   assert n['selectedV_HOST_completed_expand_calls']==1-mode and x['FP8_runtime_runs_delta']==mode and x['FP8_runtime_after']['invalid']==0
   for v in x['numerics']:
    assert v['finite']and all(isinstance(v[k],(int,float))and math.isfinite(v[k])for k in ['NRMSE','maxabs','reference_RMS'])and v['NRMSE']<=.02;allmetrics.append(v)
   assert n['complete_attention_CAPI_ms']>0 and r['checked_query_wall_s']>0 and r['seed_wall_s']>0
 assert len(set(masks))==32 and len(pcis)==8 and len(inputs)==1
 timing={}
 for c,case in enumerate(cases):
  indices=[c*6+j for j in range(2,6)];modes=[0,1,1,0]
  record=dict(modes=modes,perrank_fullCAPI_ms={str(q['rank']):[q['records'][i]['result']['native']['complete_attention_CAPI_ms']for i in indices]for q in owners})
  for name,key,scale in [('fullCAPI_ms','complete_attention_CAPI_ms',1),('dispatch_wall_ms','dispatch_wall_s',1000),('checkeddriver_ms','checked_query_wall_s',1000)]:
   def value(q,i):
    r=q['records'][i]
    return scale*(r[key]if key=='checked_query_wall_s'else r['result'][key]if key=='dispatch_wall_s'else r['result']['native'][key])
   v=[max(value(q,i)for q in owners)for i in indices];off=(v[0]+v[3])/2;on=(v[1]+v[2])/2
   record[name]=dict(same_trial_max8=v,OFFmean=off,ONmean=on,ON_wall_reduction_percent=100*(1-on/off),OFFendpoint_drift_percent=100*abs(v[3]-v[0])/off,adjacent_pair_wall_reduction_percent=[100*(1-v[1]/v[0]),100*(1-v[2]/v[3])])
  record['prefixseed_s_per_rank']={str(q['rank']):[r['seed_wall_s']for r in q['records'][c*6:c*6+6]]for q in owners}
  timing[case]=record
 return dict(passed=True,seal_SHA256=hashlib.sha256(raw).hexdigest(),original_files=len(files),actual_GPU=watch['actual_step'],actual_keeper='910407.867',query_CAPI_per_owner=12,total_query_CAPI=96,normal8=True,all8_numeric_checks=len(allmetrics),max_NRMSE=max(v['NRMSE']for v in allmetrics),maxabs=max(v['maxabs']for v in allmetrics),timing=timing,scope='Synthetic geometry/checkpoint layer38 completeattention; approximatevalue path diagnostic. NoModel/quality/HTTP/SLO claim; prefixseed/reset/load separate. Workspacehighwater/free snapshots are not peakallocation.',performance_promotion_automatic=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('sealed');p.add_argument('--output');a=p.parse_args();q=audit(a.sealed);text=json.dumps(q,indent=2,allow_nan=False)+'\n'
 if a.output:Path(a.output).write_text(text,encoding='utf8')
 else:print(text)
