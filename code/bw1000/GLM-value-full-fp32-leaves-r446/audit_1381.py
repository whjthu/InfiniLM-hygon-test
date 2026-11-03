"""Offline raw-output/max8 audit; no Runtime, GPU, Model, subprocess or polling."""
from pathlib import Path
from array import array
import argparse,base64,gzip,hashlib,json,math,struct,sys
MF='8cb35541e90d7512d207fce8af47a4df00cbf6b05c7ab7e597cae813792ded5a'
LIB='a3f33c7ca44fd43f42f93ff040c916bb03e79935332533fcc4f5d4bb00e5c1b5'
ROOT='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
def norm(p):return '/refactor'+str(p)[len(ROOT):]if str(p).startswith(ROOT+'/')else str(p)
def values(raw,kind):
 a=array('H'if kind=='BF16'else'f',raw)
 if sys.byteorder!='little':a.byteswap()
 if kind=='F32':return a
 return [struct.unpack('<f',struct.pack('<I',v<<16))[0]for v in a]
def metric(a,b):
 assert len(a)==len(b)and len(a)>0 and all(map(math.isfinite,a))and all(map(math.isfinite,b))
 diff=sum((x-y)**2 for x,y in zip(a,b));ref=sum(x*x for x in b);assert ref>0
 return dict(finite=True,NRMSE=math.sqrt(diff/ref),reference_RMS=math.sqrt(ref/len(b)),maxabs=max(abs(x-y)for x,y in zip(a,b)))
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def e4(b):
 s,e,m=b>>7,(b>>3)&15,b&7
 assert not(e==15 and m==7)
 v=math.ldexp(m,-9)if e==0 else math.ldexp(8+m,e-10)
 return -v if s else v
def audit(path):
 seal=json.loads(Path(path).read_bytes());assert seal['actual_GPU']=='910374.1381'and seal['actual_keeper']=='910407.923'and seal['all_frozen_GPU_source_members_unchanged']and seal['GPU_source_manifest_SHA256']==MF
 files={};hashes={}
 for row in seal['files']:
  key=norm(row['path']);assert key not in files;raw=gzip.decompress(base64.b64decode(row['gzip_base64']));assert len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['SHA256'];files[key]=raw;hashes[key]=row['SHA256']
 def get(name):
  keys=[k for k in files if k.endswith('/'+name)];assert len(keys)==1,(name,keys);return json.loads(files[keys[0]])
 def raw(ref):
  b=files[norm(ref['path'])];assert len(b)==ref['nbytes']and hashlib.sha256(b).hexdigest()==ref['SHA256'];return b
 watch=get('GLM446-watch-v1.json');assert watch['completed']
 # Preserve the original HOST collector failure. It happened after all device
 # work/srun ended, while reading a container path directly on the HOST.
 assert watch['passed']is False and "FileNotFoundError"in watch['error']and "sha(raw['path'])"in watch['traceback']and "'/refactor/results/GLM446-910374.1381-rank0-raw/"in watch['traceback']
 assert watch['last_observation']['srun_exit']==0 and watch['last_observation']['row'][2:4]==['COMPLETED','0:0']
 assert watch['failure_post']['passed']and not watch['failure_post']['KFD_pids']and not watch['failure_post']['expected_PIDs_still_exist']
 assert watch['GPU_source_manifest_SHA256']==MF and watch['actual_step']=='910374.1381'and len(watch['submissions'])==1
 rows={r[0]:r for r in seal['accounting']};assert rows['910374.1381'][2:4]==['COMPLETED','0:0']and rows['910407.923'][2:4]==['FAILED','1:0']and rows['910374.1382'][2:4]==['COMPLETED','0:0']
 roles=[get('GLM446-910374.1381-rank%d.json'%i)for i in range(8)]
 assert all(q['completed']and q['normal']and q['normal_destroy']and q['Runtime_weak_expired']and q['Torch_after_destroy']and q['library_SHA256']==LIB and q['GPU_source_manifest_SHA256']==MF for q in roles)
 masks=[q['CPU_affinity']for q in roles];assert all(len(m)==4 for m in masks)and len(set(c for m in masks for c in m))==32
 assert len({q['physical_PCI']for q in roles})==8
 checked=0;metrics=[];groups={};typed={}
 for rank,q in enumerate(roles):
  assert q['rank']==rank and len(q['records'])==12 and [r['mode']for r in q['records']]==[0,1,0,1,1,0]*2
  assert q['first_successful_seed_affinity']['actual_mask']==q['CPU_affinity']==q['first_successful_seed_affinity']['expected_mask']
  before=q['first_successful_seed_affinity']['actual_owner_report'];after=q['owner_affinity_after_queries'];assert before==after
  stats=q['FP8_runtime_final_stats'];assert stats['created']==stats['destroyed']and stats['runs']==stats['entries']==stats['bytes']==stats['invalid']==0
  references={}
  for record in q['records']:
   assert record['Native_CAPI_completed']and record['Native_CAPI_RC']==0 and record['qualification_passed']and record['raw_outputs_complete']and record['nonzero_original_reference']
   result=record['result'];n=result['native'];mode=record['mode'];assert result['canary']and result['identity_valid']and n['past']==32767 and n['union_U']==2048 and n['mode']is True and n['persistent_V_bytes']==0
   assert [n[k]for k in ['r433_chosen','r433_rowquant_submits','r433_roundtrip_F32_identity_calls','r446_F32_aggregation_GEMV_leaves','r446_F32_V_GEMV_leaves','r446_final_software_BF16_cast_leaves']]==[mode,mode,mode,8*mode,8*mode,mode]
   if mode:assert n['r433_identity_device_check_passed']and result['successful_fullfp32_source_contract']['cast_calls']==3
   key=(record['case'],record['index']);groups.setdefault(key,[]).append(dict(rank=rank,mode=mode,warm=record['warm'],dispatch_s=result['dispatch_wall_s'],CAPI_ms=n['complete_attention_CAPI_ms'],checked_s=record['checked_query_wall_s']))
   for i,ref in enumerate(record['raw_outputs']):
    a=values(raw(ref),'BF16');name=ref['name'];k=(record['case'],name)
    if k not in references:assert mode==0 and record['index']==0;references[k]=a
    m=metric(a,references[k]);old=result['numerics'][i]
    for field in ['NRMSE','reference_RMS','maxabs']:assert math.isclose(m[field],old[field],rel_tol=2e-5,abs_tol=1e-8),(rank,record['case'],record['index'],field,m[field],old[field])
    assert m['NRMSE']<=.02;metrics.append(dict(rank=rank,case=record['case'],index=record['index'],mode=mode,name=name,**m));checked+=1
   diag=record.get('typed_diagnostic')
   if record['case']=='decode-U2048-periodic'and record['index']<2:
    assert diag['collection_passed']and diag['clear_RC']==0 and diag['descriptor']['query_CAPI_completed']
    slots={r['name']:r for r in diag['files']}
    for ref in slots.values():raw(ref)
    if rank==0:typed[mode]={k:(v,raw(v))for k,v in slots.items()}
   else:assert diag is None and n['diagnostic_snapshot_detail']==0
 assert checked==192 and len(groups)==12
 max8=[]
 for (case,index),rs in sorted(groups.items()):
  assert len(rs)==8 and {r['rank']for r in rs}==set(range(8));max8.append(dict(case=case,index=index,mode=rs[0]['mode'],warm=rs[0]['warm'],dispatch_s=max(r['dispatch_s']for r in rs),CAPI_s=max(r['CAPI_ms']for r in rs)/1000,checked_s=max(r['checked_s']for r in rs),all8=rs))
 comparison=[]
 for case in sorted({r['case']for r in max8}):
  rs={r['index']:r for r in max8 if r['case']==case};off=sum(rs[i]['dispatch_s']for i in [2,5])/2;on=sum(rs[i]['dispatch_s']for i in [3,4])/2;drift=abs(rs[5]['dispatch_s']/rs[2]['dispatch_s']-1)
  pairs=[dict(OFF_index=a,ON_index=b,OFF_s=rs[a]['dispatch_s'],ON_s=rs[b]['dispatch_s'],latency_reduction=1-rs[b]['dispatch_s']/rs[a]['dispatch_s'])for a,b in [(2,3),(5,4)]]
  reduction=1-on/off;extra={}
  for field in ['CAPI_s','checked_s']:
   a=sum(rs[i][field]for i in [2,5])/2;b=sum(rs[i][field]for i in [3,4])/2
   extra[field]=dict(OFF_mean_max8_s=a,ON_mean_max8_s=b,latency_reduction=1-b/a,OFF_endpoint_relative_drift=abs(rs[5][field]/rs[2][field]-1))
  comparison.append(dict(case=case,primary_scope='Complete ctypes dispatch including original CAPI/return/synchronization, same-trial max8',OFF_mean_max8_s=off,ON_mean_max8_s=on,speedup=off/on,latency_reduction=reduction,OFF_endpoint_relative_drift=drift,pairs=pairs,both_pairs_positive=all(p['latency_reduction']>0 for p in pairs),positive_beyond_OFF_drift=reduction>drift,other_complete_cost_scopes=extra))
 # Actual rank0 row dequant and all8 interleaved V head/global128 scale mappings.
 _,on=typed[1]['row_roundtrip_F32'];xq=values(on,'F32');packed=typed[1]['packed_latent_U8'][1];xs=values(typed[1]['row_scale_F32'][1],'F32')
 max_x=0.
 for i,got in enumerate(xq):
  code=packed[i];assert code not in [127,255];expected=f32(e4(code)*xs[(i//512)*4+(i%512)//128]);max_x=max(max_x,abs(got-expected))
 assert max_x==0
 w=typed[1]['original_W_U8'][1];ws=values(typed[1]['global_W_scale_F32'][1],'F32');max_w=0.;count=0
 for head in range(8):
  actual=values(typed[1]['Vhead%d_F32'%head][1],'F32')
  for k in range(512):
   for v in range(256):
    row=head*448+192+v;expected=f32(e4(w[row*512+k])*ws[(row//128)*4+k//128]);max_w=max(max_w,abs(actual[k*256+v]-expected));count+=1
 assert max_w==0 and count==1048576
 return dict(passed=True,original_run_passed=False,original_keeper_FAILED=True,original_collector_error=watch['error'],original_collector_traceback=watch['traceback'],original_forced_cleanup_record=watch['owned_failure_cleanup'],original_seal_SHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),files=len(files),original_bytes=seal['original_bytes'],actual_GPU='910374.1381',actual_keeper='910407.923',accounting=seal['accounting'],normal_actual8owners_and_post_clean=True,all96_queries_checked=True,BF16_output_metrics_recomputed=checked,formal_max8_comparison=comparison,formal_queries_all8=64,warm_queries_all8=32,max8_samples=max8,max_hidden_NRMSE=max(m['NRMSE']for m in metrics if m['name']=='hidden'),max_attention_NRMSE=max(m['NRMSE']for m in metrics if m['name']=='attention'),metric_details=metrics,rank0_row_roundtrip_maxabs=max_x,rank0_V_global128_head448_mapping_maxabs=max_w,rank0_V_mapping_values=count,Model_quality_qualified=False,Model_speedup='UNKNOWN',public_HTTP_SLO=False,limits=['Original overall keeperFAIL retained; offline audit is separate, not a successful controller rerun','Two synthetic hidden/selection geometries with real checkpoint layer38; not Model activation/routing/quality','Only oneABBA/fixture; max8 same-trial dispatcher and CPU timing, originalOFF drift retained','Warm coldsetup/snapshots excluded from formal comparisons; first400-to45ms is not a speedup claim','Typed W/row dequant exactness independently checked only savedrank0; all8 rawoutput/counter/finite checks pass'])
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('seal');a.add_argument('--output');args=a.parse_args();q=audit(args.seal);raw=json.dumps(q,indent=2)+'\n'
 if args.output:Path(args.output).write_text(raw,encoding='utf8')
 print(json.dumps({k:v for k,v in q.items()if k not in ['metric_details','max8_samples']},indent=2))
