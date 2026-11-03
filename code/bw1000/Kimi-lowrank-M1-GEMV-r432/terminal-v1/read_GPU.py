"""Offline stdlib audit of original 754/871/755 and all20 small raw artifacts."""
from pathlib import Path
import argparse,base64,hashlib,json,math,statistics,struct
sha=lambda b:hashlib.sha256(b).hexdigest()
def raw_file(v):
 b=base64.b64decode(v['base64'],validate=True);assert len(b)==v['bytes']and sha(b)==v['SHA256'];return b
def bf16(b):return[struct.unpack('<f',struct.pack('<I',v<<16))[0]for(v,)in struct.iter_unpack('<H',b)]
def f32(b):return[v for(v,)in struct.iter_unpack('<f',b)]
def numeric(a,b):
 assert len(a)==len(b)and a and all(math.isfinite(v)for v in a+b)
 n=len(a);rms=math.sqrt(math.fsum(v*v for v in b)/n);nr=math.sqrt(math.fsum((v-w)**2 for v,w in zip(a,b))/n)/max(rms,1e-30)
 return dict(finite=True,NRMSE=nr,max_abs=max(abs(v-w)for v,w in zip(a,b)),reference_RMS=rms,elements=n)
def same_numeric(x,y):
 for k in ['NRMSE','max_abs','reference_RMS']:assert math.isclose(x[k],y[k],rel_tol=2e-10,abs_tol=1e-15),(k,x[k],y[k])
 assert x['elements']==y['elements']and y['finite']
def table(rows,key,container='metrics'):
 def v(i):return rows[i][container][key]if container else rows[i][key]
 off=[v(2),v(5)];on=[v(3),v(4)];a=statistics.mean(off);b=statistics.mean(on);pairs=[v(3)/v(2),v(4)/v(5)];drift=off[1]/off[0]-1
 return dict(OFF=off,ON=on,OFF_mean=a,ON_mean=b,ON_over_OFF=b/a,mean_change_percent=(b/a-1)*100,OFF_endpoint_signed_drift_percent=drift*100,OFF_endpoint_abs_drift_percent=abs(drift)*100,adjacent_ON_over_OFF=pairs,adjacent_change_percent=[(x-1)*100 for x in pairs],all_adjacent_ON_faster=all(x<1 for x in pairs),mean_gain_exceeds_abs_endpoint_drift=1-b/a>abs(drift),minimum_adjacent_gain_exceeds_abs_endpoint_drift=min(1-x for x in pairs)>abs(drift),units='milliseconds'if key=='GPU_stream_event_ms'else'seconds')
def audit(path):
 envelope=Path(path).read_bytes();q=json.loads(envelope);wblock=q['watch'];oblock=q['owner']
 for block in [wblock,oblock]:b=block['rawtext'].encode();assert len(b)==block['bytes']and sha(b)==block['SHA256']
 w=json.loads(wblock['rawtext']);r=json.loads(oblock['rawtext']);assert w['completed']and w['passed']and w['actual_step']=='910371.754'and w['normal1']and w['Slurm_COMPLETED0']and w['srun_exit0']
 assert w['GPU_source_manifest_SHA256']=='ab65b836de3711f85b805d80cfc18b1c250eda89644d75f14dfc229a2c5d4b65'and w['publication_commit']=='965a19d843ce5b3808a500606919e7cedc57dda8'
 assert r['actual_step']=='910371.754'and r['completed']and r['passed']and r['normal_destroy']and r['destroy_rc']==0
 assert r['source_manifest_SHA256']==w['GPU_source_manifest_SHA256']and r['actual_call_counts']==[3,4,3,4]and r['visible_count']==1
 assert r['entry_affinity']==r['affinity_after_create']==r['final_affinity']==[0,1,2,3]and r['wrong_owner_rejected']
 assert r['actual_weight_layout']==[3584,7168,7168,1,1,7168,3584,3584,1,1]
 assert r['actual_BLAS_policy']['failures']==0 and r['actual_BLAS_policy']['owners']and all(v['created']==v['verified']for v in r['actual_BLAS_policy']['owners'])
 assert r['host_weights_after_SHA256']==[v['payload_SHA256']for v in r['weights']]and len(r['records'])==12 and len(r['diagnostics'])==2
 expected_maps={'libinfiniops.so':'6d5a6204f7eef70e84deb9cc737a36c6ae2c1fbc75a0f70645557c8441ed85ca','_infinilm.cpython':'2e1458609cc81a31351c451004c045b8e84f7581844afedf1a577df2c940ccb2','libcublas.so':'c06fcaa76e04342e10034e149fc14a3cef79938b6a5867c38aced26a40108c2a','libinfinilm_runtime.so':'04a6a1a86c97d176e4304b9bb4c1d6fcb8550784beaa47e0f5841fb7603ffabf','libKimi_lowrank_M1_r432.so':'a431011ac89f009dca15b029191007a29eae13b126ac56ff94e463b2c5623489','libcudart.so':'ac547071522961b136d7000d2268b8cb470b1f085007b3be3cb9039d50281f2e'}
 for name,h in expected_maps.items():assert[v for p,v in r['mapped_libraries'].items()if name in p]==[h],name
 post=w['post'];assert post['passed']and post['actual_step']=='910371.755'and not post['KFD_pids']and not post['matching_GPU_driver_PIDs']and not post['expected_PIDs_still_exist']and not post['GPU_initialized'];assert r['physical_PCI']in post['physical_PCI']
 account=q['accounting']['stdout'];assert q['accounting']['rc']==0
 expected={'910371.753':1,'910371.754':38,'910371.755':1,'910407.871':42}
 for step in expected:
  lines=[s.split('|')for s in account.splitlines()if s.startswith(step+'|')];assert len(lines)==1 and lines[0][2:4]==['COMPLETED','0:0']
  elapsed=lines[0][4];parts=[int(v)for v in elapsed.split(':')];seconds=sum(x*60**(len(parts)-1-i)for i,x in enumerate(parts));assert seconds==expected[step]
 artifacts=q['raw_output_artifacts'];assert len(artifacts)==20
 def artifact_key(path):
  root='/public/home/aclsylqidf/hygon-inference-refactor-20261014/'
  if path.startswith(root):return '/refactor/'+path[len(root):]
  assert path.startswith('/refactor/results/'),'Unexpected artifact namespace';return path
 raws={artifact_key(p):raw_file(v)for p,v in artifacts.items()};assert len(raws)==20;numeric_rows=[];timing=[]
 for projection,n,k in [(0,3584,7168),(1,7168,3584)]:
  a=[v for v in r['records']if v['projection']==projection];diag=[v for v in r['diagnostics']if v['projection']==projection];assert[v['mode']for v in a]==[0,1,0,1,1,0]and len(diag)==1 and diag[0]['guarded_diagnostic']
  refs=r['input_reference_raw_artifacts'][projection]
  for kind,meta in refs.items():
   b=raws[meta['path']];assert sha(b)==meta['SHA256']and len(b)==meta['bytes']and meta['endianness']=='little'
  inp=raws[refs['input']['path']];assert len(inp)==k*2
  ref_b=raws[refs['CPUref']['path']];ref=bf16(ref_b);unrounded=f32(raws[refs['CPUref_F32']['path']]);assert len(ref)==len(unrounded)==n
  # Independently verify actual CPU-reference F32 -> BF16 RNE conversion bytes.
  rounded=b''.join(struct.pack('<H',((u+0x7fff+((u>>16)&1))>>16)&0xffff)for(u,)in struct.iter_unpack('<I',raws[refs['CPUref_F32']['path']]))
  assert rounded==ref_b
  base=bf16(raws[a[0]['output_path']])
  for rec in diag+a:
   b=raws[rec['output_path']];assert sha(b)==rec['output_SHA256']and len(b)==rec['output_bytes']==n*2 and rec['output_dtype']=='BF16'and rec['output_shape']==[n]and rec['output_endianness']=='little'
   assert rec['rc']==0 and rec['output_valid_by_producer']and rec['input_SHA256']==sha(inp)and rec['metrics']['canary_ok']==1
   vals=bf16(b);vs_ref=numeric(vals,ref);same_numeric(vs_ref,rec['vs_CPU_reference']);assert vs_ref['NRMSE']<=.02
   vs_base=numeric(vals,base)
   if 'vs_original_OFF'in rec:same_numeric(vs_base,rec['vs_original_OFF']);assert vs_base['NRMSE']<=.02
   numeric_rows.append(dict(projection=projection,slot=rec['slot'],mode=rec['mode'],guard=rec['guarded_diagnostic'],vs_rounded_CPU_F32=vs_ref,vs_unrounded_CPU_F32=numeric(vals,unrounded),vs_warm_original_OFF=vs_base,output_SHA256=rec['output_SHA256']))
   assert all(math.isfinite(v)and v>=0 for v in rec['metrics'].values())and rec['ctypes_dispatch_checked_s']>0
  timing.append(dict(projection=projection,N=n,K=k,warmup_indices=[0,1],formal_indices=[2,3,4,5],warm_original_OFF=a[0]['metrics'],warm_candidate_ON=a[1]['metrics'],guarded_first_candidate=diag[0]['metrics'],allocation_dispatch_drain=table(a,'allocation_dispatch_drain_s'),complete_native_call=table(a,'complete_native_call_s'),GPU_stream_interval=table(a,'GPU_stream_event_ms'),ctypes_complete_checked=table(a,'ctypes_dispatch_checked_s',None)))
 recomputed_max=dict(vs_rounded_CPU_F32=max(x['vs_rounded_CPU_F32']['NRMSE']for x in numeric_rows),vs_unrounded_CPU_F32=max(x['vs_unrounded_CPU_F32']['NRMSE']for x in numeric_rows),vs_original_OFF=max(x['vs_warm_original_OFF']['NRMSE']for x in numeric_rows))
 known_serial=r['checkpoint_load_s']+r['create_s']+sum(v['seconds']for v in r['independent_CPU_reference'])+sum(v['ctypes_dispatch_checked_s']for v in r['records']+r['diagnostics'])
 return dict(passed=True,state='SCOPED_NUMERIC_AND_LIFECYCLE_PASS_MEAN_COMPONENT_GAIN_WITH_BASELINE_DRIFT_NOT_MODEL',evidence=dict(sealed_file=str(Path(path).resolve()),sealed_SHA256=sha(envelope),watch_SHA256=wblock['SHA256'],owner_report_SHA256=oblock['SHA256'],raw_artifacts=20,raw_bytes=sum(len(v)for v in raws.values())),identity=dict(publication=w['publication_commit'],GPU_manifest_SHA256=w['GPU_source_manifest_SHA256'],actual_GPU='910371.754',actual_keeper='910407.871',actual_before='910371.753',actual_post='910371.755',original_start=w['original_start'],physical_PCI=r['physical_PCI'],visible_count=1,actual_affinity=r['entry_affinity']),scope=dict(mapped_libraries=r['mapped_libraries'],actual_checkpoint_layer1_BF16_weights=True,input='Two synthetic M1 vectors, not Model activations',cases=2,timed_calls=12,guarded_diagnostics=2,formal_ABBA=True,No_Model=True,No_MoE_or_TP=True,actual_F32_reference_provider=r['CPU_reference_provider']),numeric_max_NRMSE=recomputed_max,numeric_recomputed_from_original_BF16_and_CPU_reference_bytes=numeric_rows,timing=timing,startup_and_cardcost=dict(checkpoint_load_s=r['checkpoint_load_s'],create_s=r['create_s'],untimed_CPU_F32_reference=r['independent_CPU_reference'],complete_process_wall_s=r['complete_process_wall_s'],sum_nonoverlapping_measured_checkpoint_create_CPUref_and_ctypes_s=known_serial,remaining_processwall_validation_import_hash_SDK_observer_overhead_s=r['complete_process_wall_s']-known_serial,remaining_not_attributed_to_coldkernel=True,GPU_step_allocation_s=38,all_component_cards=1,GPU_card_seconds=38,keeper_CPU_elapsed_s=42,CPU_sourcebuild_elapsed_s=11.023776452057064,cold_first_original_OFF_down_s=timing[0]['warm_original_OFF']['complete_native_call_s'],no_cold_cost_disappears=True),lifecycle=dict(create_once_source_contract=True,destroy_once_rc0=True,owner_rejection_pass=True,actual_call_counters=r['actual_call_counts'],normal1=True,srun_and_Slurm_all0=True,post=post,blas_policy=r['actual_BLAS_policy']),decision=dict(default_OFF=True,Model_GO=False,both_projection_mean_complete_gain_exceeds_endpoint_drift=all(v['complete_native_call']['mean_gain_exceeds_abs_endpoint_drift']for v in timing),both_projection_every_adjacent_complete_gain_exceeds_endpoint_drift=all(v['complete_native_call']['minimum_adjacent_gain_exceeds_abs_endpoint_drift']for v in timing),interpretation='Both adjacent pairs ON faster and both ratio-of-means component gains exceed each measured OFF endpoint drift. However the smaller pair gains do not both exceed drift; few samples and down baseline warming remain. Keep source candidate, no auto full-MoE/Model/device expansion. Full MoE benefit and cold/steady amortization UNKNOWN; a future separate full-MoE card requires scope/source claim.',no_dense_peak_or_HBM_lower_bound_Model_prediction=True,no_quality_or_acceptance_claim=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('sealed');p.add_argument('--output');a=p.parse_args();v=audit(a.sealed);text=json.dumps(v,indent=2,allow_nan=False)+'\n'
 if a.output:Path(a.output).write_text(text,encoding='utf8')
 else:print(text,end='')
