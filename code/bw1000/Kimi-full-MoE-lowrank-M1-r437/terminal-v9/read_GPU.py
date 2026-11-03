"""Original775 full-MoE raw audit. Stdlib, no SDK/weights/job execution."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,math,statistics,struct
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
MF='65bbf365fafb3e96cf4707663ad8be09a1c316f35def762b9044d6042a8ee343'
LIB='422f49d3d1a188ad391eb31ea2c42583a892e96ef4f8647a7f1cb079b8f0eba3'
CPUSHA='7105696f77363f449971ef7ec8f8157a9411b97f9b15e36316c229c9a8893c9b'
sha=lambda b:hashlib.sha256(b).hexdigest()
def audit(archive,observation):
 blob=Path(archive).read_bytes();assert sha(blob)=='151559edc8523b701aab325a1c2665d32847d56c6739e34e7c24fa5bfd2197db'
 seal=json.loads(gzip.decompress(blob));assert seal['new_jobs']==seal['new_GPU']==0 and len(seal['files'])==141
 data={}
 for p,v in seal['files'].items():
  raw=base64.b64decode(v['data'])if v['encoding']=='base64'else v['data'].encode();assert len(raw)==v['bytes']and sha(raw)==v['SHA256'];data[p]=raw
 def path(p):
  if p.startswith('/refactor/'):return R+p[len('/refactor'):]
  assert p.startswith(R+'/');return p
 def raw(p):return data[path(p)]
 def get(p):return json.loads(raw(p))
 observed=json.loads(Path(observation).read_bytes());assert observed['accounting']['rc']==0
 accounting={line.split('|')[0]:line.split('|')for line in observed['accounting']['stdout'].splitlines()if line}
 for step,label in [('910371.774','Kimi-r437-full-MoE-row-span-v8-before'),('910371.775','Kimi-r437-full-MoE-row-span-v8'),('910371.776','Kimi-r437-full-MoE-row-span-v8-post'),('910407.907','Kimi-r437-full-MoE-row-span-v8-keeper')]:
  assert accounting[step][1:4]==[label,'COMPLETED','0:0']
 assert observed['watch']['SHA256']==sha(raw('/refactor/results/Kimi437-watch-v8.json'))
 w=get('/refactor/results/Kimi437-watch-v8.json');assert w['completed']and w['passed']and w['normal16']and w['all_96_forward_results_persisted']and w['srun_exit0']and w['Slurm_COMPLETED0']and not w.get('forced_cleanup')and not w.get('error')
 assert w['actual_step']=='910371.775'and w['GPU_source_manifest_SHA256']==MF and w['profile']==0 and w['no_retry']
 assert len(w['post'])==2 and all(v['passed']and v['actual_step']=='910371.776'and not v['KFD_pids']and not v['matching_GPU_driver_PIDs']and not v['expected_PIDs_still_exist']and v['passive_master_port40917_bindable']for v in w['post'])
 assert len({(v['node'],pci)for v in w['before']for pci in v['physical_PCI']})==16
 mpath='/refactor/incoming/Kimi-full-MoE-lowrank-M1-r437/gpu-v8/GPU_SOURCE_MANIFEST.json';assert sha(raw(mpath))==MF;m=get(mpath)
 assert m['GPU_GO']and not m['Model_GO']and m['modes']==[0,1,0,1,1,0]and m['total_forward_calls']==96 and m['library_SHA256']==LIB and m['CPU_receipt_SHA256']==CPUSHA
 assert sha(raw(m['CPU_receipt']))==CPUSHA;cpu=get(m['CPU_receipt']);assert cpu['passed']and cpu['actual_CPU']=='910407.905'and cpu['library_SHA256']==LIB and len(cpu['compiled'])==2
 assert cpu['actual_pure_address_contract']['address_and_rejection_cases']==33 and cpu['actual_source_contract']['original_MoE_body_byte_identical']
 assert sha(raw(m['owner_source_root']+'/SOURCE_MANIFEST.json'))==m['owner_source_SHA256']
 rows=[];raw_metrics=[];descs=[];nodes=[];global_mask_pairs=set()
 for node in[0,1]:
  py=get('/refactor/results/Kimi437-910371.775-node%d.json'%node)
  assert py['completed']and py['passed']and py['normal_close']and py['returncode']==0 and py['actual_step']=='910371.775'and py['node_rank']==node and py['source_manifest_SHA256']==MF and py['original_epoch']==w['original_start']
  assert py['node']==['a02r3n03','a02r3n07'][node]and py['TP']==16 and py['PP']==1 and py['actual_C1']and py['padded_scale_storage']and not py['long_factory']and not py['Model']and py['HTTP']==0
  assert py['CPU_weights_and_input_unchanged']and py['input']['SHA256']=='3f84e8d80178c05212edcc5b703f907290df20925684fb610838c6d1f259ecca'
  assert py['visible_count']==8 and len(set(py['physical_PCI_by_visible_device']))==8 and len(py['raw_outputs'])==48
  assert all(v['SHA256']==LIB for v in py['actual_CABI_bindings'].values())
  for p,h in m['mapped_dependencies_SHA256'].items():assert py['mapped_libraries'].get(p)==h,(p,py['mapped_libraries'])
  assert py['BLAS_policy']['failures']==0 and len(py['BLAS_policy']['owners'])==8 and all(v['created']==v['verified']for v in py['BLAS_policy']['owners'])
  cap=py['CAPI_result'];assert cap['completed']and cap['all8_owners_joined_before_group_destroy']and cap['normal_CommunicationGroup_close']and len(cap['reports'])==8
  for rank,row in enumerate(cap['reports']):
   stem='/refactor/results/Kimi437-910371.775-node%d-raw/rank%d'%(node,rank);assert get(stem+'.json')==row
   assert row['completed']and row['normal_Runtime_close']and row['both_modules_closed']and row['affinity_restored']and row['local_rank']==rank and row['global_tp_rank']==node*8+rank and row['TP']==16 and row['PP']==1
   assert row['entry_mask']==py['entry_affinity']==row['Runtime_created_mask']==row['restored_mask']and len(row['entry_mask'])>=32 and row['pinned_mask']==row['owner_mask']==row['entry_mask'][4*rank:4*rank+4]
   global_mask_pairs.update((node,c)for c in row['owner_mask']);assert len(row['arms'])==6 and[a['mode']for a in row['arms']]==[0,1,0,1,1,0]
   reference=None
   for arm,a in enumerate(row['arms']):
    assert a['arm']==arm and a['warm']==(arm<2)and a['outer_completion_included']and a['global_barrier_sum']==16 and a['input_and_tail_canary_unchanged']and a['down_up_selected_delta']==[a['mode'],a['mode']]and a['nonfinite']==0
    bits=list(struct.iter_unpack('<H',raw(stem+'-arm%d.bf16'%arm)));assert len(bits)==a['raw_output_elements']==7168;u=[v[0]for v in bits];values=[struct.unpack('<f',struct.pack('<I',v<<16))[0]for v in u];assert all(map(math.isfinite,values))
    if reference is None:reference=(u,values)
    refbits,ref=reference;d2=sum((x-y)**2 for x,y in zip(ref,values));b2=sum(x*x for x in ref);nr=math.sqrt(d2/max(b2,1e-30));err=max(abs(x-y)for x,y in zip(ref,values));different=sum(x!=y for x,y in zip(refbits,u))
    assert math.isclose(nr,a['relative_RMS_vs_first_OFF'],rel_tol=1e-12,abs_tol=1e-14)and err==a['max_abs_vs_first_OFF']and different==a['bit_differences_diagnostic_only']and nr<=.1
    meta=next(v for v in py['raw_outputs']if v['rank']==rank and v['arm']==arm);assert meta['SHA256']==sha(raw(stem+'-arm%d.bf16'%arm))and meta['finite']and math.isclose(math.sqrt(sum(v*v for v in values)/7168),meta['RMS'],rel_tol=1e-12)
    raw_metrics.append(dict(node=node,rank=rank,arm=arm,mode=a['mode'],relative_RMS=nr,max_abs=err,bit_differences=different))
   events=[json.loads(line)for line in raw(stem+'-projection-descriptors.jsonl').decode().splitlines()];saved=next(v for v in py['projection_descriptor_files']if v['local_rank']==rank);assert events==saved['events']and sha(raw(stem+'-projection-descriptors.jsonl'))==saved['SHA256']and len(events)==2
   for which,event in enumerate(events):
    n,k=(3584,7168)if which==0 else(7168,3584);x,weight,y=event['input'],event['weight'],event['output']
    assert event['which']==which and event['arm']==1 and event['phase']=='allocated_before_launch'and event['local_rank']==rank and event['global_tp_rank']==node*8+rank and event['node_rank']==node and event['current_device_read_rc']==0 and event['current_cuda_device']==rank
    assert len(event['guard_predicates'])==21 and all(v is True for v in event['guard_predicates'].values())and event['down_up_hits']==([0,0]if which==0 else[1,0])
    assert all(v['present']and v['data_address']>0 and v['dtype']and v['underlying_Runtime_pool_offset']=='UNKNOWN_not_exposed_publicly'for v in[x,weight,y])
    assert x['shape']==[1,k]and x['strides'][0]>0 and x['strides'][1]==1 and x['nbytes_logical']==k*2 and weight['shape']==[n,k]and weight['strides']==[k,1]and y['shape']==[1,n]and y['strides']==[n,1]and y['logical_offset0_source_guarantee']
    if which==0:assert x['strides']==[14336,1]and not x['is_contiguous_strict']and x['observed_byte_displacement_from_canary_storage']==0 and x['strong_canary_storage_bytes']==28672
    descs.append(dict(node=node,rank=rank,which=which,input_shape=x['shape'],input_strides=x['strides'],input_dtype=x['dtype'],input_strict_contiguous=x['is_contiguous_strict'],canary_byte_displacement=x['observed_byte_displacement_from_canary_storage'],weight_shape=weight['shape'],weight_strides=weight['strides'],output_shape=y['shape'],all21predicates=True,pool_offset='UNKNOWN',source_guarantee_not_pool_observation=y['logical_offset0_source_guarantee']))
   rows.append(row)
  nodes.append(dict(node=node,CPU_checkpoint_load_s=py['CPU_checkpoint_load_s'],CPU_checkpoint_bytes=py['CPU_checkpoint_bytes'],complete_Native_session_s=py['complete_Native_session_s'],complete_process_wall_s=py['complete_process_wall_s'],memory_before_component=py['memory_before_component'],memory_after_component=py['memory_after_component'],memory_scope=py['memory_scope'],per_owner_dual_instance_weight_load_ms=[v['two_private_instances_weight_load_ms']for v in cap['reports']]))
 assert len(rows)==16 and len(raw_metrics)==96 and len(descs)==32 and len(global_mask_pairs)==64
 timing={};per_rank=[]
 for key in['full_MoE_forward_and_completion_ms','checked_call_including_output_readback_save_release_ms']:
  arms=[max(row['arms'][i][key]for row in rows)for i in range(6)];off=[arms[2],arms[5]];on=[arms[3],arms[4]];ratio=statistics.mean(on)/statistics.mean(off);drift=abs(off[1]/off[0]-1)
  timing[key]=dict(max16_by_arm_ms=arms,formal_OFF_ms=off,formal_ON_ms=on,ratio_ON_over_OFF=ratio,gain_pct=(1-ratio)*100,OFF_endpoint_drift_pct=drift*100,adjacent_ratios_ON_over_OFF=[on[0]/off[0],on[1]/off[1]],performance_gate_pass=(1-ratio)>drift,scope='Max16 rank-local elapsed intervals; no cross-node absolute clock subtraction; warm0/1 excluded, allformal2..5 retained')
  for row in rows:
   off=[row['arms'][i][key]for i in[2,5]];on=[row['arms'][i][key]for i in[3,4]];per_rank.append(dict(global_rank=row['global_tp_rank'],metric=key,ratio_ON_over_OFF=statistics.mean(on)/statistics.mean(off),OFF_endpoint_drift_pct=abs(off[1]/off[0]-1)*100))
 assert not timing['full_MoE_forward_and_completion_ms']['performance_gate_pass']
 return dict(audit_passed=True,run_numerics_lifecycle_passed=True,performance_retained=False,decision='STOP_DEFAULT_OFF_NO_MODEL',scope='Actual TP16 PP1 C1 layer1, long_factoryfalse paddedscales; not currentPP2 exactscale Model qualification',archive_SHA256=sha(blob),archive_files=141,original_GPU='910371.775 COMPLETED0/98s',keeper='910407.907 COMPLETED0/122s',post='910371.776 COMPLETED0/18s;bothnodesKFD/PIDemptyportbindable',GPU_manifest_SHA256=MF,library_SHA256=LIB,CPU905_receipt_SHA256=CPUSHA,original_epoch=w['original_start'],normal_Runtime_owners=16,all16_joined_groupclosed=True,projection_dispatches={'down_ON':48,'up_ON':48,'label':'Native successful kernel submission counts plus originalcompletion/finite rawoutputs, not hardware event counters'},raw_results=96,warm_results=32,formal_results=64,first_ON_descriptor_events=32,max_raw_relative_RMS=max(v['relative_RMS']for v in raw_metrics),max_raw_absolute_error=max(v['max_abs']for v in raw_metrics),alert02_count=sum(v['relative_RMS']>.02 for v in raw_metrics),numerical_scope='Baseline point comparison and finite/canary only; .02alert/.10stop are not Model quality bounds, no GPU F64 or bitexact gate',timing=timing,per_rank_intervals=per_rank,raw_numerical_metrics=raw_metrics,actual_descriptor_summary=descs,cold_load_observer_and_memory=nodes,memory_scope='memGetInfo snapshots before/aftercomponent only, not peak/Modelcapacity; dualmodule weight storage explicit',allocated_gpu_card_s=16*98,keeper_CPU_allocated_core_s=2*122,cost_scope='Slurm elapsed includes load/setup/warm/ABBA; native/process subintervals nested and overlap acrossnodes, do not add. First process/shape invocation, not proven physicalGPUcold.',legacy_PLAN_note='Inherited CPU_qualification887 is original prerequisite; current execution identity is GPUmanifest905/422f49 +actualdladdr. Frozen source not edited postrun.',original_failures_preserved=['757','765','769','883','885','GPUv2HOST0intent'],GPU_GO=False,Model_GO=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--observation',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive,a.observation);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps({k:q[k]for k in['audit_passed','run_numerics_lifecycle_passed','performance_retained','decision','max_raw_relative_RMS','allocated_gpu_card_s']}))
