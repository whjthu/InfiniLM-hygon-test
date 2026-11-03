"""Original778 full-MoE raw audit. Stdlib, no SDK/weights/job execution."""
from pathlib import Path
import argparse,base64,gzip,hashlib,json,math,statistics,struct
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
MF='1ffd1baf4ed10f63b289651b3f1d8b3c4dd8170a42d866ce2eef9ddde25f9561'
LIB='fe1d0b59f69b74fa973f10a1623216423b00e07c55d91eb6d5afde82df041ad9'
CPUSHA='e11cac669cb6c650879c4b32d8e8d06afe407e125d5a7e18471f61fb72cc62e8'
sha=lambda b:hashlib.sha256(b).hexdigest()
def audit(archive,observation):
 blob=Path(archive).read_bytes();assert sha(blob)=='3129f7e1611d6f4f7649e33fc3b6830d07580152dcb2744997bef859b0b707f6'
 seal=json.loads(gzip.decompress(blob));assert seal['new_jobs']==seal['new_GPU']==0 and len(seal['files'])==125
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
 for step,label in [('910371.777','Kimi-r440-full-MoE-rowshard-v5-before'),('910371.778','Kimi-r440-full-MoE-rowshard-v5'),('910371.779','Kimi-r440-full-MoE-rowshard-v5-post'),('910407.914','Kimi-r440-full-MoE-rowshard-v5-keeper')]:
  assert accounting[step][1:4]==[label,'COMPLETED','0:0']
 assert observed['watch']['SHA256']==sha(raw('/refactor/results/Kimi440-watch-v5.json'))
 w=get('/refactor/results/Kimi440-watch-v5.json');assert w['completed']and w['passed']and w['normal16']and w['all_96_forward_results_persisted']and w['srun_exit0']and w['Slurm_COMPLETED0']and not w.get('forced_cleanup')and not w.get('error')
 assert w['actual_step']=='910371.778'and w['GPU_source_manifest_SHA256']==MF and w['profile']==0 and w['no_retry']
 assert len(w['post'])==2 and all(v['passed']and v['actual_step']=='910371.779'and not v['KFD_pids']and not v['matching_GPU_driver_PIDs']and not v['expected_PIDs_still_exist']and v['passive_master_port40917_bindable']for v in w['post'])
 assert len({(v['node'],pci)for v in w['before']for pci in v['physical_PCI']})==16
 mpath='/refactor/incoming/Kimi-lowrank-TP-rowshard-r440/source-span-v2/gpu-v5/GPU_SOURCE_MANIFEST.json';assert sha(raw(mpath))==MF;m=get(mpath)
 assert m['GPU_GO']and not m['Model_GO']and m['modes']==[0,1,0,1,1,0]and m['total_forward_calls']==96 and m['library_SHA256']==LIB and m['CPU_receipt_SHA256']==CPUSHA
 assert sha(raw(m['CPU_receipt']))==CPUSHA;cpu=get(m['CPU_receipt']);assert cpu['passed']and cpu['actual_CPU']=='910407.912'and cpu['library_SHA256']==LIB and len(cpu['compiled'])==2
 assert cpu['reused_CPU910_pure_contract']['result']['changed_contract_cases']==27 and not cpu['reused_CPU910_pure_contract']['reexecuted']and cpu['actual_source_contract']['whole_original_MoE_body_except_exact_five_projection_calls']
 assert sha(raw(m['owner_source_root']+'/SOURCE_MANIFEST.json'))==m['owner_source_SHA256']
 rows=[];raw_metrics=[];scopes=[];nodes=[];global_mask_pairs=set();replicated=[]
 for node in[0,1]:
  py=get('/refactor/results/Kimi440-910371.778-node%d.json'%node)
  assert py['completed']and py['passed']and py['normal_close']and py['returncode']==0 and py['actual_step']=='910371.778'and py['node_rank']==node and py['source_manifest_SHA256']==MF and py['original_epoch']==w['original_start']
  assert py['node']==['a02r3n03','a02r3n07'][node]and py['TP']==16 and py['PP']==1 and py['actual_C1']and py['padded_scale_storage']and not py['long_factory']and not py['Model']and py['HTTP']==0
  replicated.append(py['projection_checkpoint_identity'])
  for role,expected in m['projection_checkpoint_expected'].items():assert all(py['projection_checkpoint_identity'][role][k]==v for k,v in expected.items())
  assert py['complete_Native_session_s']<=300
  assert py['CPU_weights_and_input_unchanged']and py['input']['SHA256']=='3f84e8d80178c05212edcc5b703f907290df20925684fb610838c6d1f259ecca'
  assert py['visible_count']==8 and len(set(py['physical_PCI_by_visible_device']))==8 and len(py['raw_outputs'])==48
  assert all(v['SHA256']==LIB for v in py['actual_CABI_bindings'].values())
  for p,h in m['mapped_dependencies_SHA256'].items():assert py['mapped_libraries'].get(p)==h,(p,py['mapped_libraries'])
  assert py['BLAS_policy']['failures']==0 and len(py['BLAS_policy']['owners'])==8 and all(v['created']==v['verified']for v in py['BLAS_policy']['owners'])
  cap=py['CAPI_result'];assert cap['completed']and cap['all8_owners_joined_before_group_destroy']and cap['normal_CommunicationGroup_close']and len(cap['reports'])==8
  for rank,row in enumerate(cap['reports']):
   stem='/refactor/results/Kimi440-910371.778-node%d-raw/rank%d'%(node,rank);assert get(stem+'.json')==row
   assert row['completed']and row['normal_Runtime_close']and row['both_modules_closed']and row['affinity_restored']and row['local_rank']==rank and row['global_tp_rank']==node*8+rank and row['TP']==16 and row['PP']==1
   assert row['entry_mask']==py['entry_affinity']==row['Runtime_created_mask']==row['restored_mask']and len(row['entry_mask'])>=32 and row['pinned_mask']==row['owner_mask']==row['entry_mask'][4*rank:4*rank+4]
   global_mask_pairs.update((node,c)for c in row['owner_mask']);assert len(row['arms'])==6 and[a['mode']for a in row['arms']]==[0,1,0,1,1,0]
   reference=None
   for arm,a in enumerate(row['arms']):
    assert a['arm']==arm and a['warm']==(arm<2)and a['outer_completion_included']and a['global_barrier_sum']==16 and a['input_and_tail_canary_unchanged']and a['down_up_selected_delta']==[a['mode'],a['mode']]and a['nonfinite']==0 and a['pending_Tensor_handles_before_completion']==(14 if a['mode']else 0)and a['existing_completion_and_lease_release_timed']
    bits=list(struct.iter_unpack('<H',raw(stem+'-arm%d.bf16'%arm)));assert len(bits)==a['raw_output_elements']==7168;u=[v[0]for v in bits];values=[struct.unpack('<f',struct.pack('<I',v<<16))[0]for v in u];assert all(map(math.isfinite,values))
    if reference is None:reference=(u,values)
    refbits,ref=reference;d2=sum((x-y)**2 for x,y in zip(ref,values));b2=sum(x*x for x in ref);nr=math.sqrt(d2/max(b2,1e-30));err=max(abs(x-y)for x,y in zip(ref,values));different=sum(x!=y for x,y in zip(refbits,u))
    assert math.isclose(nr,a['relative_RMS_vs_first_OFF'],rel_tol=1e-12,abs_tol=1e-14)and err==a['max_abs_vs_first_OFF']and different==a['bit_differences_diagnostic_only']and nr<=.1
    meta=next(v for v in py['raw_outputs']if v['rank']==rank and v['arm']==arm);assert meta['SHA256']==sha(raw(stem+'-arm%d.bf16'%arm))and meta['finite']and math.isclose(math.sqrt(sum(v*v for v in values)/7168),meta['RMS'],rel_tol=1e-12)
    raw_metrics.append(dict(node=node,rank=rank,arm=arm,mode=a['mode'],relative_RMS=nr,max_abs=err,bit_differences=different))
   first=row['arms'][1];assert first['mode']==1 and first['down_up_selected_delta']==[1,1]and first['pending_Tensor_handles_before_completion']==14
   saved=next(v for v in py['projection_scope_summaries']if v['local_rank']==rank);assert saved==dict(local_rank=rank,global_tp_rank=node*8+rank,firstON=first)
   scopes.append(dict(global_tp_rank=node*8+rank,firstON_counts=[1,1],firstON_retained_Tensor_handles=14,original_completion_and_lease_release_timed=True,not_unique_allocation_count=True))
   rows.append(row)
  nodes.append(dict(node=node,CPU_checkpoint_load_s=py['CPU_checkpoint_load_s'],CPU_checkpoint_bytes=py['CPU_checkpoint_bytes'],complete_Native_session_s=py['complete_Native_session_s'],complete_process_wall_s=py['complete_process_wall_s'],memory_before_component=py['memory_before_component'],memory_after_component=py['memory_after_component'],memory_scope=py['memory_scope'],per_owner_dual_instance_weight_load_ms=[v['two_private_instances_weight_load_ms']for v in cap['reports']]))
 assert len(replicated)==2 and replicated[0]==replicated[1]
 assert len(rows)==16 and len(raw_metrics)==96 and len(scopes)==16 and len(global_mask_pairs)==64
 timing={};per_rank=[]
 for key in['full_MoE_forward_and_completion_ms','checked_call_including_output_readback_save_release_ms']:
  arms=[max(row['arms'][i][key]for row in rows)for i in range(6)];off=[arms[2],arms[5]];on=[arms[3],arms[4]];ratio=statistics.mean(on)/statistics.mean(off);drift=abs(off[1]/off[0]-1)
  timing[key]=dict(max16_by_arm_ms=arms,formal_OFF_ms=off,formal_ON_ms=on,ratio_ON_over_OFF=ratio,gain_pct=(1-ratio)*100,OFF_endpoint_drift_pct=drift*100,adjacent_ratios_ON_over_OFF=[on[0]/off[0],on[1]/off[1]],performance_gate_pass=(1-ratio)>drift,scope='Max16 rank-local elapsed intervals; no cross-node absolute clock subtraction; warm0/1 excluded, allformal2..5 retained')
  for row in rows:
   off=[row['arms'][i][key]for i in[2,5]];on=[row['arms'][i][key]for i in[3,4]];per_rank.append(dict(global_rank=row['global_tp_rank'],metric=key,ratio_ON_over_OFF=statistics.mean(on)/statistics.mean(off),OFF_endpoint_drift_pct=abs(off[1]/off[0]-1)*100))
 assert not timing['full_MoE_forward_and_completion_ms']['performance_gate_pass']
 return dict(audit_passed=True,run_numerics_lifecycle_passed=True,performance_retained=False,decision='STOP_DEFAULT_OFF_NO_MODEL',scope='Actual TP16 PP1 C1 layer1, long_factoryfalse paddedscales; not currentPP2 exactscale Model qualification',archive_SHA256=sha(blob),archive_files=125,original_GPU='910371.778 COMPLETED0/96s',keeper='910407.914 COMPLETED0/121s',post='910371.779 COMPLETED0/18s;bothnodesKFD/PIDemptyportbindable',GPU_manifest_SHA256=MF,library_SHA256=LIB,CPU912_receipt_SHA256=CPUSHA,original_epoch=w['original_start'],normal_Runtime_owners=16,all16_joined_groupclosed=True,projection_dispatches={'down_ON':48,'up_ON':48,'AllGather_rank_enqueue_calls':96,'world_AllGather_instances':6,'label':'Native successful guarded row-shard projection+collective submissions and full completion, not hardware events or HBM traffic'},raw_results=96,warm_results=32,formal_results=64,first_ON_scope_records=16,max_raw_relative_RMS=max(v['relative_RMS']for v in raw_metrics),max_raw_absolute_error=max(v['max_abs']for v in raw_metrics),alert02_count=sum(v['relative_RMS']>.02 for v in raw_metrics),numerical_scope='Baseline point comparison and finite/canary only; .02alert/.10stop are not Model quality bounds, no GPU F64 or bitexact gate',timing=timing,per_rank_intervals=per_rank,raw_numerical_metrics=raw_metrics,actual_first_ON_scope_summary=scopes,cold_load_observer_and_memory=nodes,memory_scope='memGetInfo snapshots before/aftercomponent only, not peak/Modelcapacity; dualmodule weight storage explicit',allocated_gpu_card_s=16*96,keeper_CPU_allocated_core_s=2*121,cost_scope='Slurm elapsed includes load/setup/warm/ABBA; native/process subintervals nested and overlap acrossnodes, do not add. First process/shape invocation, not proven physicalGPUcold.',source_scope_note='CPU912 fe1d/GPUMF1ffd/actualdladdr; original Dense+two AllGather,14 retained handles released in primary timer. Old775/GEMV descriptors are not imported as new observations.',original_failures_preserved=['CPU909','CPU910','old437775_no_gain'],GPU_GO=False,Model_GO=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--observation',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive,a.observation);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf8');print(json.dumps({k:q[k]for k in['audit_passed','run_numerics_lifecycle_passed','performance_retained','decision','max_raw_relative_RMS','allocated_gpu_card_s']}))
