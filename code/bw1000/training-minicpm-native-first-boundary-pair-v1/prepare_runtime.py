"""Reuse qualified157 lifecycle; add only declared internal boundary observation."""
from pathlib import Path
import ast,hashlib,json,subprocess
D=Path(__file__).resolve().parent;B=D.parent;old=B/'training-minicpm-native-first-tail-entry-v1/package';P=D/'package';P.mkdir(exist_ok=False)
F=B/'hygon-training-integration';G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
framework='f9aaaa8f89624fae3c0ed960780b62e1fee375da';label=D.name
original=json.loads((B/'training-minicpm-forward-boundary-HIP-CPU-v1/transport-outcome.json').read_bytes())['value'];cpu=original['CPU-result.json']['value']
root=json.loads((B/'training-minicpm-forward-boundary-HIP-CPU-v1/root-reader-v2-retry2-result.json').read_bytes())['value'];assert root['passed'] and root['owner_identity_independently_verified'] and cpu['tests_run']==26
assert cpu['framework_commit']==framework and original['accounting'][0][0]=='1087656.182'
for name in ['scope.py','controller.py','native.py','owned_processes.py','snapshot.py','slurm_accounting.py','original-launch.json','INITIAL_CHECKPOINT.json','audit_pair_CPU.py','verify_package.py','receive_once.py','read_original_terminal.py']:
 raw=(old/name).read_bytes()
 if name.endswith('.py'):
  text=raw.decode().replace('training-minicpm-native-first-tail-entry-v1',label)
  if name=='scope.py':text=text.replace('PORT=29897','PORT=29905')
  ast.parse(text);raw=text.encode()
 (P/name).write_bytes(raw)
assert (P/'snapshot.py').read_bytes()==(old/'snapshot.py').read_bytes()
assert (P/'owned_processes.py').read_bytes()==(old/'owned_processes.py').read_bytes()
assert (P/'slurm_accounting.py').read_bytes()==(old/'slurm_accounting.py').read_bytes()
deps=json.loads((old/'DEPENDENCIES.json').read_bytes());R='/public/home/aclsylqidf/hygon-inference-refactor-20261014';qualified=R+'/results/training-minicpm-forward-boundary-HIP-CPU-v1'
deps['framework']=dict(framework_commit=framework,framework_tree=subprocess.check_output(['git','-c','safe.directory='+F.as_posix(),'-C',str(F),'rev-parse',framework+'^{tree}']).decode().strip(),source_directory=qualified+'/framework',source_manifest_SHA256=cpu['framework_manifest_SHA256'],source_files=353,CPU_step='1087656.182',CPU_result_SHA256=original['CPU-result.json']['SHA256'],CPU_result_path=qualified+'/evidence/CPU-result.json',qualification='ActualCPU182 full353sources and26affectedhook/resume tests independentlyclosed;realnative52resolveranddevicevaluesunqualified.')
deps['framework_files_manifest']=qualified+'/source/FRAMEWORK_FILES.json';deps['swift']['swift_script']=qualified+'/framework/vlm/runtime/swift_entry.py'
deps['minicpm_provider_source_SHA256']='a37ef94154e0b0ef3918ae60dfb43e871a5964d6a3cf48cd501dc36bf863119b'
deps['boundary_CPU_providers']=cpu['actual_runtime_providers'];deps['new_CPU182_replayed']=False
(P/'DEPENDENCIES.json').write_bytes((json.dumps(deps,ensure_ascii=False,indent=2)+'\n').encode())
p=P/'verify_package.py';s=p.read_text(encoding='utf8').replace('0a3bd62c1d941f2993052eee0449fb248969177b',framework).replace('==308','==353').replace('framework_files=308','framework_files=353').replace("'1087656.155'","'1087656.182'").replace("CPU_prerequisite='1087656.155'","CPU_prerequisite='1087656.182'").replace("cpu['tests_run']==33","cpu['tests_run']==26").replace('bc8551eac870bc45699ca8149fe69b818327083e61f0e30bd9000b1472f5ec38',original['CPU-result.json']['SHA256'])
needle=" tf=Path(deps['processor']['host_transformers_site']);assert sha(tf/'transformers/models/qwen3_5/modeling_qwen3_5.py')==deps['qwen_provider_source_SHA256']";assert needle in s
s=s.replace(needle,needle+"\n assert sha(tf/'transformers/models/minicpmv4_6/modeling_minicpmv4_6.py')==deps['minicpm_provider_source_SHA256']\n for pin in deps['boundary_CPU_providers'].values():\n  name=pin['path'];path=(tf/name[len('/minicpm-transformers/'):]) if name.startswith('/minicpm-transformers/') else Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/containers/dtk26-sandbox-r1')/name.lstrip('/')\n  assert path.stat().st_size==pin['bytes'] and sha(path)==pin['SHA256']")
ast.parse(s);p.write_bytes(s.encode())
p=P/'native.py';s=p.read_text(encoding='utf8')
s=s.replace("entry_state_step=20,diagnostic_stop_step=21)","entry_state_step=20,diagnostic_stop_step=21,first_divergence_trace=True)")
needle="   if owner.is_file():item['owner_SHA256']=sha(owner)";assert needle in s
s=s.replace(needle,"""   boundary=root/'forward-boundaries.json';item['boundary_present']=boundary.is_file()
   if boundary.is_file() and path.is_file():
    assert 0<boundary.stat().st_size<=2*1024**2 and sha(boundary)==value['boundary_trace_SHA256']
    trace=json.loads(boundary.read_bytes());assert trace['complete'] and not trace['failed'] and trace['planned_count']==trace['completed_boundaries']==52 and len(trace['records'])==52 and trace['global_step']==20
    assert trace['forward_ordinal']==(41 if name=='control' else 1) and trace['observer_RNG_unchanged'] and trace['raw_values_exported'] is False
    assert 0<=trace['copied_logical_bytes']<=trace['max_logical_bytes']<=2*1024**3
    item.update(boundary_bytes=boundary.stat().st_size,boundary_SHA256=sha(boundary),boundary_records=52,boundary_logical_bytes=trace['copied_logical_bytes'],boundary_observer_seconds=trace['observer_seconds'])
"""+needle)
s=s.replace("r['entry_present'] for r in records","r['entry_present'] and r['boundary_present'] for r in records")
s=s.replace("independent_pair_audit_pending=True", "planned_boundary_traces=16,planned_boundary_records=832,independent_pair_audit_pending=True")
assert s.count('first_divergence_trace=True')==2
ast.parse(s);p.write_bytes(s.encode())
p=P/'audit_pair_CPU.py';s=p.read_text(encoding='utf8')
s=s.replace('from vlm.runtime.resume_entry_pair import compare_forward_entry_pair','from vlm.runtime.resume_entry_pair import compare_forward_entry_pair\n from vlm.runtime.forward_boundary_trace import compare_boundary_traces')
s=s.replace('loss_deltas=[];entry_costs=[]','loss_deltas=[];entry_costs=[];boundary_rows=[];boundary_pins=[];boundary_costs=[]')
s=s.replace('values=[];entries=[]','values=[];entries=[];boundaries=[]')
needle="   entry_costs.append(";pos=s.index(needle)
insert="""   boundary_path=base/'forward-boundaries.json';boundary_raw=boundary_path.read_bytes();assert 0<len(boundary_raw)<=2*1024**2
   boundary=json.loads(boundary_raw);assert sha(boundary_path)==value['boundary_trace_SHA256']==expected['boundary_SHA256'] and len(boundary_raw)==expected['boundary_bytes']
   assert boundary['complete'] and not boundary['failed'] and boundary['planned_count']==boundary['completed_boundaries']==52 and len(boundary['records'])==52 and boundary['global_step']==20 and boundary['forward_ordinal']==(41 if arm=='control' else 1)
   assert boundary['observer_RNG_unchanged'] and boundary['raw_values_exported'] is False and 0<=boundary['copied_logical_bytes']<=boundary['max_logical_bytes']<=2*1024**3
   boundaries.append(boundary);boundary_pins.append(dict(arm=arm,rank=rank,bytes=len(boundary_raw),SHA256=sha(boundary_path),records=52))
   boundary_costs.append(dict(arm=arm,rank=rank,copied_logical_bytes=boundary['copied_logical_bytes'],observer_seconds=boundary['observer_seconds'],total_wall_seconds=boundary['total_wall_seconds']))
"""
s=s[:pos]+insert+s[pos:]
needle="  rank_rows.append(compare_forward_entry_pair(values[0],values[1],entries[0],entries[1]))";assert needle in s
s=s.replace(needle,needle+"\n  boundary_rows.append(dict(rank=rank,**compare_boundary_traces(boundaries[0],boundaries[1])))")
s=s.replace('rank_rows=rank_rows,original_rank_records=raw_records,','rank_rows=rank_rows,original_rank_records=raw_records,boundary_rank_rows=boundary_rows,original_boundary_records=boundary_pins,boundary_observer_costs=boundary_costs,planned_boundary_traces=16,planned_boundary_records=832,completed_boundary_traces=len(boundary_pins),completed_boundary_records=sum(p[\'records\'] for p in boundary_pins),')
# All16 declared slots survive early native/audit failure; retain existing partial
# progress without calling a saved trace bound/qualified before its witness SHA.
needle="except BaseException as error:\n result=dict(passed=False";assert needle in s
new="""except BaseException as error:
 partial=[]
 for arm in ('control','restored'):
  for rank in range(8):
   folder=ROOT/arm/'train/resume-witness'/('rank-%02d'%rank);item=dict(arm=arm,rank=rank,planned_boundaries=52,final_present=(folder/'forward-boundaries.json').is_file(),progress_present=(folder/'forward-boundaries-progress.json').is_file(),independently_bound=False)
   for name in ('forward-boundaries.json','forward-boundaries-progress.json'):
    p=folder/name
    if p.is_file():
     try:
      assert 0<p.stat().st_size<=2*1024**2;raw=p.read_bytes();v=json.loads(raw);item[name]=dict(bytes=len(raw),SHA256=sha(p),completed_boundaries=v.get('completed_boundaries'),planned_count=v.get('planned_count'),failed=v.get('failed'),complete=v.get('complete'),closed=v.get('closed'))
     except BaseException as parse_error:item[name]=dict(error_type=type(parse_error).__name__)
   partial.append(item)
 result=dict(passed=False"""
s=s.replace(needle,new).replace('completed_rank_comparisons=len(rank_rows),raw_records_seen=raw_records,','completed_rank_comparisons=len(rank_rows),raw_records_seen=raw_records,planned_boundary_traces=16,planned_boundary_records=832,partial_boundary_inventory=partial,')
ast.parse(s);p.write_bytes(s.encode())
p=P/'read_original_terminal.py';s=p.read_text(encoding='utf8').replace('==308','==353').replace('0a3bd62c1d941f2993052eee0449fb248969177b',framework)
s=s.replace('pair_rows=[];records_verified=entries_verified=0','pair_rows=[];records_verified=entries_verified=0;boundary_rows=[];boundary_records_verified=0;boundary_files_verified=0')
needle=" compare=runpy.run_path(str(source/'vlm/runtime/resume_entry_pair.py'))['compare_forward_entry_pair']";assert needle in s
s=s.replace(needle,needle+"\n boundary_compare=runpy.run_path(str(source/'vlm/runtime/forward_boundary_trace.py'))['compare_boundary_traces']")
s=s.replace('values=[];entries=[]','values=[];entries=[];boundaries=[]')
needle="  pair_rows.append(compare(values[0],values[1],entries[0],entries[1]))";assert needle in s
s=s.replace(needle,"""   boundary_path=folder/'forward-boundaries.json';boundary_raw=boundary_path.read_bytes();assert 0<len(boundary_raw)<=2*1024**2
   pin=next(r for r in audit['original_boundary_records'] if r['rank']==rank and r['arm']==arm)
   assert len(boundary_raw)==pin['bytes'] and sha(boundary_path)==pin['SHA256']==value['boundary_trace_SHA256']
   trace=json.loads(boundary_raw);assert trace['planned_count']==trace['completed_boundaries']==52 and trace['global_step']==20 and trace['forward_ordinal']==(41 if arm=='control' else 1)
   boundaries.append(trace);boundary_files_verified+=1;boundary_records_verified+=len(trace['records'])
"""+needle+"\n  boundary_rows.append(dict(rank=rank,**boundary_compare(boundaries[0],boundaries[1])))")
s=s.replace(" assert pair_rows==audit['rank_rows'] and len(audit['original_rank_records'])==16"," assert pair_rows==audit['rank_rows'] and len(audit['original_rank_records'])==16\n assert boundary_rows==audit['boundary_rank_rows'] and boundary_files_verified==audit['completed_boundary_traces']==16 and boundary_records_verified==audit['completed_boundary_records']==audit['planned_boundary_records']==832")
s=s.replace('recomputed_pair_rows=pair_rows,','recomputed_pair_rows=pair_rows,recomputed_boundary_rows=boundary_rows,boundary_files_verified=boundary_files_verified,boundary_records_verified=boundary_records_verified,planned_boundary_traces=16,planned_boundary_records=832,boundary_observer_costs=audit.get(\'boundary_observer_costs\'),')
ast.parse(s);p.write_bytes(s.encode())
# The new scripts use the same controller/deadlines/accounting, not a new service.
for name in ('controller.py','receive_once.py','snapshot.py','owned_processes.py','slurm_accounting.py'):ast.parse((P/name).read_bytes())
print(json.dumps(dict(runtime_files=len(list(P.iterdir())),framework_commit=framework,actual_GPU_launches=0,planned_traces=16,planned_boundary_records=832,qualified_helpers_unchanged=True)))
