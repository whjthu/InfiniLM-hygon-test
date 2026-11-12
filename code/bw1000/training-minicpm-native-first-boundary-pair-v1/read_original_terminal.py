"""Read the exact first-tail terminal; diagnostic success never replaces recovery."""
from pathlib import Path
import datetime,hashlib,json,runpy,stat,subprocess
ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native-first-boundary-pair-v1');label=ROOT.name

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
intent=json.loads((ROOT/'intent.json').read_bytes());mf=json.loads((ROOT/'src/SOURCE_FILES.json').read_bytes());assert intent['epoch']+1800==intent['deadline'] and sha(ROOT/'src/SOURCE_FILES.json')==intent['source_manifest_SHA256']
for name,pin in mf['files'].items():assert (ROOT/'src'/name).stat().st_size==pin['bytes'] and sha(ROOT/'src'/name)==pin['SHA256']
raw=(ROOT/'outcome.json').read_bytes();out=json.loads(raw);assert out['epoch']==intent['epoch'] and out['deadline']==intent['deadline'] and out['claim_commit']==intent['claim_commit']
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
rows=[s.split('|') for s in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if any('|'+label+suffix+'|' in s for suffix in ('','-pre','-post','-audit'))];account={r[1]:r for r in rows};assert len(account)==len(rows)
for run in out['runs']:
 assert run['accounting_columns']==columns and sha(ROOT/(run['name']+'.log'))==run['log_SHA256']
 if run['terminal']['terminal_visible']:assert account.get(run['name'])==run['terminal']['row'] and account[run['name']][2].split()[0] in terminal
for name in ('assigned.json','native-outcome.json','actual-versions.json','source-and-asset-preflight.json','independent-pair-audit.json'):
 if name in out:
  record=out[name];p=ROOT/name;assert p.stat().st_size==record['bytes'] and sha(p)==record['SHA256'] and json.loads(p.read_bytes())==record['value']
gpu=account.get(label);gpu_terminal=bool(gpu and gpu[2].split()[0] in terminal);post=account.get(label+'-post');before=out.get('before');after=out.get('after')
clean=bool(gpu_terminal and post and post[2:4]==['COMPLETED','0:0'] and before and after and before['pci']==after['pci'] and before['boot_id']==after['boot_id'] and before['full_fd_inventory_performed'] is True and after['full_fd_inventory_performed'] is True and after['owned_processes_remaining']==[] and after['kfd_holder_count']==0 and after['master_port_available'])
native=out.get('native-outcome.json',{}).get('value',{});audit=out.get('independent-pair-audit.json',{}).get('value',{});pair_rows=[];records_verified=entries_verified=0;boundary_rows=[];boundary_records_verified=0;boundary_files_verified=0
if audit.get('passed'):
 assert gpu_terminal and clean and len(native['arms'])==2 and native['normal_native_exit'] and audit['completed'] and audit['compared_ranks']==8 and audit['planned_tail_forwards']==32
 auditrow=account[label+'-audit'];assert auditrow[2:4]==['COMPLETED','0:0'] and auditrow[5]=='f11r4n10' and audit['owner']['step']==auditrow[0] and audit['GPU_cards']==0 and audit['CUDA_initialized'] is False
 deps=json.loads((ROOT/'src/DEPENDENCIES.json').read_bytes());fmp=Path(deps['framework_files_manifest']);assert sha(fmp)==deps['framework']['source_manifest_SHA256'];fm=json.loads(fmp.read_bytes());source=Path(deps['framework']['source_directory']);assert len(fm['files'])==353 and fm['framework_commit']=='f9aaaa8f89624fae3c0ed960780b62e1fee375da'
 for name,pin in fm['files'].items():
  p=source/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode']
 compare=runpy.run_path(str(source/'vlm/runtime/resume_entry_pair.py'))['compare_forward_entry_pair']
 boundary_compare=runpy.run_path(str(source/'vlm/runtime/forward_boundary_trace.py'))['compare_boundary_traces']
 for rank in range(8):
  values=[];entries=[];boundaries=[]
  for arm in ('control','restored'):
   folder=ROOT/arm/'train/resume-witness'/('rank-%02d'%rank);p=folder/'result.json';raw_rank=p.read_bytes();assert 0<len(raw_rank)<2*1024**2;entry=next(r for r in audit['original_rank_records'] if r['rank']==rank and r['arm']==arm);assert len(raw_rank)==entry['bytes'] and sha(p)==entry['SHA256'];value=json.loads(raw_rank);e=folder/'forward-entry-state.json';entry_raw=e.read_bytes();assert 0<len(entry_raw)<=8*1024**2 and len(entry_raw)==entry['entry_bytes'] and sha(e)==entry['entry_SHA256']==value['entry_state_SHA256'];assert sha(folder/'owner.json')==entry['owner_SHA256'];values.append(value);entries.append(entry_raw);records_verified+=1;entries_verified+=1
   boundary_path=folder/'forward-boundaries.json';boundary_raw=boundary_path.read_bytes();assert 0<len(boundary_raw)<=2*1024**2
   pin=next(r for r in audit['original_boundary_records'] if r['rank']==rank and r['arm']==arm)
   assert len(boundary_raw)==pin['bytes'] and sha(boundary_path)==pin['SHA256']==value['boundary_trace_SHA256']
   trace=json.loads(boundary_raw);assert trace['planned_count']==trace['completed_boundaries']==52 and trace['global_step']==20 and trace['forward_ordinal']==(41 if arm=='control' else 1)
   boundaries.append(trace);boundary_files_verified+=1;boundary_records_verified+=len(trace['records'])
  pair_rows.append(compare(values[0],values[1],entries[0],entries[1]))
  boundary_rows.append(dict(rank=rank,**boundary_compare(boundaries[0],boundaries[1])))
 assert pair_rows==audit['rank_rows'] and len(audit['original_rank_records'])==16
 assert boundary_rows==audit['boundary_rank_rows'] and boundary_files_verified==audit['completed_boundary_traces']==16 and boundary_records_verified==audit['completed_boundary_records']==audit['planned_boundary_records']==832
 assert set(audit['checkpoint_sources'])=={'control20'}
 for key,record in audit['checkpoint_sources'].items():
  folder=Path(record['path']);assert folder.resolve().is_relative_to(ROOT.resolve()) and len(record['files'])==14
  for name,pin in record['files'].items():
   p=folder/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 assert audit['input_equal_tail_forwards']==sum(r['input_equal_forwards'] for r in pair_rows) and audit['exact_loss_tail_forwards']==sum(r['exact_loss_forwards'] for r in pair_rows) and audit['first_tail_exact_loss_ranks']==sum(r['first_tail_loss_exact'] for r in pair_rows)
 assert audit['entry_equal_rank_counts']=={key:sum(r['entry_fields'][key] for r in pair_rows) for key in pair_rows[0]['entry_fields']}
 assert not audit['full25_native_recovery_compared'] and not audit['recovery_qualified'] and not audit['numerical_cause_attributed']
progress=[]
for arm in ('control','restored'):
 for rank in range(8):
  p=ROOT/arm/'train/resume-witness'/('rank-%02d'%rank)/'progress.json';row=dict(arm=arm,rank=rank,present=p.is_file())
  if p.is_file():
   v=json.loads(p.read_bytes());row.update(inputs=v['inputs'],outputs=v['outputs'],completed_steps=v['completed_steps'],failed=v['failed'],RNG_loads=v['RNG_loads'],saved_checkpoints=v['saved_checkpoints'],entry_state_captured=v['entry_state_captured'],diagnostic_stop_requested=v['diagnostic_stop_requested'])
  progress.append(row)
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_original_first_tail_terminal_verified=True,claim=intent['claim_commit'],request_id=intent['request_id'],epoch=intent['epoch'],deadline=intent['deadline'],original_outcome_SHA256=sha(ROOT/'outcome.json'),accounting_columns=columns,accounting=rows,GPU_terminal=gpu_terminal,original_keeper_passed=out['keeper_passed'],independent_HOST_post_cleanup_clean=clean,original_native_arms=[{k:v for k,v in arm.items() if k!='rank_records'} for arm in native.get('arms',[])],original_error=out.get('error'),native_error=native.get('error'),audit_completed=audit.get('completed'),audit_error_type=audit.get('error_type'),rank_record_bytes_SHA_verified=records_verified,entry_record_bytes_SHA_verified=entries_verified,recomputed_pair_rows=pair_rows,recomputed_boundary_rows=boundary_rows,boundary_files_verified=boundary_files_verified,boundary_records_verified=boundary_records_verified,planned_boundary_traces=16,planned_boundary_records=832,boundary_observer_costs=audit.get('boundary_observer_costs'),original_audit_SHA256=sha(ROOT/'independent-pair-audit.json') if (ROOT/'independent-pair-audit.json').exists() else None,entry_comparison={k:audit.get(k) for k in ('input_equal_tail_forwards','exact_loss_tail_forwards','first_tail_exact_loss_ranks','entry_equal_rank_counts','registered_state_equal_ranks','final_observed_training_state_equal_ranks','entry_observer_costs','loss_magnitude_deltas')},progress=progress,full25_recovery_compared=False,numerical_cause_attributed=False,recovery_qualified=False,formal_performance=False,cross_platform_numerical_qualified=False,heldout_qualified=False,Model_GO=False)))
