"""Read the exact original pair terminal; failures and cleanup stay independent."""
from pathlib import Path
import datetime,hashlib,json,runpy,socket,stat,subprocess
ROOT=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-paired-native-recovery-v1');label=ROOT.name

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
native=out.get('native-outcome.json',{}).get('value',{});audit=out.get('independent-pair-audit.json',{}).get('value',{});pair_rows=[];records_verified=0
if audit.get('passed'):
 assert gpu_terminal and clean and len(native['arms'])==2 and native['normal_native_exit'] and audit['completed'] and audit['compared_ranks']==8 and audit['planned_tail_forwards']==160
 auditrow=account[label+'-audit'];assert auditrow[2:4]==['COMPLETED','0:0'] and auditrow[5]=='f11r4n10' and audit['owner']['step']==auditrow[0] and audit['GPU_cards']==0 and audit['CUDA_initialized'] is False
 deps=json.loads((ROOT/'src/DEPENDENCIES.json').read_bytes());fmp=Path(deps['framework_files_manifest']);assert sha(fmp)==deps['framework']['source_manifest_SHA256'];fm=json.loads(fmp.read_bytes());source=Path(deps['framework']['source_directory'])
 for name,pin in fm['files'].items():
  p=source/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode']
 compare=runpy.run_path(str(source/'vlm/runtime/resume_pair.py'))['compare_resume_pair']
 for rank in range(8):
  values=[]
  for arm in ('control','restored'):
   p=ROOT/arm/'train/resume-witness'/('rank-%02d'%rank)/'result.json';raw_rank=p.read_bytes();assert len(raw_rank)<2*1024**2;entry=next(r for r in audit['original_rank_records'] if r['rank']==rank and r['arm']==arm);assert len(raw_rank)==entry['bytes'] and sha(p)==entry['SHA256'];values.append(json.loads(raw_rank));records_verified+=1
  pair_rows.append(compare(*values))
 assert pair_rows==audit['rank_rows']
 for key,record in audit['checkpoint_sources'].items():
  folder=Path(record['path']);assert folder.resolve().is_relative_to(ROOT.resolve()) and len(record['files'])==14
  for name,pin in record['files'].items():
   p=folder/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256']
 assert audit['input_equal_tail_forwards']==sum(r['input_equal_forwards'] for r in pair_rows) and audit['exact_loss_tail_forwards']==sum(r['exact_loss_forwards'] for r in pair_rows)
 if audit['same_platform_recovery_window_exact']:assert all(r['all_recorded_tail_inputs_equal'] and r['all_recorded_tail_losses_exact'] and r['all_recorded_tail_states_exact'] and r['exact_observed_runtime_cursors']==20 for r in pair_rows) and all(audit['final_checkpoint_state_exact'].values()) and all(audit['final_per_rank_RNG_exact'].values())
progress=[]
for arm in ('control','restored'):
 for rank in range(8):
  p=ROOT/arm/'train/resume-witness'/('rank-%02d'%rank)/'progress.json'
  if p.is_file():
   row=json.loads(p.read_bytes());progress.append(dict(arm=arm,rank=rank,inputs=row['inputs'],outputs=row['outputs'],completed_steps=row['completed_steps'],failed=row['failed'],RNG_loads=row['RNG_loads'],saved_checkpoints=row['saved_checkpoints']))
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_original_pair_terminal_verified=True,claim=intent['claim_commit'],request_id=intent['request_id'],epoch=intent['epoch'],deadline=intent['deadline'],original_outcome_SHA256=sha(ROOT/'outcome.json'),accounting_columns=columns,accounting=rows,GPU_terminal=gpu_terminal,original_keeper_passed=out['keeper_passed'],independent_HOST_post_cleanup_clean=clean,original_native_arms=[{k:v for k,v in arm.items() if k!='rank_records'} for arm in native.get('arms',[])],original_error=out.get('error'),native_error=native.get('error'),audit_completed=audit.get('completed'),audit_error_type=audit.get('error_type'),rank_record_bytes_SHA_verified=records_verified,recomputed_pair_rows=pair_rows,original_audit_SHA256=sha(ROOT/'independent-pair-audit.json') if (ROOT/'independent-pair-audit.json').exists() else None,checkpoint_CPU_comparison={k:audit.get(k) for k in ('same_platform_recovery_window_exact','input_equal_tail_forwards','exact_loss_tail_forwards','exact_recorded_state_updates','final_checkpoint_state_exact','final_per_rank_RNG_exact','control20_matches_historical_original20')},progress=progress,formal_performance=False,cross_platform_numerical_qualified=False,heldout_qualified=False,Model_GO=False)))