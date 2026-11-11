from pathlib import Path
import hashlib,json,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native25-HIP-diagnostic-v3');label=W.name
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw=(W/'outcome.json').read_bytes();out=json.loads(raw);assert sha(W/'outcome.json')==EXPECTED['files']['outcome.json']['SHA256'] and out['keeper_passed'] is False
intent=json.loads((W/'intent.json').read_bytes());assert intent['claim_commit']=='9c23f37fc25d7fd6e7c20c723a22a37b8dd46434' and out['epoch']==intent['epoch'] and out['deadline']==intent['deadline']==out['epoch']+7800
mf=json.loads((W/'src/SOURCE_FILES.json').read_bytes());assert sha(W/'src/SOURCE_FILES.json')==intent['source_manifest_SHA256'] and len(mf['files'])==13
for name,pin in mf['files'].items():assert (W/'src'/name).stat().st_size==pin['bytes'] and sha(W/'src'/name)==pin['SHA256']==intent['files'][name]
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if any('|'+label+s+'|' in x for s in ('','-pre','-post'))]
assert rows==EXPECTED['accounting'] and [x[0] for x in rows]==['1087656.142','1087656.143','1087656.144'] and [x[2:4] for x in rows]==[['COMPLETED','0:0'],['FAILED','1:0'],['COMPLETED','0:0']]
for run in out['runs']:assert sha(W/(run['name']+'.log'))==run['log_SHA256']
a=out['assigned.json']['value'];n=out['native-outcome.json']['value'];before=out['before'];after=out['after']
assert len(set(a['PCI_by_index']))==8 and sorted(a['PCI_by_index'])==before['pci']==after['pci'] and a['step']=='1087656.143'
assert before['full_fd_inventory_performed'] is after['full_fd_inventory_performed'] is True and before['kfd_holder_count']==after['kfd_holder_count']==0 and before['boot_id']==after['boot_id'] and after['owned_processes_remaining']==[] and after['master_port_available']
assert n['native_returncode']==0 and not n['timed_out'] and n['error'] is None and n['cleanup']==dict(owned_processes_remaining=0,termination_needed=False)
assert n['rank_record_counts']==dict(baseline=8,capture=8,native25=8,masks=0) and not n['normal_native_exit']
assert sha(W/'native.log')==n['native_log_SHA256']
config=json.loads((W/'src/MASK_CONFIG.json').read_bytes());maskroot=Path(config['output']);assert maskroot==W.parent/'training-minicpm-native25-HIP-diagnostic-v2/masks' and maskroot!=W/'masks'
owners=n['owners'];ranks=[];files=[]
for rank in range(8):
 root=W/'capture'/('rank-%02d'%rank);receipt=json.loads((root/'receipt.json').read_bytes());assert receipt['rank']==rank and receipt['world_size']==8 and receipt['completed_global_step']==2 and receipt['optimizer_updates']==2 and receipt['micro_forwards']==8 and receipt['clip_calls']==2 and len(receipt['files'])==25
 progress=json.loads((root/'progress.json').read_bytes());assert not progress['failed'] and progress['completed_updates']==2
 complete=json.loads((root/'native25-complete.json').read_bytes());assert complete==dict(rank=rank,global_step=25)
 owner=json.loads((root/'owner.json').read_bytes());assert owner['rank']==owner['local_rank']==rank and owner['epoch']==intent['epoch'] and owner['deadline']==intent['epoch']+7200
 for row in receipt['files']:
  assert Path(row['name']).name==row['name'] and 0<row['bytes']<2*1024**3 and (root/row['name']).stat().st_size==row['bytes'];files.append(dict(rank=rank,**row))
 mroot=maskroot/('rank-%02d'%rank);mask=json.loads((mroot/'complete.json').read_bytes());reference=mroot.parent.parent.parent/'training-minicpm-native-mask-assets-v1/assets/masks'/('rank-%02d'%rank)/'complete.json'
 reference=Path(config['reference'])/('rank-%02d'%rank)/'complete.json';assert sha(reference)==config['reference_manifest_SHA256_by_rank'][str(rank)];ref=json.loads(reference.read_bytes())
 assert mask['mode']=='replay' and mask['contract']==ref['contract'] and len(mask['files'])==8 and mask['native_dropout_calls']==1488
 for actual,original in zip(mask['files'],ref['files']):
  assert actual['reference_SHA256']==original['SHA256'] and all(actual[k]==original[k] for k in ('global_step','micro_ordinal','input_SHA256','native_calls','mask_identity_SHA256'))
 initial=json.loads((mroot/'initial-parameters.json').read_bytes());assert initial['named_parameters_SHA256']=='9ac0f645b35f75f38aa29770c26f9298e765a4ee4190c86c9481a34bea6bac16' and initial['parameter_tensors']==372 and initial['parameter_numel']==10822656 and initial['global_step']==initial['optimizer_state_entries']==0
 baseline=W/'train/nvidia-baseline'/('rank-%02d.json'%rank);base=json.loads(baseline.read_bytes())
 registry=owners.get(str(owner['PID']));ranks.append(dict(rank=rank,native_global_step=25,capture_receipt_SHA256=sha(root/'receipt.json'),native25_SHA256=sha(root/'native25-complete.json'),owner=owner,owner_in_registry=bool(registry and registry['startticks']==owner['startticks']),baseline_SHA256=sha(baseline),baseline_top_keys=list(base),mask_receipt_SHA256=sha(mroot/'complete.json'),mask_packets=8,native_dropout_calls=1488,common_input_and_mask_identities_verified=True,initial_state_verified=initial))
records=[json.loads(line) for line in (W/'entry-receipt.jsonl').read_bytes().splitlines()]
print(json.dumps(dict(independent_original_terminal_metadata_verified=True,source_files_verified=13,accounting_columns=out['runs'][0]['accounting_columns'],accounting=rows,outcome_SHA256=sha(W/'outcome.json'),native_returncode=0,producer_keeper_passed=False,producer_GPU_scope_FAILED_preserved=True,post_HOST_PCI_KFD_FD_PORT_and_ownedPID_clean=True,native25_marker_ranks=8,baseline_record_ranks=8,rank_metadata=ranks,capture_files=files,capture_declared_bytes=sum(x['bytes'] for x in files),raw_capture_file_sizes_verified=len(files),raw_capture_SHA_verification_pending=True,common_input_mask_packets_verified=64,actual_configured_mask_output=str(maskroot),mask_scope_path_mismatch=True,entry_stages=[x['stage'] for x in records],Model_GO=False,qualification='Actual terminal/cleanup and full25 marker metadata plus common input/mask/initial audit. Raw200 tensor SHA/content, checkpoint recovery, numerical thresholds, formal performance and heldout remain unqualified.')))
