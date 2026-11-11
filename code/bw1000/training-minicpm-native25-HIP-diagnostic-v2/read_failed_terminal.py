from pathlib import Path
import hashlib,json,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-native25-HIP-diagnostic-v2');label=W.name
raw=(W/'outcome.json').read_bytes();out=json.loads(raw);assert hashlib.sha256(raw).hexdigest()==EXPECTED['files']['outcome.json']['SHA256'] and out['keeper_passed'] is False
intent=json.loads((W/'intent.json').read_bytes());assert intent['claim_commit']=='92a441ce81339eb1b25ca8aea780e85ea8f18ce2' and out['epoch']==intent['epoch'] and out['deadline']==intent['deadline']==out['epoch']+7800
mfraw=(W/'src/SOURCE_FILES.json').read_bytes();assert hashlib.sha256(mfraw).hexdigest()==intent['source_manifest_SHA256']=='630e2bc02746ecbcd0989de8d0e44f0affefa7b6a43328f8eee365f1500d8bc0';mf=json.loads(mfraw)
for name,pin in mf['files'].items():
 data=(W/'src'/name).read_bytes();assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['SHA256']==intent['files'][name]
assert len(mf['files'])==13
rows=[x.split('|') for x in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if any('|'+label+s+'|' in x for s in ('','-pre','-post'))]
assert rows==EXPECTED['accounting'] and [x[0] for x in rows]==['1087656.137','1087656.138','1087656.139'] and [x[2:4] for x in rows]==[['COMPLETED','0:0'],['FAILED','1:0'],['COMPLETED','0:0']]
for run in out['runs']:
 data=(W/(run['name']+'.log')).read_bytes();assert hashlib.sha256(data).hexdigest()==run['log_SHA256']
a=out['assigned.json']['value'];n=out['native-outcome.json']['value'];before=out['before'];after=out['after'];assert len(set(a['PCI_by_index']))==8 and sorted(a['PCI_by_index'])==before['pci']==after['pci'] and a['step']=='1087656.138'
assert a['leased_snapshot']['kfd_holder_count']==0 and a['leased_snapshot']['full_fd_inventory_performed'] is False and before['full_fd_inventory_performed'] is after['full_fd_inventory_performed'] is True
assert before['kfd_holder_count']==after['kfd_holder_count']==0 and before['boot_id']==after['boot_id'] and after['owned_processes_remaining']==[] and after['master_port_available']
assert n['native_returncode']==2 and not n['timed_out'] and n['error'] is None and n['cleanup']==dict(owned_processes_remaining=0,termination_needed=False)
assert set(n['rank_record_counts'].values())=={0}
log=(W/'native.log').read_bytes();assert hashlib.sha256(log).hexdigest()==n['native_log_SHA256'] and log.strip()==b'missing Swift entry script or model directory'
records=[json.loads(line) for line in (W/'entry-receipt.jsonl').read_bytes().splitlines()];assert [x['stage'] for x in records]==['planned','failed'] and records[-1]['error']=='missing Swift entry script or model directory'
print(json.dumps(dict(independent_failed_terminal_verified=True,source_files_verified=13,source_manifest_SHA256=intent['source_manifest_SHA256'],accounting_columns=out['runs'][0]['accounting_columns'],accounting=rows,outcome_SHA256=hashlib.sha256(raw).hexdigest(),actual_assigned=a,native_returncode=2,rank_record_counts=n['rank_record_counts'],native_process_cleanup=n['cleanup'],post_HOST_PCI_KFD_FD_PORT_and_ownedPID_clean=True,original_keeper_passed=False,entry_receipt_SHA256=hashlib.sha256((W/'entry-receipt.jsonl').read_bytes()).hexdigest(),entry_stages=['planned','failed'],projectHIP_identity_calls_completed=9,model_command_not_spawned=True,completed_native_updates=0,Model_GO=False)))
