from pathlib import Path
import datetime,hashlib,json,socket,subprocess
W=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-paired-resume-CPU-v1');label=W.name
raw=(W/'outcome.json').read_bytes();out=json.loads(raw);assert out==EXPECTED['outcome'] and out['passed'] and not out['forced_cleanup'] and out['owned_srun_reaped']
intent=json.loads((W/'intent.json').read_bytes());assert intent['claim']==EXPECTED['claim'] and intent['request_id']==EXPECTED['request'] and intent['epoch']==out['epoch']==EXPECTED['epoch'] and intent['deadline']==out['deadline']==EXPECTED['deadline']==out['epoch']+300 and intent['work_deadline']==out['epoch']+220 and intent['GPU_cards']==0
assert len(intent['actual_nodes'])==len(set(intent['actual_nodes']))==40 and intent['actual_parent_resources']['JobState']=='RUNNING'
for name,pin in EXPECTED['source_files'].items():assert hashlib.sha256((W/'source'/name).read_bytes()).hexdigest()==pin
mf=json.loads((W/'source/SOURCE_FILES.json').read_bytes());assert len(mf['files'])==29 and mf['framework_commit']=='480cee6ed85a8499a57b0890c5cbbb56f77da9ce'
for name,pin in mf['files'].items():assert (W/'source'/name).stat().st_size==pin['bytes'] and hashlib.sha256((W/'source'/name).read_bytes()).hexdigest()==pin['SHA256']
fm=json.loads((W/'source/FRAMEWORK_FILES.json').read_bytes());assert fm['framework_commit']=='480cee6ed85a8499a57b0890c5cbbb56f77da9ce' and len(fm['files'])==304
import stat
for name,pin in fm['files'].items():
 p=W/'framework'/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['SHA256'] and stat.S_IMODE(p.stat().st_mode)==pin['mode']
columns=['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList'];assert out['accounting_columns']==columns
rows=[s.split('|') for s in subprocess.check_output(['sacct','-n','-P','-j','1087656','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'],text=True,timeout=15).splitlines() if '|'+label+'|' in s];assert rows==out['accounting'] and len(rows)==1 and rows[0][0]=='1087656.149' and rows[0][2:4]==['COMPLETED','0:0'] and rows[0][5]=='f11r4n10'
assert hashlib.sha256((W/'CPU.log').read_bytes()).hexdigest()==out['log_SHA256']
p=W/'evidence/CPU-result.json';result_raw=p.read_bytes();result=json.loads(result_raw);record=out['CPU-result.json'];assert len(result_raw)==record['bytes'] and hashlib.sha256(result_raw).hexdigest()==record['SHA256'] and result==record['value']
assert result['passed'] and result['tests_run']==14 and result['source_files_verified']==304 and result['remote_source_files_verified']==29 and result['source_manifest_SHA256']==EXPECTED['source_files']['SOURCE_FILES.json'] and result['CUDA_initialized'] is False and result['GPU_cards']==0 and result['raw_tensor_transfer_bytes']==0 and not result['original_model_loaded'] and not result['native_GPU_restore_executed']
assert result['owner']['step']==rows[0][0] and result['owner']['node']==rows[0][5] and result['epoch']==out['epoch'] and result['deadline']==out['epoch']+220 and 0<result['wall_seconds']<220
traw=(W/'evidence/tests.txt').read_bytes();assert hashlib.sha256(traw).hexdigest()==result['tests_SHA256'];tests=traw.decode();assert 'Ran 14 tests' in tests and tests.rstrip().endswith('OK') and len([x for x in tests.splitlines() if x.startswith('test_') and x.endswith(' ... ok')])==14
cp=result['original_checkpoint10_loaded_CPU'];assert cp['checkpoint_files_verified']==14 and cp['adapter']['tensors']==372 and cp['adapter']['elements']==10822656 and len(cp['rank_RNG'])==8 and cp['state']==dict(global_step=10,max_steps=25,epoch=5.0)
assert result['normal_unified_entry_resume_plan_verified'] and result['pre_runtime_checkpoint_hash_check_passed'] and not result['recovery_qualified'] and not result['Model_GO']
owner=json.loads((W/'owned-srun.json').read_bytes());host_matches=socket.gethostname()==owner['host'];match=None
if host_matches:
 path=Path('/proc',str(owner['PID']),'stat');match=path.exists() and int(path.read_text().split(') ',1)[1].split()[19])==owner['startticks'];assert not match
print(json.dumps(dict(observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_original149_verified=True,original_outcome_SHA256=hashlib.sha256(raw).hexdigest(),original_CPU_result_SHA256=record['SHA256'],source_files_verified=30,full_framework_source_files_verified=304,framework_commit=fm['framework_commit'],framework_tree=fm['framework_tree'],framework_manifest_SHA256=EXPECTED['source_files']['FRAMEWORK_FILES.json'],accounting_columns=columns,accounting=rows,epoch=out['epoch'],deadline=out['deadline'],forced_cleanup=False,owned_srun_reaped=True,owned_srun_HOST_matches=host_matches,owned_srun_startticks_still_present=match,actual_parent_resources=intent['actual_parent_resources'],actual_parent_node_names=intent['actual_nodes'],result=result,qualification='Exact original CPU terminal/source/log/result accounting verified. Actual-model native checkpoint recovery, original continuous-tail identity, quality and performance remain unqualified.')))
