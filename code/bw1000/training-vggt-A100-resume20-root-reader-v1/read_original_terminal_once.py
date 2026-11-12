"""Independent CPU hash audit of the new resumed artifacts against qualified originals."""
from pathlib import Path
import hashlib,json,math,os,signal,socket,subprocess,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-A100-native-resume20-v1');A=W/'root-reader-v1';A.mkdir(mode=0o700,exist_ok=False);epoch=time.time();mono=time.monotonic();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):
   if time.monotonic()-mono>=170 or time.time()>=epoch+170:raise TimeoutError('one reader epoch exhausted')
   h.update(b)
 return h.hexdigest()
def dump(n,v):
 with (A/n).open('x') as f:json.dump(v,f,indent=2)
def live(pid,row):
 p=Path('/proc')/str(pid)/'stat';return p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==row['startticks']
def expired(*_):raise TimeoutError('one reader hard deadline exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,170)
dump('intent.json',dict(reader_claim=READER_CLAIM,original_claim=ORIGINAL_CLAIM,original_request=ORIGINAL_REQUEST,epoch=epoch,deadline=epoch+180,PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),GPU_cards=0,benchmark_replayed=False,raw_export_bytes=0))
phase='original-terminal'
try:
 intent=json.loads((W/'intent.json').read_bytes());o=json.loads((W/'outcome.json').read_bytes());assert intent['claim']==o['claim']==ORIGINAL_CLAIM and intent['request_id']==o['request_id']==ORIGINAL_REQUEST and sha(W/'outcome.json')==ORIGINAL_OUTCOME_SHA256
 assert o['native_returncode']==0 and o['keeper_passed'] and o['cleanup']==dict(owned_processes_remaining=0,termination_needed=False) and not o['timed_out'] and o['controller_error'] is None
 assert o['epoch']==intent['epoch'] and o['native_deadline']==o['epoch']+820 and o['deadline']==o['epoch']+900
 assert socket.gethostname()==intent['owner']['host'] and not live(intent['owner']['PID'],intent['owner']) and all(not live(pid,row) for pid,row in o['owners'].items())
 prior=json.loads((W/'src/PRIOR_REFERENCE.json').read_bytes());R=Path(prior['root']);continuous=json.loads((R/'train/result.json').read_bytes());previous=json.loads((R/'outcome.json').read_bytes());qualified=json.loads((R/'root-reader-v1/outcome.json').read_bytes());assert sha(R/'train/result.json')==prior['result_SHA256'] and sha(R/'outcome.json')==prior['outcome_SHA256'] and sha(R/'root-reader-v1/outcome.json')==prior['reader_outcome_SHA256']
 assert qualified['passed'] and qualified['capture_files_all_SHA256_verified'] and qualified['native_global_steps']==25 and qualified['checkpoints'][-1]['global_step']==25 and qualified['checkpoints'][-1]['integrity_SHA256']==prior['checkpoint25_integrity']
 assert previous['native_returncode']==0 and previous['keeper_passed'] and not set(o['owners'])&set(previous['owners']) and all(not live(pid,row) for pid,row in previous['owners'].items())
 raw=(W/'train/result.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==o['result_SHA256'];resumed=json.loads(raw);m=resumed['metrics'];cm=continuous['metrics'];assert m['completed_global_step_before_resume']==20 and m['executed_optimizer_steps']==m['measurement_steps']==5 and m['global_optimizer_step']==25
 assert m['resume_source_sha256']==prior['checkpoint20']['SHA256'] and m['checkpoint_identity']==cm['checkpoint_identity'] and m['final_state_sha256']==cm['final_state_sha256']==prior['checkpoint25_integrity']
 assert resumed['checks']['resume_boundary'] and resumed['checks']['frozen_aggregator_unchanged'] and not m['frozen_aggregator_feature_cache']
 original_config=json.loads((W/'src/original-config.json').read_bytes());config=json.loads((W/'train/config.json').read_bytes());assert config.pop('resume_from_checkpoint')==prior['checkpoint20']['path'] and config==original_config
 phases=[]
 for root,result in [(R,continuous),(W,resumed)]:
  assert len(result['logs'])==1;item=result['logs'][0];p=Path(item['path']);assert p==root/'train/steps.ndjson' and sha(p)==item['sha256'];phases.append([json.loads(line) for line in p.read_bytes().splitlines()])
 a=[v for v in phases[0] if v['global_optimizer_step']>=21];b=phases[1];assert len(a)==len(b)==5 and [v['global_optimizer_step'] for v in b]==list(range(21,26))
 pairs=[]
 for x,y in zip(a,b):
  assert x['global_optimizer_step']==y['global_optimizer_step'] and x['batch_cursor']==y['batch_cursor'] and x['loss']==y['loss'] and math.isfinite(y['loss'])
  pairs.append(dict(step=y['global_optimizer_step'],batch_cursor=y['batch_cursor'],continuous_loss=x['loss'],resumed_loss=y['loss'],absolute_loss_error=0.0))
 phase='new-artifact-hashes';original_artifacts={Path(v['path']).name:v for v in qualified['artifacts']};artifacts=[]
 assert {Path(v['path']).name for v in resumed['artifacts']}=={'checkpoint-25.pt','camera_head.pt'}
 for item in resumed['artifacts']:
  p=Path(item['path']);assert p.parent==W/'train' and p.is_file() and not p.is_symlink();pin=original_artifacts[p.name];assert p.stat().st_size==pin['bytes'] and sha(p)==item['sha256']==pin['SHA256'];artifacts.append(dict(path=str(p),bytes=p.stat().st_size,SHA256=item['sha256'],qualified_continuous_path=pin['path'],complete_file_bytes_identical=True))
 phase='independent-post';uuid=o['post']['card'][1];apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=12).strip().splitlines();assert not [x for x in apps if x.split(',')[0].strip()==uuid]
 value=dict(passed=True,state='TERMINAL_ORIGINAL_A100_VGGT_FRESH_PROCESS20_TO25_EXACT_RECOVERY_PASS',reader_claim=READER_CLAIM,original_claim=ORIGINAL_CLAIM,original_request=ORIGINAL_REQUEST,original_outcome_SHA256=ORIGINAL_OUTCOME_SHA256,original_gpu_outcome=o,original_owner_alive=False,continuous_owner_alive=False,fresh_process=True,original_checkpoint20=prior['checkpoint20'],same_native_identity=True,original_global_step_before_resume=20,executed_actual_updates=5,final_global_step=25,exact_loss_pairs=pairs,all5_losses_exact=True,all5_batch_cursors_exact=True,complete_final_checkpoint_byte_identical=True,complete_camera_head_byte_identical=True,complete_parameter_optimizer_RNG_cursor_equality_basis='Fresh actual SHA of the complete new checkpoint25 matches the original checkpoint25 that was independently deserialized, finite-checked, schema/state/RNG/optimizer69-slot validated and integrity-checked. Original full artifact was not re-read; the new entire2594191251 bytes were hashed.',checkpoint25_integrity_SHA256=prior['checkpoint25_integrity'],artifacts=artifacts,original_trainable_tensors=69,original_trainable_parameters=216174610,optimizer_slots=69,scheduler='none in original VGGT recipe',gradient_accumulation='one complete update per original two-frame batch',frozen_aggregator_unchanged=True,online_aggregator=True,independent_selected_GPU_compute_apps=[],GPU_cards_used_for_reader=0,GPU_context_created=False,torch_imported=False,raw_export_bytes=0,full25_replayed=False,first_update_recaptured=False,formal_performance_qualified=False,cross_platform_alignment_qualified=False,heldout_effects_qualified=False)
except BaseException as error:value=dict(passed=False,phase=phase,error_type=type(error).__name__,message=str(error)[-500:],reader_claim=READER_CLAIM,original_claim=ORIGINAL_CLAIM,original_request=ORIGINAL_REQUEST,GPU_cards_used_for_reader=0,raw_export_bytes=0,benchmark_replayed=False)
value.update(reader_epoch=epoch,reader_deadline=epoch+180,reader_wall_seconds=time.monotonic()-mono);dump('outcome.json',value);print(json.dumps(value))
if not value['passed']:raise SystemExit(1)
