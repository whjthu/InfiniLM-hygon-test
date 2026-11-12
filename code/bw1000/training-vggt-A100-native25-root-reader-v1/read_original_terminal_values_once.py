"""CPU-only audit of exact original native VGGT values; no tensor export."""
from pathlib import Path
import hashlib,json,math,os,resource,signal,socket,subprocess,sys,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-A100-native25-first-update-v1');A=W/'root-reader-v1';epoch=time.time();mono=time.monotonic();A.mkdir(mode=0o700,exist_ok=False)
def dump(n,v):
 with (A/n).open('x') as f:json.dump(v,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def check():
 if time.time()>=epoch+350 or time.monotonic()-mono>=350:raise TimeoutError('one original reader epoch exhausted')
def expired(*_):raise TimeoutError('reader hard deadline exhausted')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,350)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);os.environ.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
dump('intent.json',dict(reader_claim=READER_CLAIM,original_claim=ORIGINAL_CLAIM,original_request=ORIGINAL_REQUEST,epoch=epoch,deadline=epoch+360,GPU_cards=0,raw_export_bytes=0,replay_allowed=False,PID=os.getpid(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])))
phase='original-terminal';result=None
try:
 original=json.loads((W/'intent.json').read_bytes());o=json.loads((W/'outcome.json').read_bytes());assert original['claim']==o['claim']==ORIGINAL_CLAIM and original['request_id']==o['request_id']==ORIGINAL_REQUEST
 assert sha(W/'outcome.json')==ORIGINAL_OUTCOME_SHA256 and o['native_returncode']==0 and o['keeper_passed'] and o['cleanup']==dict(owned_processes_remaining=0,termination_needed=False) and not o['timed_out']
 assert o['epoch']==original['epoch'] and o['native_deadline']==o['epoch']+820 and o['deadline']==o['epoch']+900
 def live(pid,identity):
  p=Path('/proc')/str(pid)/'stat';return p.exists() and int(p.read_text().split(') ',1)[1].split()[19])==identity['startticks']
 assert all(not live(pid,v) for pid,v in o['owners'].items());assert socket.gethostname()==original['owner']['host'] and not live(original['owner']['PID'],original['owner'])
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=12).strip().splitlines();uuid=o['post']['card'][1];assert not [x for x in apps if x.split(',')[0].strip()==uuid]
 native=json.loads((W/'train/result.json').read_bytes());assert sha(W/'train/result.json')==o['result_SHA256'];m=native['metrics'];assert m['global_optimizer_step']==m['executed_optimizer_steps']==25 and m['measurement_steps']==20 and m['frozen_aggregator_feature_cache'] is False and native['checks']['frozen_aggregator_unchanged'] and native['checks']['camera_head_weights_changed'] and native['checks']['checkpoint_resume'] is None
 capture=json.loads((W/'capture/progress.json').read_bytes());assert sha(W/'capture/progress.json')==o['capture_progress_SHA256'] and capture['phase']=='NATIVE25_RETURNED' and capture['completed_global_steps']==25 and capture['first_update_complete'] and not capture['failed']
 binding=json.loads((W/'src/NATIVE_BINDING.json').read_bytes());manifest=binding['trainable_manifest'];assert capture['trainable_manifest']==manifest and len(manifest)==69 and sum(v['numel'] for v in manifest)==216174610 and m['checkpoint_identity']==binding['native_identity']
 expected={g+'/%04d.pt'%i for g in ['initial','preclip','postclip','updated','optimizer'] for i in range(69)}|{'initial_optimizer.pt','input.pt','target.pt','forward.pt','clip_norm.pt','optimizer_groups.pt'};pins={v['name']:v for v in capture['files']};assert len(pins)==len(capture['files'])==351 and set(pins)==expected
 ledger=[json.loads(line) for line in (W/'capture/files.ndjson').read_bytes().splitlines()];assert ledger==capture['files'] and sum(v['bytes'] for v in pins.values())==capture['artifact_bytes']<=12*1024**3
 phase='CPU-values';import torch
 assert not torch.cuda.is_initialized()
 def forbid(*_a,**_kw):raise RuntimeError('GPU initialization forbidden in independent VGGT reader')
 torch.cuda._lazy_init=forbid;torch.set_num_threads(4)
 F=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-vggt-native-observer-CPU-v1/framework');sys.path.insert(0,str(F/'generation/src'));from gencuda import hy_resume_checkpoint as common
 def finite(value):
  if isinstance(value,torch.Tensor):assert value.device.type=='cpu' and not value.requires_grad and (not value.is_floating_point() or bool(torch.isfinite(value).all()))
  elif isinstance(value,dict):
   for v in value.values():finite(v)
  elif isinstance(value,(list,tuple)):
   for v in value:finite(v)
 def load(name):
  check();pin=pins[name];p=W/'capture'/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==pin['bytes']<=256*1024**2 and sha(p)==pin['SHA256'];v=torch.load(p,map_location='cpu',weights_only=True);finite(v);return v
 initial_opt=load('initial_optimizer.pt');assert initial_opt['state']=={} and len(initial_opt['param_groups'])==1 and initial_opt['param_groups'][0]['params']==list(range(69))
 groups=load('optimizer_groups.pt');assert groups==initial_opt['param_groups'] and groups[0]['lr']==2e-5 and groups[0]['weight_decay']==.01
 input=load('input.pt');assert common.tree_sha256(input['batch'])==input['input_SHA256'];assert input['batch']['images'].shape==(1,2,3,224,224) and set(input['RNG'])=={'python','numpy','torch_CPU','device'} and set(input['RNG']['device'])=={0}
 target=load('target.pt');forward=load('forward.pt');assert target['pose_encoding'].shape==(1,2,9) and bool(target['valid_frames'].all()) and len(forward['pose_enc_list'])==4 and all(v.shape==target['pose_encoding'].shape for v in forward['pose_enc_list']);first_loss=float(forward['loss']);clip_norm=float(load('clip_norm.pt'));assert math.isfinite(first_loss) and math.isfinite(clip_norm) and clip_norm>0
 pre_ss=post_ss=delta_ss=0.;changed=0;moments=0;delta_by_parameter=[]
 for index,row in enumerate(manifest):
  values={}
  for stage in ['initial','preclip','postclip','updated']:
   v=load(stage+'/%04d.pt'%index);assert set(v)=={'name','value'} and v['name']==row['name'];t=v['value'];assert list(t.shape)==row['shape'] and str(t.dtype)==row['dtype']=='torch.float32';values[stage]=t
  a=values['preclip'].double();pre_ss+=float(a.square().sum());del a
  a=values['postclip'].double();post_ss+=float(a.square().sum());del a
  delta=values['updated'].double()-values['initial'].double();squared=float(delta.square().sum());delta_ss+=squared;changed+=bool(squared);delta_by_parameter.append(dict(name=row['name'],squared_update_norm=squared));del delta
  slots=load('optimizer/%04d.pt'%index);assert slots['name']==row['name'] and set(slots['state'])=={'step','exp_avg','exp_avg_sq'} and float(slots['state']['step'])==1
  for key in ['exp_avg','exp_avg_sq']:assert list(slots['state'][key].shape)==row['shape'] and str(slots['state'][key].dtype)=='torch.float32'
  assert bool((slots['state']['exp_avg_sq']>=0).all());moments+=1;del slots,values
 assert moments==69 and changed>0 and pre_ss>0 and post_ss>0 and delta_ss>0
 assert math.isclose(math.sqrt(pre_ss),clip_norm,rel_tol=5e-5,abs_tol=1e-7) and math.sqrt(post_ss)<=1.00005
 phase='checkpoints';artifacts=[];checkpoint_metadata=[]
 for item in native['artifacts']:
  check();p=Path(item['path']);assert p.parent==W/'train' and p.name in {'checkpoint-10.pt','checkpoint-20.pt','checkpoint-25.pt','camera_head.pt'} and p.is_file() and not p.is_symlink() and sha(p)==item['sha256'];artifacts.append(dict(path=str(p),bytes=p.stat().st_size,SHA256=item['sha256']))
  if p.name.startswith('checkpoint-'):
   step=int(p.stem.split('-')[-1]);v=torch.load(p,map_location='cpu',weights_only=False,mmap=True);assert set(v)=={'metadata','state','rng','integrity_sha256'};meta=v['metadata'];assert meta['global_optimizer_step']==step and meta['contract']['identity']==binding['native_identity'] and meta['contract']['device_ids']==[0] and meta['contract']['seed']==20260911
   assert common.tree_sha256({k:x for k,x in v.items() if k!='integrity_sha256'})==v['integrity_sha256'];state=v['state'];assert set(state)=={'camera_head','optimizer','optimizer_placements'} and list(state['camera_head'])==[r['name'] for r in manifest]
   finite(state);assert len(state['optimizer']['state'])==69 and all(float(x['step'])==step for x in state['optimizer']['state'].values());assert set(v['rng'])=={'python','numpy','torch_cpu','cuda'} and common.validate_rng(v['rng'],[0],native=False)=='cpu-validated'
   checkpoint_metadata.append(dict(global_step=step,metadata=meta,RNG_fields=list(v['rng']),state_fields=list(state),optimizer_slots=69,integrity_SHA256=v['integrity_sha256']));del v,state
 assert sorted(x['global_step'] for x in checkpoint_metadata)==[10,20,25]
 log=Path(native['logs'][0]['path']);assert log==W/'train/steps.ndjson' and sha(log)==native['logs'][0]['sha256'];steps=[json.loads(line) for line in log.read_bytes().splitlines()];assert len(steps)==25 and [v['global_optimizer_step'] for v in steps]==list(range(1,26)) and math.isclose(steps[0]['loss'],first_loss,rel_tol=1e-7)
 assert all(math.isfinite(v['loss']) and math.isfinite(v['grad_norm']) for v in steps) and not torch.cuda.is_initialized()
 result=dict(passed=True,state='TERMINAL_ORIGINAL_A100_VGGT_NATIVE25_AND_FIRST_UPDATE_INDEPENDENT_PASS',reader_claim=READER_CLAIM,original_claim=ORIGINAL_CLAIM,original_request=ORIGINAL_REQUEST,original_outcome_SHA256=ORIGINAL_OUTCOME_SHA256,original_keeper=o,original_owner_alive=False,independent_selected_GPU_compute_apps=[],native_result_SHA256=o['result_SHA256'],binding_SHA256=sha(W/'src/NATIVE_BINDING.json'),framework_commit='5d12acb6fc4693243acca44cd27ede1b24567889',framework_files=321,trainable_tensors=69,trainable_parameters=216174610,capture_files=351,capture_bytes=capture['artifact_bytes'],capture_files_all_SHA256_verified=True,capture_all_tensors_finite=True,initial_optimizer_empty=True,first_optimizer_step=1,optimizer_moment_parameters=moments,changed_parameter_tensors=changed,initial_value_files={v['name']:v for v in capture['files'] if v['name'].startswith('initial/')},first_loss=first_loss,global_preclip_gradient_norm=math.sqrt(pre_ss),global_postclip_gradient_norm=math.sqrt(post_ss),recorded_clip_norm=clip_norm,first_update_norm=math.sqrt(delta_ss),update_by_parameter=delta_by_parameter,checkpoints=checkpoint_metadata,artifacts=artifacts,native_global_steps=25,measured_steps=20,frozen_aggregator_unchanged=True,online_aggregator=True,peak_GPU_reserved_bytes=m['cuda_memory']['peak_reserved_bytes'],diagnostic_seconds=capture['diagnostic_seconds'],GPU_cards_used_for_reader=0,GPU_initialized=False,raw_export_bytes=0,benchmark_replayed=False,fresh_process_recovery_qualified=False,cross_platform_numeric_alignment_qualified=False,formal_performance_qualified=False,heldout_effects_qualified=False)
except BaseException as error:result=dict(passed=False,phase=phase,error_type=type(error).__name__,message=str(error)[-500:],reader_claim=READER_CLAIM,original_claim=ORIGINAL_CLAIM,original_request=ORIGINAL_REQUEST,GPU_cards_used_for_reader=0,benchmark_replayed=False,raw_export_bytes=0)
result.update(reader_epoch=epoch,reader_deadline=epoch+360,reader_wall_seconds=time.monotonic()-mono,reader_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump('outcome.json',result);print(json.dumps(result))
if not result['passed']:raise SystemExit(1)
