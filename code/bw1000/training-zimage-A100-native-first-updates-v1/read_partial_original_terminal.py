"""Independent private native25/checkpoint/firstupdate and ownedGPU reader."""
from pathlib import Path
import hashlib,json,math,os,signal,subprocess,sys,time
W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-zimage-A100-native-first-updates-v1')
signal.alarm(240);started=time.monotonic();artifacts={};failure=None;post=None;post_error=None;summary={}
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  while True:
   raw=stream.read(4*1024**2)
   if not raw:break
   h.update(raw)
 return h.hexdigest()
def small(name):
 p=W/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size<=8*1024**2
 raw=p.read_bytes();artifacts[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
 return json.loads(raw)
def proc(pid):
 try:
  f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
 except (FileNotFoundError,ProcessLookupError):return None
try:
 for n,pin in EXPECTED_FILES.items():assert (W/n).stat().st_size==pin['bytes'] and sha(W/n)==pin['SHA256']
 i=small('intent.json');o=small('outcome.json');r=small('GPU-result.json')
 assert i['request_id']==o['request_id']==REQUEST and i['claim']==o['claim']==CLAIM
 assert i['epoch']==o['epoch']==r['epoch'] and i['deadline']==o['deadline']==i['epoch']+3600 and r['deadline']==i['epoch']+3500
 assert o['passed'] is False and o['returncode']==1 and o['owned_group_reaped'] and o['stop_reason'] is None
 assert o['GPU_result']==r and artifacts['GPU-result.json']['SHA256']==o['GPU_result_SHA256']
 assert r['passed'] is False and r['stage']=='independent-private-checkpoint-comparison' and len(r['phases'])==2
 fm=json.loads((W/'src/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==365 and fm['framework_commit']=='b570fc14907dd7fafd53dd739a39c19795a1cc4a'
 for n,pin in fm['files'].items():
  p=W/'framework'/n;assert not p.is_symlink() and p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'] and p.stat().st_mode&0o777==pin['mode']
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 sys.path.insert(0,str(W/'framework/generation/src'))
 from gencuda import zimage_training_resume_checkpoint as resume
 from gencuda import hy_resume_checkpoint as common
 phases={p['mode']:p for p in r['phases']};assert set(phases)=={'continuous','resume20'}
 assert phases['continuous']['owner']['PID']!=phases['resume20']['owner']['PID']
 owners=[o['owner'],o['child'],* [p['owner'] for p in r['phases']]]
 for owner in owners:
  current=proc(owner['PID']);assert current is None or current['startticks']!=owner['startticks'] or current['state']=='Z'
 for n in os.listdir('/proc'):
  if n.isdigit():
   row=proc(int(n));assert not(row and row['PGID']==o['child']['PGID'] and row['state']!='Z')
 cp25={};checkpoint_rows=[];frozen_by_mode={}
 for mode,planned,expected_cps in [('continuous',list(range(1,26)),[10,20,25]),('resume20',list(range(21,26)),[25])]:
  value=small(mode+'/result.json');assert value==phases[mode]['result'] and value['passed'] and value['completed_global_steps']==planned
  assert value['actual_optimizer_updates']==len(planned) and value['frozen_parameters_unchanged'] and value['initial_tensors']==476 and value['initial_elements']==9922560
  before=small(mode+'/frozen-before.json');after=small(mode+'/frozen-after.json');assert before==after and len(before)==value['frozen_tensors'] and len(before)>=521;frozen_by_mode[mode]=before
  assert [p['global_step'] for p in value['checkpoints']]==expected_cps
  for pin in value['checkpoints']:
   p=W/mode/('checkpoint-%d'%pin['global_step'])/'state.private.pt'
   assert str(p)==pin['path'] and p.stat().st_size==pin['bytes'] and pin['bytes']<=1024**3 and sha(p)==pin['SHA256']
   payload=resume.load_checkpoint(p);m=payload['metadata'];assert m['global_step']==pin['global_step'] and m['completed_optimizer_steps']==pin['global_step'] and m['next_data_cursor']==pin['global_step']%20
   assert set(payload['state']['trainable']) and len(payload['state']['trainable'])==476 and sum(t.numel() for t in payload['state']['trainable'].values())==9922560
   assert {'trainable','optimizer','optimizer_progress','modes','buffers'}<=set(payload['state']) and set(payload['rng']['cuda'])=={0}
   assert payload['integrity_sha256']==common.tree_sha256({k:v for k,v in payload.items() if k!='integrity_sha256'})
   checkpoint_rows.append(dict(mode=mode,global_step=m['global_step'],bytes=pin['bytes'],SHA256=pin['SHA256'],state_fields=list(payload['state']),next_cursor=m['next_data_cursor']))
   if m['global_step']==25:cp25[mode]=payload
  metrics=value['step_metrics'];assert [m['global_step'] for m in metrics]==planned
  assert all(math.isfinite(m['loss']) and math.isfinite(m['grad_norm']) and m['grad_norm']>0 and m['sample_index']==(m['global_step']-1)%20 for m in metrics)
  allocator=small(mode+'/pre-exit-allocator.json');assert allocator['requires_after_process_owned_GPU_postcheck']
 a=cp25['continuous'];b=cp25['resume20'];assert a['metadata']==b['metadata']
 parameter_difference=max(float((a['state']['trainable'][n].float()-b['state']['trainable'][n].float()).abs().max()) for n in a['state']['trainable'])
 state_equal=common.tree_sha256(a['state'])==common.tree_sha256(b['state'])
 scheduler_equal=common.tree_sha256(a['scheduler'])==common.tree_sha256(b['scheduler']);rng_equal=common.tree_sha256(a['rng'])==common.tree_sha256(b['rng'])
 metricsA={x['global_step']:x for x in phases['continuous']['result']['step_metrics']};metricsB={x['global_step']:x for x in phases['resume20']['result']['step_metrics']}
 step_differences=[dict(global_step=i,continuous_loss=metricsA[i]['loss'],resumed_loss=metricsB[i]['loss'],loss_abs=abs(metricsA[i]['loss']-metricsB[i]['loss']),
  continuous_grad_norm=metricsA[i]['grad_norm'],resumed_grad_norm=metricsB[i]['grad_norm']) for i in range(21,26)]
 frozen_mismatch=[n for n in frozen_by_mode['continuous'] if frozen_by_mode['continuous'].get(n)!=frozen_by_mode['resume20'].get(n)]
 comparison=small('resume-comparison.json');assert comparison['passed'] is False and comparison['parameter_max_abs']==parameter_difference and parameter_difference>1e-6
 root=W/'continuous/first-update-private';progress=small('continuous/first-update-private/progress.json')
 names={'initial.pt','initial_optimizer.pt','input.pt','prepared.pt','target.pt','forward.pt','preclip.pt','postclip.pt','clip_norm.pt','updated.pt','optimizer.pt','optimizer_groups.pt'}
 assert progress['phase']=='NATIVE25_RETURNED' and progress['completed_global_steps']==25 and progress['first_update_complete'] and progress['planned_global_steps']==25
 assert {p['name'] for p in progress['files']}==names and len(progress['files'])==12
 total=0
 for pin in progress['files']:
  p=root/pin['name'];assert p.stat().st_size==pin['bytes'] and pin['bytes']<=256*1024**2 and sha(p)==pin['SHA256'];total+=pin['bytes']
 assert total==progress['artifact_bytes'] and total<=2*1024**3
 def load(name):return torch.load(root/name,map_location='cpu',weights_only=True)
 initial=load('initial.pt');updated=load('updated.pt');pre=load('preclip.pt');postclip=load('postclip.pt');opt=load('optimizer.pt')
 assert len(initial)==476 and set(initial)==set(updated)==set(pre)==set(postclip)==set(opt)
 assert sum(t.numel() for t in initial.values())==9922560
 changed=0
 for n,t in initial.items():
  assert t.dtype==torch.bfloat16 and all(v[n].shape==t.shape and bool(torch.isfinite(v[n]).all()) for v in (initial,updated,pre,postclip))
  assert float(opt[n]['step'])==1 and opt[n]['exp_avg'].shape==opt[n]['exp_avg_sq'].shape==t.shape
  changed+=not torch.equal(t,updated[n])
 assert changed>0
 prepared=load('prepared.pt');target=load('target.pt')['training_target'];forward=load('forward.pt')
 assert list(prepared['noise'].shape)==[1,16,32,32] and list(target.shape)==[1,16,32,32] and list(forward['noise_prediction'].shape)==[1,16,32,32]
 assert prepared['timestep_id'].numel()==prepared['timestep'].numel()==1 and bool(torch.isfinite(forward['loss']))
 summary=dict(continuous_updates=25,resumed_updates=5,first_capture_files=12,first_capture_bytes=total,trainable_tensors=476,trainable_elements=9922560,
  first_update_changed_tensors=changed,actual_noise_shape=list(prepared['noise'].shape),actual_timestep_id=int(prepared['timestep_id'].item()),
  first_loss=float(forward['loss']),checkpoint_rows=checkpoint_rows,resume_parameter_max_abs=parameter_difference,complete_resume_state_equal=state_equal,resume_acceptance_passed=False,original_resume_atol=1e-6,step_differences=step_differences,scheduler_equal=scheduler_equal,RNG_equal=rng_equal,frozen_cross_process_mismatch_count=len(frozen_mismatch),frozen_cross_process_mismatch_names=frozen_mismatch[:20])
 assert not torch.cuda.is_initialized()
except BaseException as error:
 import traceback
 failure=dict(error_type=type(error).__name__,traceback=traceback.format_exc()[-8000:])
finally:
 try:
  cards=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
  selected=[[v.strip() for v in line.split(',')] for line in cards if GPU_UUID in line]
  post=dict(cards=selected,compute_apps=[line for line in apps if GPU_UUID in line])
  assert len(selected)==1 and selected[0][1]==GPU_PCI and int(selected[0][2])<=16 and not post['compute_apps']
 except BaseException as error:post_error=type(error).__name__
print(json.dumps(dict(passed=failure is None and post_error is None,failure=failure,post_error=post_error,post=post,summary=summary,
 claim=CLAIM,request_id=REQUEST,artifacts=artifacts,original_run_passed=False,original_returncode=1,original_failure_preserved=True,resume_acceptance_passed=False,native_continuous25_and_first_capture_qualified=failure is None,source_files=len(EXPECTED_FILES),new_CPU_or_GPU_invocations=0,
 original_pretrained_weight_body_bytes_read=0,private_values_exported=False,inside_seconds=time.monotonic()-started,formal_performance=False,Model_GO=False)))
if failure is not None or post_error is not None:raise SystemExit(1)
