"""One conditional policy qualification/causal pair/full25+freshresume chain."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time
W=Path(__file__).resolve().parents[1];S=W/'src';F=W/'framework';c=json.loads((S/'CLAIM.json').read_bytes());b=json.loads((S/'BINDING.json').read_bytes())
UUID=c['GPU_UUID'];PCI=c['GPU_PCI'];epoch=float(os.environ['CONTRACT_GPU_EPOCH']);deadline=float(os.environ['CONTRACT_GPU_DEADLINE']);anchor=time.monotonic();remaining=deadline-time.time();assert 0<remaining<=3500
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original policy chain epoch expired')));signal.setitimer(signal.ITIMER_REAL,remaining)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,v):
 with (W/n).open('x') as f:json.dump(v,f,indent=2)
def proc(pid):
 f=Path('/proc/%d/stat'%pid).read_text().split(') ',1)[1].split();return dict(PID=pid,state=f[0],PGID=int(f[2]),startticks=int(f[19]))
def idle():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines();cards=[[v.strip() for v in row.split(',')] for row in rows if UUID in row]
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
 assert len(cards)==1 and cards[0][1]==PCI and int(cards[0][2])<=16 and not any(UUID in row for row in apps);return dict(cards=cards,owned_compute_apps=[])
owner=proc(os.getpid());dump('GPU-owner.json',owner);phases=[];stage='source';cpu=None;sdk=None;pair=None;full=None
def launch(label,args,env=None,limit=None):
 before=idle() if label!='CPU-policy-checks' else None
 with (W/(label+'.private.log')).open('xb') as log:
  child=subprocess.Popen(args,cwd=F,env=env or os.environ.copy(),stdout=log,stderr=subprocess.STDOUT);owned=proc(child.pid);assert owned['PGID']==owner['PGID'];dump(label+'-owner.json',owned)
  rc=child.wait(timeout=min(deadline-time.time(),remaining-(time.monotonic()-anchor),limit if limit is not None else 3500))
 row=dict(label=label,owner=owned,returncode=rc,before_GPU=before);phases.append(row);dump(label+'-returned.json',row);return row
def native(label,mode,checkpoint=None,parent=False,warm=False):
 args=[sys.executable,'-B','-u',str(F/'tools/hygon_zimage_native_record.py'),'--binding',str(S/'BINDING.json'),'--output',str(W/label),'--mode',mode,'--epoch',str(epoch),'--deadline',str(deadline)]
 if checkpoint:args+=['--checkpoint',str(checkpoint)]
 if parent:args+=['--first-resumed-update-only','--diagnostic-parent-checkpoint']
 if warm:args+=['--warmup-before-restored-update']
 row=launch(label,args);p=W/label/'result.json';value=json.loads(p.read_bytes()) if p.is_file() else None
 dump(label+'-native-result.json',dict(returncode=row['returncode'],result=value,result_SHA256=sha(p) if p.is_file() else None))
 assert row['returncode']==0 and value and value['passed'] and value['attention_backward_policy']['forced_calls']>0
 assert value['attention_backward_policy']['policy']['deterministic_backward'] and value['diagnostic_parent_checkpoint']==parent
 dump(label+'-after-GPU.json',idle());return value
try:
 fm=json.loads((S/'FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==379
 for n,pin in fm['files'].items():assert (F/n).stat().st_size==pin['bytes'] and sha(F/n)==pin['SHA256']
 stage='CPU-policy-checkpoint-qualification';env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
 row=launch('CPU-policy-checks',[sys.executable,'-B','-u',str(S/'check_policy_checkpoint_CPU.py')],env=env,limit=150)
 cpu=json.loads((W/'CPU-policy-checks.json').read_bytes());assert row['returncode']==0 and cpu['passed'] and cpu['tests']==2 and cpu['skipped']==0 and not cpu['GPU_initialized']
 stage='SDK-exact-observed-shapes';sdk_deadline=min(epoch+300,deadline)
 row=launch('SDK-policy-qualification',[sys.executable,'-B','-u',str(F/'tools/hygon_zimage_flash2_policy_probe.py'),'--binding',str(S/'BINDING.json'),'--output',str(W/'SDK-policy-qualification'),'--epoch',str(epoch),'--deadline',str(sdk_deadline)],limit=sdk_deadline-time.time())
 sdk=json.loads((W/'SDK-policy-qualification/result.json').read_bytes());assert row['returncode']==0 and sdk['passed'] and sdk['actual_forward_calls']==sdk['actual_backward_calls']==12
 dump('SDK-policy-after-GPU.json',idle())
 cp=Path(b['diagnostic_checkpoint20']['path']);assert cp.stat().st_size==b['diagnostic_checkpoint20']['bytes'] and sha(cp)==b['diagnostic_checkpoint20']['SHA256']
 stage='diagnostic-cold';cold=native(stage,'resume20',cp,parent=True)
 stage='diagnostic-warm';warm=native(stage,'resume20',cp,parent=True,warm=True)
 assert cold['completed_global_steps']==warm['completed_global_steps']==[21] and cold['actual_optimizer_updates']==1 and warm['actual_optimizer_updates']==2 and warm['diagnostic_warmup']['restored20_after_warmup']
 stage='compare-policy-causal-pair'
 import torch
 sys.path.insert(0,str(F/'generation/src'));from gencuda.checkpoint_compare import compare_trees
 assert not torch.cuda.is_initialized();groups=[]
 roots=[W/arm/'resumed-update-private' for arm in ('diagnostic-cold','diagnostic-warm')]
 manifests=[json.loads((root/'progress.json').read_bytes()) for root in roots];pins=[{v['name']:v for v in m['files']} for m in manifests]
 names=['initial.pt','initial_optimizer.pt','input.pt','prepared.pt','target.pt','forward.pt','preclip.pt','postclip.pt','updated.pt','optimizer.pt','clip_norm.pt','optimizer_groups.pt']
 for n in names:
  values=[]
  for root,manifest in zip(roots,pins):
   p=root/n;pin=manifest[n];assert p.stat().st_size==pin['bytes'] and sha(p)==pin['SHA256'];values.append(torch.load(p,map_location='cpu',weights_only=True))
  groups.append(dict(name=n,**compare_trees(*values)));del values
 states=[torch.load(W/arm/'checkpoint-21/state.private.pt',map_location='cpu',weights_only=False) for arm in ('diagnostic-cold','diagnostic-warm')]
 assert states[0]['metadata']==states[1]['metadata'] and states[0]['metadata']['contract']['identity']['config']==b['config_canonical_SHA256']
 complete=compare_trees(*states);del states
 assert sha(W/'diagnostic-cold/frozen-before.json')==sha(W/'diagnostic-warm/frozen-before.json')
 pair=dict(passed=all(v['tree_equal'] for v in groups) and complete['tree_equal'],groups=groups,complete_checkpoint21=complete,original_default_drift_not_replayed=True,actual_native_optimizer_calls=3)
 dump('policy-causal-pair.json',pair)
 if pair['passed']:
  stage='candidate-continuous25';continuous=native(stage,'continuous')
  assert continuous['completed_global_steps']==list(range(1,26)) and continuous['actual_optimizer_updates']==25
  stage='candidate-resume20';resumed=native(stage,'resume20',W/'candidate-continuous25/checkpoint-20/state.private.pt')
  assert resumed['completed_global_steps']==list(range(21,26)) and resumed['actual_optimizer_updates']==5
  stage='compare-full25-freshresume'
  values=[torch.load(W/arm/'checkpoint-25/state.private.pt',map_location='cpu',weights_only=False) for arm in ('candidate-continuous25','candidate-resume20')]
  assert values[0]['metadata']==values[1]['metadata'] and values[0]['metadata']['global_step']==25
  allstate=compare_trees(*values);del values
  ca={v['global_step']:v for v in continuous['step_metrics']};ra={v['global_step']:v for v in resumed['step_metrics']}
  loss=max(abs(ca[i]['loss']-ra[i]['loss']) for i in range(21,26))
  full=dict(passed=allstate['tree_equal'] and loss<=1e-6,complete_checkpoint25=allstate,loss_max_abs=loss,original_resume_atol=1e-6,actual_continuous_updates=25,actual_resumed_updates=5,
   same_explicit_attention_policy=True,independent_fresh_processes=True,formal_performance=False,new_platform_alignment_qualified=False)
  dump('candidate-full25-resume-comparison.json',full)
 finalpost=idle();dump('final-after-GPU.json',finalpost)
 result=dict(passed=True,owner=owner,epoch=epoch,deadline=deadline,framework_commit=fm['framework_commit'],framework_files=379,phases=phases,CPU_checks=cpu,SDK_qualification=sdk,
  pair=pair,full25_resume_comparison=full,candidate_causal_passed=pair['passed'],candidate_full25_resume_passed=bool(full and full['passed']),
  actual_native_model_loads=4 if full is not None else 2,actual_native_optimizer_calls=33 if full is not None else 3,actual_GPU_worker_processes=5 if full is not None else 3,
  GPU_cards=1,wall_seconds=time.monotonic()-anchor,old_default_native_resume_FAIL_preserved=True,formal_performance=False,new_platform_alignment_qualified=False,heldout_effect_qualified=False,Model_GO=False)
except BaseException as error:
 import traceback
 dump('failure-private.json',dict(stage=stage,error_type=type(error).__name__,traceback=traceback.format_exc()[-10000:],phases=phases))
 result=dict(passed=False,stage=stage,error_type=type(error).__name__,owner=owner,epoch=epoch,deadline=deadline,phases=phases,CPU_checks=cpu,SDK_qualification=sdk,pair=pair,full25_resume_comparison=full,
  GPU_cards=1,wall_seconds=time.monotonic()-anchor,Model_GO=False)
dump('GPU-result.json',result);print(json.dumps(dict(passed=result['passed'],stage=stage,error_type=result.get('error_type'),candidate_causal_passed=result.get('candidate_causal_passed'),candidate_full25_resume_passed=result.get('candidate_full25_resume_passed'),GPU_cards=1)))
if not result['passed']:raise SystemExit(1)
