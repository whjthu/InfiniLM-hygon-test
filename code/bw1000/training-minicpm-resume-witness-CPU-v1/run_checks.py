from pathlib import Path
import datetime,hashlib,io,json,os,signal,sys,time,unittest
W=Path('/evidence');source=Path('/resume-framework');sys.path.insert(0,str(source));start=time.monotonic();epoch=float(os.environ['RESUME_CPU_EPOCH']);deadline=float(os.environ['RESUME_CPU_DEADLINE']);initial=deadline-time.time()
assert 0<initial<=220
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU original epoch exhausted')));signal.setitimer(signal.ITIMER_REAL,initial)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
owner=dict(PID=os.getpid(),PGID=os.getpgrp(),startticks=int(Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),node=os.uname().nodename.split('.')[0],step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'])
assert owner['node']=='f11r4n10' and os.environ['SLURM_CPUS_PER_TASK']=='4'
with (W/'owner.json').open('x') as f:json.dump(owner,f,indent=2)
result=None;stage='source'
try:
 mf=json.loads((source/'SOURCE_FILES.json').read_bytes());verified=[]
 for name,row in mf['files'].items():assert (source/name).stat().st_size==row['bytes'] and h(source/name)==row['SHA256'];verified.append(name)
 stage='unit-tests';log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromNames(['vlm.tests.test_resume_witness','vlm.tests.test_resume_cursor','vlm.tests.test_hygon_vlm_recipe']);tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 (W/'tests.txt').write_text(log.getvalue());assert tests.wasSuccessful() and tests.testsRun==25,log.getvalue()[-1200:]
 import torch,transformers,peft
 import vlm.plugins.resume_training as plugin
 holder=object();assert plugin.NativeResume(None,holder).trainer is holder and plugin.callbacks_map[plugin.KEY] is plugin.NativeResume
 stage='original-checkpoint-CPU-loader';config=json.loads((source/'CHECKPOINT_CONFIG.json').read_bytes())
 from vlm.runtime.resume_witness import pinned_checkpoint
 state=pinned_checkpoint(config['checkpoint'],config['checkpoint_files'],8);assert state['state']['global_step']==20 and len(state['RNG'])==8 and state['adapter']['tensors']==372 and state['adapter']['elements']==10822656
 stage='normal-entry-plan-and-stdlib-preflight'
 sys.path.insert(0,str(source/'tools'));import hygon_training_recipe as entry
 argv=['T05.sft','--a100-launch',str(source/'original-launch.json'),'--swift','/verified/swift_entry.py','--models','/verified/models','--dataset','/verified/coco64.jsonl','--plugins','/verified/plugins','--output','/new-resume/train','--master-port','29883','--diagnostic-plugin',str(source/'vlm/plugins/resume_training.py'),'--diagnostic-callback','native_resume_witness','--resume-from-checkpoint',config['checkpoint']]
 out=io.StringIO()
 import contextlib
 with contextlib.redirect_stdout(out):assert entry.main(argv)==0
 launch=json.loads(out.getvalue());assert launch['command'][launch['command'].index('--max_steps')+1]=='25' and 'nvidia_baseline' not in launch['command']
 import hygon_vlm_recipe as recipe
 recipe.validate_diagnostic_outputs(launch,dict(NATIVE_RESUME_CONFIG=str(source/'CHECKPOINT_CONFIG.json'),NATIVE_RESUME_CONFIG_SHA256=h(source/'CHECKPOINT_CONFIG.json')))
 assert not torch.cuda.is_initialized() and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['HIP_VISIBLE_DEVICES']==os.environ['ROCR_VISIBLE_DEVICES']=='-1'
 swift=Path('/swift-runtime/site/swift/arguments/sft_args.py');resume_source=None
 if swift.is_file():
  raw=swift.read_bytes();assert len(raw)<100000;lines=raw.decode().splitlines();selected=[]
  for i,line in enumerate(lines):
   if 'resume_from_checkpoint' in line or 'resume_only_model' in line:selected.append(dict(line=i+1,source='\n'.join(lines[max(0,i-2):min(len(lines),i+4)])))
  resume_source=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),resume_clauses=selected[:16])
 result=dict(passed=True,observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner=owner,epoch=epoch,deadline=deadline,source_files_verified=len(verified),source_manifest_SHA256=h(source/'SOURCE_FILES.json'),tests_run=25,tests_SHA256=h(W/'tests.txt'),test_log=log.getvalue(),plugin_registered_and_base_constructor_verified=True,original_checkpoint20_loaded_CPU=dict(state={k:state['state'][k] for k in ('global_step','max_steps','epoch')},adapter=state['adapter'],optimizer=state['optimizer'],scheduler=state['scheduler'],rank_RNG=state['RNG'],checkpoint_files_verified=14),normal_unified_entry_resume_plan_verified=True,pre_runtime_checkpoint_hash_check_passed=True,swift_resume_source=resume_source,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,Python=sys.version,CUDA_initialized=False,GPU_cards=0,original_model_loaded=False,raw_tensor_transfer_bytes=0,native_GPU_restore_executed=False,recovery_qualified=False,Model_GO=False,wall_seconds=time.monotonic()-start,qualification='Actual target-container CPU fixtures plus original checkpoint CPU deserialization and unified resume command planning. No actual MiniCPM native recovery, numerical, performance or effects acceptance.')
except BaseException as error:
 result=dict(passed=False,owner=owner,epoch=epoch,deadline=deadline,stage=stage,error_type=type(error).__name__,message=str(error)[-1800:],wall_seconds=time.monotonic()-start,GPU_cards=0,Model_GO=False)
finally:
 with (W/'CPU-result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if not result['passed']:raise SystemExit(1)