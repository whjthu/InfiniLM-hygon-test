"""Public queue -> four-row same-owner Native cohort. No HTTP ownership."""
import copy,json,os,queue,time,traceback
from .affinity_env import validate as validate_affinity_env
from .checkpoint_loader import bind_library,read_model_metadata
from .native_binding import binding as verify_native_chain
from .native_context import NativeContext
from .cohort_continuation import run as continue_cohort
from .cohort_admission import release
from .profile import NAME,validate
from .protocol import Run
def configure_process(config):
 validate(config);assert(config['TP'],config['PP'])==(8,1)
 assert validate_affinity_env(os.environ)['requested_width']==4 and len(os.sched_getaffinity(0))>=32
 assert os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']==config['actual_step']
 assert os.environ['INFINILM_DEEP_MOE_TAIL512_R7211']==os.environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']=='0'
 assert not any(os.environ.get(k)for k in['BASH_ENV','PYTHONSTARTUP','PYTHONINSPECT','INFINILM_DEEP_LAYER_TRACE_R4026','INFINILM_DEEP_POSITION_PROOF_R8726'])
def torch_setup():
 import torch
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0);return torch
def safe_open_dependency():
 from safetensors import safe_open
 return safe_open
def serve(context,prepared,config,run,stop_intent,report,save):
 validate(config);engine=prepared['engine'];Reason=prepared['FinishReason'];current=[]
 assert prepared['identity']['family']=='Deep'and prepared['identity']['eos_token_ids']==[1]
 try:
  while not stop_intent.is_set()and not run.stop_requested():
   run.deadline_check('fixed cohort admission')
   try:first=engine.take(timeout=.1)
   except queue.Empty:continue
   if first is None:break
   current=[first];sentinel=False;limit=min(time.monotonic()+config['cohort_assembly_seconds'],time.monotonic()+max(0,run.remaining()))
   while len(current)<4:
    if stop_intent.is_set()or run.stop_requested():break
    if time.monotonic()>=limit:raise TimeoutError('Incomplete fixed4 cohort at bounded assembly deadline')
    try:req=engine.take(timeout=min(.1,max(.001,limit-time.monotonic())))
    except queue.Empty:continue
    if req is None:sentinel=True;break
    current.append(req)
   if len(current)<4:
    if sentinel or stop_intent.is_set()or run.stop_requested():break
    raise RuntimeError('Incomplete cohort; no partial Native execution')
   with engine._admission_lock:
    assert all(getattr(q,'_admission_metadata_ready',False)for q in current)
    current.sort(key=lambda q:q.metadata['native_fixed_cohort']['row']);tickets=[copy.deepcopy(q.metadata['native_fixed_cohort'])for q in current]
    assert[q['row']for q in tickets]==list(range(4))and len({q['id']for q in tickets})==len({q['mode']for q in tickets})==1
    admissions=[copy.deepcopy(q.metadata)for q in current]
   records=[dict(row=i,case_id=q.request_id,prompt_token_ids=list(q.prompt_token_ids),max_new_tokens=q.max_new_tokens)for i,q in enumerate(current)]
   emitted=[0]*4;identifier=tickets[0]['id'];mode=tickets[0]['mode'];cohort=dict(id=identifier,mode=mode,requests=[q.request_id for q in current],admissions=admissions,completed=False)
   report.setdefault('cohorts',[]).append(cohort);save()
   def progress(phase,trial):
    for i,state in enumerate(trial['states']):
     for event in state['token_events'][emitted[i]:]:current[i].emit_token(event['token'])
     emitted[i]=len(state['token_events'])
    if phase in('failed_before_drain','terminal'):cohort.update(phase=phase,failed=trial['failed'],normal_reset_after=trial['Native_reset_after']);save()
   # One request cancel cancels this closed cohort at a returned-call boundary.
   # Finish is deferred until the synchronous all8 reset; no unsafe row reuse.
   canceled=lambda:stop_intent.is_set()or run.stop_requested()or any(q.cancel_event.is_set()for q in current)
   trial=continue_cohort(context,records,mode=mode,deadline=config['gpu_start_epoch']+config['service_seconds'],save=progress,canceled=canceled)
   assert trial['normal']and trial['Native_reset_before']and trial['Native_reset_after']
   cohort.update(completed=True,trial=trial);release(engine,identifier)
   for q,state in zip(current,trial['states']):
    reason={'EOS':Reason.EOS_TOKEN,'cap':Reason.LENGTH,'canceled':Reason.CANCELED}[state['finish']]
    q.metadata.update(fixed_cohort_id=identifier,fixed_cohort_mode=mode,real_cache_rows=4,cohort_normal_reset_after=True,cohort_row=state['row'],Native_generated_IDs=list(state['generated_ids']))
    q.finish(reason);report.setdefault('requests',[]).append(dict(request_id=q.request_id,final_metadata=copy.deepcopy(q.metadata),generated_token_ids=list(q.generated_token_ids),native_finish_reason=state['finish'],normal_reset_after=True))
   current=[];save()
 finally:
  import sys
  primary=sys.exc_info()[1]
  if primary is None:context.reset()
  with engine._lock:pending=list(engine._requests.values())
  for q in pending:
   if not q.is_finished():
    if primary is None:q.finish(q.abort_reason or Reason.CANCELED)
    else:q.fail(primary)
def owner_main(config,prepared,stop_intent):
 configure_process(config);run=Run(config);report=dict(identity=run.identity(),completed=False,Model_READY=False,normal_Native_shutdown=False,requests=[],cohorts=[],Native_SHA256=config['Native_SHA256'],TP=8,PP=1,HC_mode=0,rows=4)
 path=run.path/'owner.json'
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
 try:
  _,receipt,_=verify_native_chain();assert receipt['library_SHA256']==config['Native_SHA256']
  torch=torch_setup();safe_open=safe_open_dependency();cfg,mapping=read_model_metadata(config['model_path']);binding=bind_library(config['Native_library'],config['Native_SHA256'])
  with NativeContext(binding,receipt,config=config,cfg=cfg,mapping=mapping,torch=torch,safe_open=safe_open,report=report,deadline_check=run.deadline_check)as context:
   fixed=dict(profile=NAME,HC_mode=0,r366_fixed=0,fullpre=0,precision_policy=0,rows=4,capacity_per_row=32768,maximum_cohort=4,row_reuse=False,C1=False)
   report['fixed_profile']=fixed;save();run.write('ready-engine.json',dict(loaded=True,loaded_key_count=report['loaded_key_count'],Native_SHA256=config['Native_SHA256'],HC_mode=0,rows=4,capacity=32768,affinity=report['owner_affinity_after_weight_load'],postload_free=report['GPU_memory_after_load'],fixed_profile=fixed))
   serve(context,prepared,config,run,stop_intent,report,save)
  report.update(completed=True,Model_READY=False);save();run.write('closed-engine.json',dict(normal=report['normal_Native_shutdown'],engine_destroy_returned_zero=report['engine_worker_shutdown_completed'],observed_all8_threads_exited=report['observed_all8_threads_exited']))
 except BaseException as error:
  report.update(completed=True,passed=False,error=repr(error),traceback=traceback.format_exc());save();run.fail(error,stage=0);raise
