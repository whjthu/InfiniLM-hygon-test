"""Concrete afc6 create/raw-load/Native FIFO/decode/destroy, owned by one thread."""
import copy,json,os,queue,threading,time,traceback,sys
from pathlib import Path
import numpy as np
from .affinity_env import validate as validate_affinity_env
from .checkpoint_loader import bind_library,read_model_metadata
from .native_binding import binding as verify_native_chain
from .native_context import NativeContext
from .native_decode import generate
from .protocol import Run
from .request_policy import RequestPolicy

def configure_process(config):
 assert (config['TP'],config['PP'],config['rows'],config['context'],config['HC_mode'])==(8,1,1,32768,0)
 assert validate_affinity_env(os.environ)['requested_width']==4 and len(os.sched_getaffinity(0))>=32
 assert os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']==config['actual_step']
 assert os.environ['INFINILM_DEEP_MOE_TAIL512_R7211']=='0'and os.environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']=='0'
 assert not any(os.environ.get(k)for k in ['BASH_ENV','PYTHONSTARTUP','PYTHONINSPECT','INFINILM_DEEP_LAYER_TRACE_R4026','INFINILM_DEEP_POSITION_PROOF_R8726'])

def torch_setup():
 import torch
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 return torch

def safe_open_dependency():
 from safetensors import safe_open
 return safe_open

def serve(context,prepared,config,run,stop_intent,report,save,*,policy=None):
 engine=prepared['engine'];Reason=prepared['FinishReason'];absolute=config['gpu_start_epoch']+config['service_seconds'];ctx=context.ctx;binding=context.binding
 policy=RequestPolicy(context,config)if policy is None else policy
 assert prepared['identity']['family']=='Deep'and prepared['identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'and prepared['identity']['eos_token_ids']==[1]
 try:
  while not stop_intent.is_set()and not run.stop_requested():
   run.deadline_check('request admission')
   try:req=engine.take(timeout=.1)
   except queue.Empty:continue
   if req is None:break
   if stop_intent.is_set():break
   with engine._admission_lock:
    if not getattr(req,'_admission_metadata_ready',False):raise RuntimeError('Incomplete prepared admission must never enter Native')
    admission=copy.deepcopy(req.metadata)
   context.floor();record=dict(request_id=req.request_id,admission=admission,prompt_tokens=len(req.prompt_token_ids),max_new_tokens=req.max_new_tokens,started_monotonic=time.monotonic(),normal_reset_before=False,normal_reset_after=False,backend_token_monotonic=[],Native_forward_host_summary=dict(prefill_calls=0,prefill_s=0.,decode_calls=0,decode_s=0.),actual_forward_attempts=0,actual_forward_completed=0)
   report['requests'].append(record);failure=None;reason=Reason.LENGTH;actual_emit=req.emit_token
   def emit(token):
    tick=time.monotonic();actual_emit(token);record['backend_token_monotonic'].append(tick)
   req.emit_token=emit
   def finite(logits):
    values=(logits[0].astype(np.uint32)<<16).view(np.float32)
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite real BF16 Native logits')
   def forward(*args):
    kind='decode'if record['backend_token_monotonic']else'prefill';tick=time.monotonic();record['actual_forward_attempts']+=1
    try:
     result=policy.forward(*args);binding.check(result[0]);record['actual_forward_completed']+=1;return result
    finally:record['Native_forward_host_summary'][kind+'_calls']+=1;record['Native_forward_host_summary'][kind+'_s']+=time.monotonic()-tick
   try:
    with policy.request(req,record):
     reason,trace=generate(req,forward,binding.check,finite,Reason,context=32768,chunk_size=1024,eos=1,vocab=context.cfg['vocab_size'],absolute_deadline=absolute)
     record['Native_continuation']=trace
   except BaseException as error:failure=error;record['error']=repr(error)
   finally:
    try:
     if policy.poisoned:
      record['r454_no_retry_of_failed_or_poisoned_reset']=True
     elif policy.normal_reset_after:record['normal_reset_after']=True
     else:context.reset();record['normal_reset_after']=True
    except BaseException as cleanup:
     record['reset_error']=repr(cleanup)
     if failure is None:failure=cleanup
     elif hasattr(failure,'add_note'):failure.add_note('Deep reset: '+repr(cleanup))
    req.emit_token=actual_emit
    try:
     if not policy.poisoned:record['GPU_free_after_request']=context.floor()
    except BaseException as cleanup:
     if failure is None:failure=cleanup
     elif hasattr(failure,'add_note'):failure.add_note('Deep memory floor: '+repr(cleanup))
    record.update(ended_monotonic=time.monotonic(),native_finish_reason=getattr(reason,'value',str(reason)),actual_generated_tokens=len(req.generated_token_ids),generated_token_ids=list(req.generated_token_ids))
    try:save()
    except BaseException as persistence:
     if failure is None:failure=persistence
     elif hasattr(failure,'add_note'):failure.add_note('Deep request record: '+repr(persistence))
   if failure is not None:
    req.fail(failure);raise failure
   req.finish(reason);record['final_metadata']=copy.deepcopy(req.metadata);record['transport_finish_reason']=getattr(req.finish_reason,'value',str(req.finish_reason));save()
 finally:
  # Pending requests finish only after the original Native reset/drain boundary.
  # No admission lock or HTTP join is held across a Native operation.
  primary=sys.exc_info()[1]
  if primary is None:
   context.reset()
   with engine._lock:pending=list(engine._requests.values())
   for request in pending:
    if not request.is_finished():request.finish(request.abort_reason or Reason.CANCELED)
  else:
   with engine._lock:pending=list(engine._requests.values())
   for request in pending:
    if not request.is_finished():request.fail(primary)

def owner_main(config,prepared,stop_intent):
 configure_process(config);run=Run(config);report=dict(identity=run.identity(),completed=False,Model_READY=False,normal_Native_shutdown=False,requests=[],Native_SHA256=config['Native_SHA256'],TP=8,PP=1,HC_mode=0)
 path=run.path/'owner.json'
 def save():
  t=path.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2));t.replace(path)
 try:
  _,receipt,_=verify_native_chain();assert receipt['library_SHA256']==config['Native_SHA256']
  torch=torch_setup();safe_open=safe_open_dependency();cfg,mapping=read_model_metadata(config['model_path'])
  binding=bind_library(config['Native_library'],config['Native_SHA256'])
  context=NativeContext(binding,receipt,config=config,cfg=cfg,mapping=mapping,torch=torch,safe_open=safe_open,report=report,deadline_check=run.deadline_check)
  with context:
   policy=RequestPolicy(context,config);primary=None
   try:
    report['fixed_profile']=policy.ready();save();run.write('ready-engine.json',dict(loaded=True,loaded_key_count=report['loaded_key_count'],Native_SHA256=config['Native_SHA256'],HC_mode=0,rows=1,capacity=32768,affinity=report['owner_affinity_after_weight_load'],postload_free=report['GPU_memory_after_load'],fixed_profile=report['fixed_profile']))
    serve(context,prepared,config,run,stop_intent,report,save,policy=policy)
   except BaseException as error:primary=error;raise
   finally:
    try:policy.close()
    except BaseException as cleanup:
     if primary is None:raise
     if hasattr(primary,'add_note'):primary.add_note('Fixed profile close: '+repr(cleanup))
  report.update(completed=True,Model_READY=False);save()
  run.write('closed-engine.json',dict(normal=report['normal_Native_shutdown'],engine_destroy_returned_zero=report['engine_worker_shutdown_completed'],observed_all8_threads_exited=report['observed_all8_threads_exited']))
 except BaseException as error:
  report.update(completed=True,passed=False,error=repr(error),traceback=traceback.format_exc())
  for persist in [save,lambda:run.fail(error,stage=0)]:
   try:persist()
   except BaseException as persistence:
    if hasattr(error,'add_note'):error.add_note('Deep owner failure record: '+repr(persistence))
  raise
