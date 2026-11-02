"""One strictloaded draft residency, exact18freshContext whole8cycle plan, no target."""
from pathlib import Path
import argparse,os,sys,json,time,hashlib,traceback,gc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--deadline-unix',required=True,type=float);a=a.parse_args()
 D=Path(__file__).resolve().parent;R=Path('/refactor');B=D.parent;old=R/'incoming/Kimi-long-draft-memory-floor-r362b'
 assert os.environ.get('KIMI_PREFIX_CACHE_R376')=='1'and 30<a.deadline_unix-time.time()<=600
 assert sha(D/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']
 assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())and all(sha(n)==h for n,h in m['dependencies_SHA256'].items())
 sys.path[:0]=[str(D),str(B),str(old),str(R/'incoming/Kimi-long-draft-RPC-CPU-r333'),str(R/'incoming')]
 import torch,worker
 from sequence_consumer import run_plan,plan,ALERT,INVESTIGATE
 assert Path(worker.__file__).resolve()==old/'worker.py'and ALERT==.02 and INVESTIGATE==.10
 runtime_dir,incoming=worker.verify_sources(old,R,m['worker_manifest_SHA256'])
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi376-whole-sequence-'+step+'.json');assert not out.exists()
 q=dict(completed=False,passed=False,actual_step=step,source_manifest_SHA256=a.manifest_sha,worker_manifest_SHA256=m['worker_manifest_SHA256'],GPU_cards=1,target_cards=0,profile=False,Model_gain='UNKNOWN',acceptance='UNKNOWN',quality_qualified=False,no_retry_reload=True,synthetic_committed_prefix=True,plan=plan(),sequence_artifacts=[],timing_scope='Primary continuouschecked8cycle wall includes complete8query+7acceptedappend+pending/cat/fences plus stage/canary/CPUreadback/inline numeric/save overhead. Complete dispatch wall sums separate, not standaloneemitter/Model. Initialprefix/warm/startup/retireclose allreported separately.',substage_scope='CoreLinear compute/cat timers are Pythonhost launch enclosures, not GPUevents/device latency',process_shape_scope='Firstprocess/shape calls identified byfixedorder; physicalGPUcold and selectedSDPAalgorithm not established')
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2)+'\n');p.replace(out)
 def remaining():assert time.time()<a.deadline_unix-10,'Original absolute GPU deadline exhausted'
 runtime=None;start=time.perf_counter();save()
 try:
  torch.set_num_threads(1)
  with torch.inference_mode():
   remaining();runtime=worker.Runtime(R,runtime_dir,incoming,q);remaining()
   model=runtime.loaded.worker.model;head=runtime.head;base_codec=type(runtime.session.codec);runtime.session.close()
   signature=[(n,id(v),v.data_ptr(),tuple(v.shape),str(v.dtype),str(v.device))for n,v in model.named_parameters()]
   modules=[(n,id(v))for n,v in model.named_modules()]
   q['torch_runtime']=dict(torch_version=torch.__version__,HIP_version=torch.version.hip,default_dtype=str(torch.get_default_dtype()),matmul_allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
   torch.manual_seed(324);tick=time.perf_counter();base=(torch.randn(1,512,35840,dtype=torch.bfloat16)*.1).contiguous()
   raw=base.view(torch.uint16).numpy().tobytes();q['fixture512_SHA256']=hashlib.sha256(raw).hexdigest()
   q['CPU_fixture_generation_wall_s']=time.perf_counter()-tick
   def body(past,n):
    if n==512:return raw
    indices=(torch.arange(past,past+n)%512).long();part=base.index_select(1,indices)
    return part.contiguous().view(torch.uint16).numpy().tobytes()
   current=None;fence_start=0
   def sequence_save(value):
    nonlocal current,fence_start
    if current!=value['ordinal']:
     current=value['ordinal'];fence_start=len(q.get('memory_fences',[]));q['current_sequence']=current;save()
    p=R/'results'/('Kimi376-sequence-'+step+'-'+str(current)+'.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
   def finished(value):
    p=R/'results'/('Kimi376-sequence-'+step+'-'+str(value['ordinal'])+'.json')
    q['sequence_artifacts'].append(dict(ordinal=value['ordinal'],L=value['L'],phase=value['phase'],cache=value['cache'],path=str(p),SHA256=sha(p),fence_start=fence_start,fence_end=len(q.get('memory_fences',[])),queries=len(value['queries']),commands=len(value['commands']),passed=value['passed'],continuous_checked_8cycle_wall_s=value['continuous_checked_8cycle_wall_s'],engine_dispatch_8cycle_sum_s=value['engine_dispatch_8cycle_sum_s'],prefix_begin_append_seal_wall_s=value['prefix_begin_append_seal_wall_s'],retire_close_wall_s=value['retire_close_wall_s']))
    save();gc.collect()
   records,refs=run_plan(model,head,base_codec,'cuda:0',35840,runtime.loaded.receipt['full_SHA256'],body,sequence_save,remaining=remaining,after_sequence=finished)
   assert len(records)==18 and sum(len(r['queries'])for r in records)==144 and sum(len(r['commands'])for r in records)==840
   assert len(q['memory_fences'])==912 and all(r['normal_session_close']for r in records)
   assert signature==[(n,id(v),v.data_ptr(),tuple(v.shape),str(v.dtype),str(v.device))for n,v in model.named_parameters()]
   assert modules==[(n,id(v))for n,v in model.named_modules()]and all(not v._forward_hooks and not v._forward_pre_hooks for v in model.modules())
   q.update(passed=True,normal_session_close=True,all18_contexts_closed=True,original_model_module_parameter_identity_unchanged=True,completed_query_count=144,completed_command_count=840,qualification='Execution/finite/state/guard/checkpoint numerical investigation only; .10 protection and .02 alerts are not modelquality boundaries. Syntheticaccepted8 doesnotmeasureacceptance; no targetreload orModelgain.')
 except BaseException as e:
  q.update(error=repr(e),traceback=traceback.format_exc())
  if q.get('current_sequence')is not None:
   p=R/'results'/('Kimi376-sequence-'+step+'-'+str(q['current_sequence'])+'.json')
   if p.exists():q['partial_sequence_artifact']=dict(path=str(p),SHA256=sha(p))
  raise
 finally:
  try:
   if runtime is not None:runtime.close();q['normal_runtime_close']=True
  except BaseException as e:q.update(passed=False,cleanup_error=repr(e));raise
  finally:gc.collect();q.update(completed=True,total_driver_wall_s=time.perf_counter()-start);save()
 print(json.dumps(dict(completed=q['completed'],passed=q['passed'],output=str(out))))

if __name__=='__main__':main()
