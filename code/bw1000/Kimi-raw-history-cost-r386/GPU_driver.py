"""One strictloaded draft, original fullquery,20 empty-context phase records."""
from pathlib import Path
import argparse,ctypes,gc,hashlib,importlib.util,json,os,socket,sys,time,traceback
from phase_plan import plan,expected,run_phase

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--deadline-unix',required=True,type=float);a=a.parse_args()
 D=Path(__file__).resolve().parent;R=Path('/refactor');old=R/'incoming/Kimi-long-draft-memory-floor-r362b'
 assert os.environ.get('KIMI_RAW_HISTORY_COST_R386')=='1'and 30<a.deadline_unix-time.time()<=600
 assert sha(D/'GPU_MANIFEST.json')==a.manifest_sha;m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']is True
 assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())and all(sha(p)==h for p,h in m['dependencies_SHA256'].items())
 sys.path[:0]=[str(D),str(old),str(R/'incoming/Kimi-long-draft-RPC-CPU-r333'),str(R/'incoming')]
 import torch,worker
 assert Path(worker.__file__).resolve()==old/'worker.py'
 runtime_dir,incoming=worker.verify_sources(old,R,m['worker_manifest_SHA256'])
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi386-raw-history-'+step+'.json');assert not out.exists()
 q=dict(completed=False,passed=False,actual_step=step,PID=os.getpid(),node=socket.gethostname(),GPU_cards=1,target_cards=0,profile=False,source_manifest_SHA256=a.manifest_sha,phase_records=[],no_retry_reload=True,Model_gain='UNKNOWN',acceptance='UNKNOWN',synthetic_periodic_features=True,plan=plan())
 def save():
  temp=out.with_suffix('.tmp');temp.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n');temp.replace(out)
 def remaining():
  if time.time()>=a.deadline_unix-10:raise TimeoutError('Original single-draft deadline; no replay')
 def metric(a,b):
  x=a.float();y=b.float();finite=bool(torch.isfinite(x).all()and torch.isfinite(y).all())
  return dict(finite=finite,NRMSE=float(torch.linalg.vector_norm(x-y)/(torch.linalg.vector_norm(x)+1e-30))if finite else None,maxabs=float((x-y).abs().max())if finite else None)
 runtime=None;start=time.perf_counter();save()
 try:
  torch.set_num_threads(1)
  with torch.inference_mode():
   remaining();runtime=worker.Runtime(R,runtime_dir,incoming,q);remaining()
   model=runtime.loaded.worker.model;head=runtime.head;base_codec=type(runtime.session.codec);runtime.session.close()
   from long_rpc import DraftSession
   from long_context_r323 import LongCommittedFeatureContext
   ctx_path=R/'incoming/Kimi-raw-history-elision-r384/context.py';assert sha(ctx_path)==m['context_source_SHA256']
   spec=importlib.util.spec_from_file_location('raw_history_context_r384',ctx_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   assert module.ProjectedOnlyContext.query is LongCommittedFeatureContext.query
   class ObservedContext(module.ProjectedOnlyContext):
    def query(self,*args,return_debug=False,**kwargs):
     result=super().query(*args,return_debug=True,**kwargs)
     if result['mode']=='draft':self.last_debug=result['values'][1]
     if not return_debug and result['mode']=='draft':result=dict(result,values=result['values'][0])
     return result
   class HeadTap:
    def __init__(self,original):self.original=original;self.last=None
    def __call__(self,x):self.last=self.original(x);return self.last
   signature=[(n,id(v),v.data_ptr(),tuple(v.shape),str(v.dtype),str(v.device))for n,v in model.named_parameters()]
   props=torch.cuda.get_device_properties(0);assert 'gfx936'in props.gcnArchName
   cuda=ctypes.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');cuda.cudaDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int];bus=ctypes.create_string_buffer(64);assert cuda.cudaDeviceGetPCIBusId(bus,64,0)==0;q['PCI']=bus.value.decode()
   tick=time.perf_counter();generator=torch.Generator(device='cpu').manual_seed(324);base=(torch.randn(1,512,35840,dtype=torch.bfloat16,generator=generator)*.1).contiguous();raw=base.view(torch.uint16).numpy().tobytes();bodies={}
   for item in plan():
    L=item['L']
    for past,n in [(p,min(512,L-p))for p in range(0,L,512)]+[(L+8*j,8)for j in range(7)]:
     key=(past%512,n)
     if key not in bodies:
      bodies[key]=raw if n==512 else base.index_select(1,(torch.arange(past,past+n)%512).long()).contiguous().view(torch.uint16).numpy().tobytes()
   q['fixture']=dict(seed=324,period_rows=512,raw_dim=35840,dtype='BF16',SHA256=hashlib.sha256(raw).hexdigest(),CPU_setup_wall_s=time.perf_counter()-tick,body_mapping='past modulo512; identical for each OFF/ON empty-context sample; no target generated features')
   references={};partial={};q['strict_load_and_fixture_setup_s']=time.perf_counter()-start
   for item in plan():
    remaining();torch.cuda.reset_peak_memory_stats(0);tap=HeadTap(head);c=ObservedContext(model,runtime.loaded.receipt['full_SHA256'],enabled=True,discard_raw=item['discard_raw']);session=DraftSession(c,tap,base_codec('cuda:0'),enabled=True)
    fence_start=len(q.get('memory_fences',[]));q['current_phase']=item['ordinal'];save()
    def phase_save(value):
     partial.clear();partial.update(value)
    try:record=run_phase(session,lambda past,n:bodies[(past%512,n)],item,remaining=remaining,save=phase_save)
    except BaseException:
     q['failed_phase']=dict(partial);save();raise
    # Readbacks/numeric are after the complete timed phase, not subtracted from
    # it. Their cost and retained diagnostic tensors remain in full driver cost.
    record['Torch_peak_allocated_bytes']=torch.cuda.max_memory_allocated(0);record['Torch_peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
    record['fence_start']=fence_start;record['fence_end']=len(q.get('memory_fences',[]));assert record['fence_end']-fence_start==expected(item)['fences']
    tick=time.perf_counter();values=dict(hidden=c.last_debug['hidden'].detach().float().cpu(),logits=tap.last.detach().float().cpu(),proposals=[v['proposals']for v in record['queries']])
    if item['L']not in references:references[item['L']]=values
    ref=references[item['L']];record['last_query_hidden']=metric(ref['hidden'],values['hidden']);record['last_query_logits']=metric(ref['logits'],values['logits'])
    record['all8_proposal_position_matches']=sum(a==b for aa,bb in zip(ref['proposals'],values['proposals'])for a,b in zip(aa,bb));record['proposal_positions']=56
    record['post_phase_numeric_and_save_scope']='Finalquery fullhidden/logits only; all8queries originalmandatoryfinite/proposalchecks retained. CPUreference/readbacks outsidephase, includedfull driverwall; no stageobserver/profiler.'
    record['post_phase_numeric_wall_s']=time.perf_counter()-tick
    q['phase_records'].append(record);save()
    assert record['last_query_hidden']['finite']and record['last_query_logits']['finite']
    if max(record['last_query_hidden']['NRMSE'],record['last_query_logits']['NRMSE'])>.02:raise RuntimeError('Unexpected same-math finite drift; diagnostics saved, no retry/quality claim')
    del session,c,tap,values;gc.collect()
   assert len(q['phase_records'])==20 and sum(len(v['queries'])for v in q['phase_records'])==160
   assert sum(len(v['commands'])for v in q['phase_records'])==1190 and len(q['memory_fences'])==1270
   assert signature==[(n,id(v),v.data_ptr(),tuple(v.shape),str(v.dtype),str(v.device))for n,v in model.named_parameters()]
   q.update(passed=True,all20_contexts_closed=True,original_model_parameter_identity_unchanged=True,normal_session_close=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  try:
   if runtime is not None:runtime.close();q['normal_Runtime_close']=True
  except BaseException as e:q.update(passed=False,cleanup_error=repr(e));raise
  finally:q.update(completed=True,whole_driver_wall_s=time.perf_counter()-start);save()
 return 0
if __name__=='__main__':raise SystemExit(main())
