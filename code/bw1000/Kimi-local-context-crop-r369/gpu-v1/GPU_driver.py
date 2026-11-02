"""One draft card, actual strict loader/head, complete session query wall, no target."""
from pathlib import Path
import os,sys,time,json,hashlib,argparse,traceback,gc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def schedule():return [('warm',False),('warm',True)]+[('measure',v)for v in [False,True,True,False]*5]

def version(t):
 try:return t._version
 except RuntimeError:return None

def tensor_sha(t):
 return hashlib.sha256(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy()).hexdigest()

def run_plan(runtime,report,save,body,*,lengths=(1024,8635,32638),remaining=lambda:None):
 import torch
 c=runtime.session.context;seq=0;capture=0
 def command(cmd,data=b'',**kw):
  nonlocal seq
  remaining();meta=dict(cmd=cmd,seq=seq)
  if cmd!='close':meta.update(epoch=epoch,request=1)
  meta.update(kw);out=runtime.session.dispatch(meta,data);assert out['seq']==seq
  seq+=1;return out
 report['queries']=[];report['contexts']=[]
 for epoch,L in enumerate(lengths,1):
  c.crop_local=False;command('begin',prompt_tokens=L)
  start=time.perf_counter()
  for past in range(0,L,512):
   n=min(512,L-past);capture+=1
   command('append_prompt',body(n),past=past,rows=n,capture_epoch=capture)
  command('seal',past=L)
  context=dict(L=L,append_seal_wall_s=time.perf_counter()-start,projection_calls=c.projection_calls)
  report['contexts'].append(context);save()
  model=c.model
  versions=[(name,id(p),p.data_ptr(),version(p))for name,p in model.state_dict(keep_vars=True).items()]
  raw_id=id(c.raw);projected_id=id(c.projected);pv=version(c.projected);rv=version(c.raw)
  state_start=time.perf_counter();raw_sha=tensor_sha(c.raw);projected_sha=tensor_sha(c.projected)
  context['immutable_check_before_wall_s']=time.perf_counter()-state_start
  baseline=None
  for trial,(phase,enabled)in enumerate(schedule()):
   remaining();c.crop_local=enabled
   tick=time.perf_counter();answer=command('query',past=L,anchor=163584,K=7);wall=time.perf_counter()-tick
   debug=c.last_debug
   assert answer['mode']=='draft'and len(answer['proposals'])==7
   assert id(c.raw)==raw_id and id(c.projected)==projected_id and version(c.projected)==pv and version(c.raw)==rv
   assert c.length==L and c.phase=='decode'and c.epoch==epoch
   # Numeric readbacks/CPU comparisons are outside full dispatch timing in both arms.
   values={name:debug[name].detach().float().cpu()for name in ['hidden','logits']}
   if baseline is None:baseline=values
   numeric={}
   for name,x in values.items():
    b=baseline[name];nr=float(torch.linalg.vector_norm(x-b)/(torch.linalg.vector_norm(b)+1e-30))
    assert bool(torch.isfinite(x).all())and nr<2e-2,(name,nr)
    numeric[name]=dict(NRMSE=nr,maxabs=float((x-b).abs().max()))
   report['queries'].append(dict(L=L,trial=trial,phase=phase,crop_local=enabled,complete_dispatch_wall_s=wall,handler_wall_s=answer['handler_wall_s'],proposals=answer['proposals'],numeric=numeric,local_context_rows=debug['local_context_rows'],global_context_rows=L,absolute_span=L+8,fixed_draft_rows=8,profile=False,state_unchanged=True))
   assert versions==[(name,id(p),p.data_ptr(),version(p))for name,p in model.state_dict(keep_vars=True).items()]
   save()
  state_start=time.perf_counter();assert raw_sha==tensor_sha(c.raw)and projected_sha==tensor_sha(c.projected)
  context.update(immutable_check_after_wall_s=time.perf_counter()-state_start,raw_context_SHA256=raw_sha,projected_context_SHA256=projected_sha,full_context_content_unchanged=True,weights_pointer_version_metadata_unchanged=True,weight_content_evidence='Pinned strict checkpoint loader plus read-only source path; inference tensors may lack version counters; no full GPU weight rehash claimed')
  assert command('retire')['removed'];assert c.raw is None and c.projected is None and c.last_debug is None
 assert command('close')['closed'];assert runtime.session.closed
 report['protocol_commands']=seq
 report['normal_session_close']=True

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--deadline-unix',required=True,type=float);a=a.parse_args()
 D=Path(__file__).resolve().parent;R=Path('/refactor');B=D.parent
 assert os.environ.get('KIMI_WINDOW_QUERY_R369')=='1'and 30<a.deadline_unix-time.time()<=600
 assert sha(D/'GPU_MANIFEST.json')==a.manifest_sha
 m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']is True
 assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert all(sha(Path(n))==h for n,h in m['dependencies_SHA256'].items())
 old=R/'incoming/Kimi-long-draft-memory-floor-r362b'
 sys.path[:0]=[str(D),str(B),str(old),str(R/'incoming/Kimi-long-draft-RPC-CPU-r333'),str(R/'incoming')]
 import torch,worker
 from window_context import WindowContext
 assert Path(worker.__file__).resolve()==old/'worker.py'
 runtime_dir,incoming=worker.verify_sources(old,R,m['worker_manifest_SHA256'])
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi369-query-'+step+'.json');assert not out.exists()
 q=dict(completed=False,passed=False,actual_step=step,source_manifest_SHA256=a.manifest_sha,worker_manifest_SHA256=m['worker_manifest_SHA256'],GPU_cards=1,target_cards=0,profile=False,Model_gain='UNKNOWN',acceptance='UNKNOWN',synthetic_seed324_periodic_features=True,timing='Complete actual DraftSession.dispatch query includes guards/fences, pack/window/layers/head/selector/finite/proposals CPU transfer; excludes wire TCP, prompt append and external numeric readbacks (reported separately)',no_retry_reload=True)
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2)+'\n');p.replace(out)
 def remaining():assert time.time()<a.deadline_unix-10,'Original absolute GPU deadline'
 runtime=None;save();start=time.perf_counter()
 try:
  torch.set_num_threads(1);remaining()
  with torch.inference_mode():
   runtime=worker.Runtime(R,runtime_dir,incoming,q);remaining()
   model=runtime.loaded.worker.model
   runtime.session.context.close();runtime.session.context=WindowContext(model,runtime.loaded.receipt['full_SHA256'],enabled=True)
   torch.manual_seed(324);base=torch.randn(1,512,35840,dtype=torch.bfloat16)*.1
   raw=base.contiguous().view(torch.uint16).numpy().tobytes();q['fixture512_SHA256']=hashlib.sha256(raw).hexdigest()
   def body(n):return raw[:n*35840*2]
   run_plan(runtime,q,save,body,remaining=remaining)
   q['passed']=True
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  try:
   if runtime is not None:runtime.close();q['normal_runtime_close']=True
  except BaseException as e:q.update(passed=False,cleanup_error=repr(e));raise
  finally:gc.collect();q.update(completed=True,total_wall_s=time.perf_counter()-start);save()
 print(json.dumps(dict(completed=q['completed'],passed=q['passed'],output=str(out))))

if __name__=='__main__':main()
