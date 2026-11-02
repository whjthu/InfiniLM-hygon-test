"""One strictloaded draft, one full query, fixed firstlayer controls; never a speed test."""
from pathlib import Path
import os,sys,time,json,hashlib,argparse,traceback,gc
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class FirstDivergence(Exception):pass

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--deadline-unix',required=True,type=float);a=a.parse_args()
 D=Path(__file__).resolve().parent;R=Path('/refactor');B=R/'incoming/Kimi-local-context-crop-r369';old=R/'incoming/Kimi-long-draft-memory-floor-r362b'
 assert os.environ.get('KIMI_FIRST_LAYER_R371')=='1'and 30<a.deadline_unix-time.time()<=600
 assert sha(D/'GPU_MANIFEST.json')==a.manifest_sha
 m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert m['GPU_GO']
 assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert all(sha(n)==h for n,h in m['dependencies_SHA256'].items())
 sys.path[:0]=[str(D),str(B),str(B/'gpu-v1'),str(old),str(R/'incoming/Kimi-long-draft-RPC-CPU-r333'),str(R/'incoming')]
 import torch,worker
 from window_context import WindowContext
 from first_layer import first_layer,capture_original_query_input,THRESHOLD
 assert THRESHOLD==.02 and Path(worker.__file__).resolve()==old/'worker.py'
 runtime_dir,incoming=worker.verify_sources(old,R,m['worker_manifest_SHA256'])
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi371-firstlayer-'+step+'.json');assert not out.exists()
 q=dict(completed=False,passed=False,actual_step=step,GPU_cards=1,target_cards=0,source_manifest_SHA256=a.manifest_sha,worker_manifest_SHA256=m['worker_manifest_SHA256'],profile=False,performance_qualified=False,Model_gain='UNKNOWN',acceptance='UNKNOWN',no_retry_reload=True,no_full_ON_or_ABBA=True)
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2)+'\n');p.replace(out)
 def remaining():assert time.time()<a.deadline_unix-10,'Original GPU absolute deadline'
 def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()
 def rope_state(module):return {n:dict(id=id(t),pointer=t.data_ptr(),dtype=str(t.dtype),shape=list(t.shape),SHA256=digest(t))for n,t in module.named_buffers()}
 runtime=None;save();start=time.perf_counter()
 try:
  torch.set_num_threads(1)
  with torch.inference_mode():
   remaining();runtime=worker.Runtime(R,runtime_dir,incoming,q);remaining()
   model=runtime.loaded.worker.model;runtime.session.context.close()
   c=WindowContext(model,runtime.loaded.receipt['full_SHA256'],enabled=True);runtime.session.context=c
   seq=0
   def command(cmd,body=b'',**fields):
    nonlocal seq
    remaining();meta=dict(cmd=cmd,seq=seq)
    if cmd!='close':meta.update(epoch=1,request=1)
    meta.update(fields);answer=runtime.session.dispatch(meta,body);seq+=1;return answer
   L=32638;command('begin',prompt_tokens=L)
   torch.manual_seed(324);base=(torch.randn(1,512,35840,dtype=torch.bfloat16)*.1).contiguous().view(torch.uint16).numpy().tobytes()
   q['fixture512_SHA256']=hashlib.sha256(base).hexdigest();tick=time.perf_counter()
   for i,past in enumerate(range(0,L,512)):
    n=min(512,L-past);command('append_prompt',base[:n*35840*2],past=past,rows=n,capture_epoch=i+1)
   command('seal',past=L);q['append_seal_wall_s']=time.perf_counter()-tick;save()
   raw_before=digest(c.raw);ctx_before=digest(c.projected);modules=[(n,id(v))for n,v in model.named_modules()]
   def save_whole(value):q['original_whole_query']=value;save()
   held,whole,seq=capture_original_query_input(runtime,seq,1,save_whole);remaining()
   cp=torch.arange(L,device=c.projected.device).view(1,-1);dp=L+torch.arange(8,device=c.projected.device).view(1,-1)
   keys=torch.cat([cp,dp],1);mask=((dp[:,:,None]-keys[:,None,:]).abs()<4096)[:,None]
   rope_before=rope_state(model.layers[0].self_attn.rotary_emb)
   def save_first(value):
    q['firstlayer']=value;save()
    check=value.get('OFF_replay_to_original_attention_output')
    if check is not None and (not check['finite']or check['above_original_02_diagnostic']):raise AssertionError('Sameinput OFF replay does not match original; stop attribution')
    stages=value.get('producer_stages',[])
    first=next((v for v in stages if not v['finite']or v['NRMSE']>0),None)
    if first is not None:
     q['first_observed_producer_divergence']=first;save()
     if not first['finite']:raise AssertionError('Nonfinite firstproducer; stop diagnosis')
     raise FirstDivergence(first['stage'])
   try:first_layer(model.layers[0].self_attn,held['draft_hidden'],c.projected,dp,cp,mask,save_first,fence=runtime.session.codec.fence,original_output=held['original_output'])
   except FirstDivergence as e:q['planned_stop_after_first_producer']=str(e)
   remaining();q['RoPE_after']=rope_state(model.layers[0].self_attn.rotary_emb);assert rope_before==q['RoPE_after']
   assert raw_before==digest(c.raw)and ctx_before==digest(c.projected)
   assert modules==[(n,id(v))for n,v in model.named_modules()]
   assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
   q.update(full_context_SHA256_unchanged=True,RoPE_unchanged=True,model_module_identity_unchanged=True)
   command('retire');assert command('close')['closed'];q['normal_session_close']=True
   q.update(protocol_commands=seq,diagnostic_execution_completed=True,passed=True,qualification='Diagnostic execution completed only; firstobservedfiniteprecision split is not quality failure or an optimization/acceptance qualification. Threshold .02 unchanged, no completeON query/retry.')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  try:
   if runtime is not None:runtime.close();q['normal_runtime_close']=True
  except BaseException as e:q.update(passed=False,cleanup_error=repr(e));raise
  finally:gc.collect();q.update(completed=True,total_wall_s=time.perf_counter()-start);save()
 print(json.dumps(dict(completed=q['completed'],passed=q['passed'],output=str(out))))

if __name__=='__main__':main()
