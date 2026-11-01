"""One pinned draft, synthetic repeated short features. Geometry/cost only, no target."""
from pathlib import Path
import os,json,hashlib,sys,time,signal,traceback,statistics,gc
R=Path('/refactor');D=R/'incoming/Kimi-C1-long-draft-cost-r324';C=R/'incoming/Kimi-C1-long-draft-r323';I=R/'incoming'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_DRAFT_COST_MANIFEST_SHA']
m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
assert sha(C/'SOURCE_MANIFEST.json')==m['consumer_manifest_SHA256'];cm=json.loads((C/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(C/n)==h for n,h in cm['files_SHA256'].items())
assert sha(I/'DFlash2-trained-manifest-r9019.json')==m['trained_manifest_SHA256'];tm=json.loads((I/'DFlash2-trained-manifest-r9019.json').read_bytes());assert all(sha(I/n)==h for n,h in tm['incoming_source_SHA256'].items())
cp=R/'candidates/Kimi-C1-long-draft-CPU-r323/CPU-prepared.json';assert sha(cp)==m['CPU637_receipt_SHA256'];cpu=json.loads(cp.read_bytes());assert cpu['completed']and cpu['passed']and cpu['actual_step']=='910407.637'
left=int(float(os.environ['KIMI_DRAFT_COST_GPUSTART_UNIX'])+570-time.time());assert left>30;signal.alarm(left)
sys.path[:0]=[str(C),str(I),str(D)]
import torch
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
from long_context_r323 import LongCommittedFeatureContext
from legacy_query import query_projected
torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Kimi-C1-long-draft-cost-r324-'+step+'.json');assert not out.exists()
q={'UTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'actual_step':step,'completed':False,'passed':False,'normal_close':False,'source_manifest_SHA256':m and sha(D/'SOURCE_MANIFEST.json'),'consumer_manifest_SHA256':sha(C/'SOURCE_MANIFEST.json'),'GPU_count':1,'target_cards':0,'HTTP':0,'synthetic_expanded_short_features':True,'target_acceptance_qualified':False,'Model_speedup_claim':False,'profile':False,'cases':[]}
loaded=head=store=model=mapping=weight=None
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2)+'\n');p.replace(out)
def timed(fn):
 torch.cuda.synchronize();t=time.perf_counter();value=fn();torch.cuda.synchronize();return value,time.perf_counter()-t
def query_once(epoch,anchor=163584):
 result=store.query(epoch,1,head,anchor,7);assert result['mode']=='draft';values=result['values'];assert tuple(values.shape)==(1,7)
 answer=values.cpu().tolist();assert all(type(x)is int and 0<=x<163840 for x in answer[0]);return answer
try:
 free,total=torch.cuda.mem_get_info();assert free>16<<30;q.update(initial_free_bytes=free,total_bytes=total,Torch_version=torch.__version__,HIP_version=torch.version.hip,Torch_C_SHA256=sha(Path(torch._C.__file__)));save()
 with torch.inference_mode():
  t=time.perf_counter();meta=json.loads((I/'DFlash2-target-metadata-r9019.json').read_bytes());h=meta['tensors']['language_model.lm_head.weight'];shard=Path(h['shard'].replace('/public/home/aclsylqidf/shared','/shared'))
  assert sha(Path('/shared/Kimi-K3/config.json'))==meta['config_SHA256']
  with shard.open('rb')as f:n=int.from_bytes(f.read(8),'little');assert hashlib.sha256(f.read(n)).hexdigest()==h['header_SHA256']
  mapping=torch.from_file(str(shard),shared=False,size=shard.stat().st_size,dtype=torch.uint8);a,b=h['data_offsets'];weight=mapping.narrow(0,h['payload_offset']+a,b-a).view(torch.bfloat16).reshape(h['shape']).to('cuda:0');head=ExternalTargetHead(weight,values_compatibility=None,proof={'actual_target_key':'language_model.lm_head.weight','shard_header_SHA256':h['header_SHA256']})
  loaded=load_pinned_worker(R/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors',head,enabled=True,device='cuda:0',source_dir=I);model=loaded.worker.model;torch.cuda.synchronize();q['load_and_full_checkpoint_hash_wall_s']=time.perf_counter()-t;q['loader_receipt']=loaded.receipt
  assert len(model.layers)==5 and all(layer.self_attn.max_position_embeddings==32768 for layer in model.layers)
  torch.manual_seed(324);base=torch.randn(1,512,35840,dtype=torch.bfloat16)*.1;q['fixture']='Seed324 CPU BF16 short512x5tap expanded periodically; no realtargetactivation or acceptance';q['fixture_CPU_bytes']=base.numel()*base.element_size()
  for epoch,N in enumerate([1024,8192,32760],1):
   store=LongCommittedFeatureContext(model,loaded.receipt['full_SHA256'],enabled=True);store.begin(epoch,1,N-8);torch.cuda.reset_peak_memory_stats();init=[]
   for past in range(0,N-8,512):
    rows=min(512,N-8-past)
    def append_prompt():
     part=base[:,:rows].to('cuda:0');return store.append_prompt(epoch,1,past,part,torch.cuda.synchronize)
    result,wall=timed(append_prompt);del result;init.append({'rows':rows,'H2D_projection_copy_fence_wall_s':wall})
   store.seal_prompt(epoch,1)
   def committed():return store.append_committed(epoch,1,N-8,base[:,:8].to('cuda:0'),torch.cuda.synchronize)
   result,commit_wall=timed(committed);del result;assert store.length==N and bool(torch.isfinite(store.projected).all());init_peak=torch.cuda.max_memory_allocated()
   row={'N':N,'prompt_tokens':N-8,'committed_decode_tokens':8,'append_chunks':init,'prompt_H2D_append_wall_s':sum(x['H2D_projection_copy_fence_wall_s']for x in init),'committed8_H2D_append_wall_s':commit_wall,'payload_bytes':store.payload_bytes(),'init_peak_allocated_bytes':init_peak,'query_memory_before_bytes':torch.cuda.memory_allocated(),'no_draft_layer_KV_cache':True}
   if N==1024:
    legacy,ld=query_projected(model,head,[store.projected],[163584],[7],enabled=True,return_debug=True,profile=False)
    actual=store.query(epoch,1,head,163584,7,return_debug=True)['values'];delta=actual[1]['hidden'].float()-ld['hidden'].float();rms=float(delta.square().mean().sqrt()/ld['hidden'].float().square().mean().sqrt().clamp_min(1e-20));assert bool(torch.isfinite(actual[1]['hidden']).all())and rms<.01
    row['short_control']={'hidden_relativeRMS_diagnostic':rms,'proposal_equal_diagnostic':bool(torch.equal(actual[0],legacy)),'actual_old_new_query_math_same':True};del actual,legacy,ld,delta
   row['warmups']=[timed(lambda:query_once(epoch))[1]for _ in range(2)];torch.cuda.reset_peak_memory_stats();samples=[]
   for index in range(8):
    answer,wall=timed(lambda:query_once(epoch));samples.append({'index':index,'complete_query_D2H_wall_s':wall,'proposal_count':len(answer[0])})
   row.update(samples=samples,query_median_s=statistics.median(x['complete_query_D2H_wall_s']for x in samples),query_peak_allocated_bytes=torch.cuda.max_memory_allocated(),query_peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes_after_queries=torch.cuda.mem_get_info()[0])
   if N==32760:
    store.append_committed(epoch,1,N,base[:,:1].to('cuda:0'),torch.cuda.synchronize);tail=store.query(epoch,1,None,163584,7);assert tail['mode']=='ar_tail' and tail['values']is None;row['32761_AR_tail_no_query']=True
   store.retire(epoch,1);assert store.raw is None and store.projected is None;store.close();store=None;gc.collect();torch.cuda.empty_cache();row['normal_retired_closed']=True;q['cases'].append(row);save()
  assert len(q['cases'])==3 and all(len(x['samples'])==8 for x in q['cases']);q['passed']=True
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc()[-6000:]);raise
finally:
 errors=[]
 for name,obj in [('store',store),('loaded',loaded),('head',head)]:
  if obj is not None:
   try:obj.close()
   except BaseException as e:errors.append({'object':name,'error':repr(e)})
 store=loaded=head=model=mapping=weight=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();q.update(completed=True,normal_close=not errors,cleanup_errors=errors);save()
print('KIMI_LONG_DRAFT_COST_R324_NORMAL',flush=True)
