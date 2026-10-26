import pathlib,json,hashlib,time,traceback,os,ctypes,gc,signal
signal.alarm(820)
import sys
sys.path.append("/refactor/incoming")
import torch
from query_baseline_r9188 import query_batched
from query_projected_r9200 import query_projected
from projected_context_r9200 import ProjectedContextStore
from draft_context_r9128 import ContextStore
import socket,threading
assert os.environ.get("KIMI_TRUE_DRAFT_BATCH_R9142")=="1"
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
torch.set_num_threads(1);r=pathlib.Path('/refactor');inc=r/'incoming';label='Kimi-draft-projected-context-fixture-r9200';out=r/'results'/(label+'-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');start=time.monotonic();d={'status':'RUNNING','level':'trained draft mechanics synthetic features','hardware':'BW1000','GPU_count':1,'target_acceptance_qualified':False,'spec_speedup_claim':False,'phases':[],'cases':[]};loaded=None;head=None;mapping=None
def save():
 t=out.with_suffix('.tmp');t.write_bytes((json.dumps(d,indent=2)+'\n').encode());t.replace(out)
def timed(name,fn):
 torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return v,time.perf_counter()-t
def filehash(path,offset,length):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb')as f:
  f.seek(offset)
  while length:
   b=f.read(min(length,8<<20));assert b;h.update(b);length-=len(b)
 return h.hexdigest()
def error(a,b):
 aa=a.float();bb=b.float();return {'finite':bool(torch.isfinite(aa).all()and torch.isfinite(bb).all()),'maxabs':float((aa-bb).abs().max()),'relativeRMS':float(((aa-bb).square().mean().sqrt()/bb.square().mean().sqrt().clamp_min(1e-20))),'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1)))}
try:
 m=json.loads((inc/'DFlash2-trained-manifest-r9019.json').read_text());assert all(hashlib.sha256((inc/n).read_bytes()).hexdigest()==h for n,h in m['incoming_source_SHA256'].items());meta=json.loads((inc/'DFlash2-target-metadata-r9019.json').read_text());receipt=json.loads((r/'results/DFlash2-CDN-detached-r9018.json').read_text());assert receipt['completed']and receipt['bytes']==7385765464 and receipt['prefix_SHA256']=='1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2'
 assert torch.cuda.device_count()==1;torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();d.update(torch_version=torch.__version__,HIP_version=torch.version.hip,initial_memory={'free':free,'total':total});assert free>16<<30
 class Dl(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('sym',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 lib=ctypes.CDLL('libc.so.6');lib.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];sym=ctypes.CDLL(torch._C.__file__);info=Dl();assert lib.dladdr(ctypes.cast(sym.PyInit__C,ctypes.c_void_p),ctypes.byref(info));d['Torch_actual_dladdr']=info.name.decode();d['Torch_C_SHA256']=hashlib.sha256(pathlib.Path(torch._C.__file__).read_bytes()).hexdigest();save()
 checkpoint=r/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors';hmeta=meta['tensors']['language_model.lm_head.weight'];emeta=meta['tensors']['language_model.model.embed_tokens.weight'];shard=pathlib.Path(hmeta['shard'].replace(str(pathlib.Path('/public/home/aclsylqidf/shared')),'/shared'));assert hashlib.sha256((pathlib.Path('/shared/Kimi-K3/config.json')).read_bytes()).hexdigest()==meta['config_SHA256']
 with shard.open('rb')as f:n=int.from_bytes(f.read(8),'little');raw=f.read(n);assert hashlib.sha256(raw).hexdigest()==hmeta['header_SHA256']
 mapping=torch.from_file(str(shard),shared=False,size=shard.stat().st_size,dtype=torch.uint8);a,b=hmeta['data_offsets'];weight=mapping.narrow(0,hmeta['payload_offset']+a,b-a).view(torch.bfloat16).reshape(hmeta['shape']).to('cuda:0');head=ExternalTargetHead(weight,values_compatibility=None,proof={'actual_target_key':'language_model.lm_head.weight','shard_header_SHA256':hmeta['header_SHA256']})
 loaded,load_s=timed('load',lambda:load_pinned_worker(checkpoint,head,enabled=True,device='cuda:0',source_dir=inc));d['load_s']=load_s;d['loader_receipt']=loaded.receipt
 model=loaded.worker.model


 import copy,statistics
 def parameter_hash():
  digest=hashlib.sha256()
  for module in [model.context_proj,model.context_norm]:
   for name,p in module.named_parameters():digest.update(name.encode());digest.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
  return digest.hexdigest()
 hash_t=time.perf_counter();weights_hash=parameter_hash();d['context_weights_hash_before']=weights_hash;d['weights_hash_validation_s']=time.perf_counter()-hash_t
 d.update(scope='onlypackedappend Linear+rowwiseRMS projectedctx cache; noattentionlayerKV; qualifiedKmax+2Dhead botharms',synthetic_features=True,target_acceptance_qualified=False,Model_speedup_claim=False,uniform_prefill=[128,384,512],C=[1,8,16,32],append_rows=[1,3,7,1,3,7],ABBA_blocks=3,raw_features_H2D_and_network_charged=False)
 def raw_fork(state):
  out=ContextStore(state.max_requests);out.epoch=state.epoch;out.values=state.values.copy();out.retired=state.retired.copy();return out
 def run_trial(raw,cache,epoch,items,parts,Ks,cached,profile=False):
  state=cache.fork()if cached else raw_fork(raw);torch.cuda.synchronize();tick=time.perf_counter();points=[]
  def point(name):
   if profile:
    event=torch.cuda.Event(enable_timing=True);event.record();points.append((name,event))
  point('append_begin');count=state.projection_calls if cached else 0
  if cached:contexts=state.append_many(epoch,items,parts,profile=profile)
  else:contexts=state.append_many(epoch,items,parts,lambda a,b:torch.cat([a,b],dim=1))
  append_enqueued=time.perf_counter();point('append_enqueued');anchors=[163584+q['request']%2 for q in items]
  if cached:tokens,debug=query_projected(model,head,contexts,anchors,Ks,enabled=True,return_debug=True,profile=profile)
  else:tokens,debug=query_batched(model,head,contexts,anchors,Ks,enabled=True,return_debug=True,crop_head=True,flatten_head=True,profile=profile)
  values=tokens.cpu().tolist();requested=[row[:K]for row,K in zip(values,Ks)];torch.cuda.synchronize();wall=time.perf_counter()-tick
  record={'mode':'cached'if cached else'fullctx','full_append_query_D2H_s':wall,'append_host_enqueue_s':append_enqueued-tick,'active_C':len(items),'new_rows':sum(q['rows']for q in items),'max_context_length':max(q['past']+q['rows']for q in items),'projection_calls_in_append':state.projection_calls-count if cached else 0,'projection_calls_in_query':0 if cached else 1,'profile':profile}
  if cached:assert state.projection_calls-count==1;record['append_metadata']=state.last_append
  if profile:
   def elapsed(ps):return[{'from':a,'to':bb,'stream_elapsed_ms':float(x.elapsed_time(y))}for(a,x),(bb,y)in zip(ps,ps[1:])]
   record['append_stream_ms']=float(points[0][1].elapsed_time(points[1][1]));record['append_internal_segments']=elapsed(state.last_profile)if cached else[];record['query_segments']=elapsed(debug['event_points']);record['query_stream_ms']=float(debug['event_points'][0][1].elapsed_time(debug['event_points'][-1][1]));record['no_segment_drain']=True;record['warning']='streamintervals include launchgaps/wait, append enqueue hosttime isnot GPUcompletion; use fullwall'
  assert all(len(row)==K for row,K in zip(requested,Ks));return record,requested,debug['hidden'],state
 persistent={'raw':ContextStore(),'cache':ProjectedContextStore(model,weights_hash)}
 def one_geometry(C,N,epoch):
  ids=([17]if C==1 else(list(range(1,32,4))if C==8 else(list(range(1,33,2))if C==16 else list(range(1,33)))));raw=persistent['raw'];raw.begin(epoch);cache=persistent['cache'];cache.begin(epoch);active=ids[:];rounds=[];K={1:7,8:7,16:3,32:1}[C]
  for number,width in enumerate([N,1,3,7,1,3,7]):
   order=active[::-1]if number%2 else active;items=[{'request':i,'past':raw.values.get((epoch,i),(0,None))[0],'rows':width}for i in order]
   with torch.inference_mode(False):parts=[torch.randn((1,width,35840),device='cuda',dtype=torch.bfloat16)*.1 for i in order]
   versions=[p._version for p in parts];Ks=[K]*len(items);cold=[]
   if number==0:
    for cached in [False,True]:cold.append(run_trial(raw,cache,epoch,items,parts,Ks,cached)[0])
   for cached in [False,True]:run_trial(raw,cache,epoch,items,parts,Ks,cached)
   rows=[];committed={}
   for block in range(3):
    for index,cached in enumerate([False,True,True,False]):
     row,values,hidden,state=run_trial(raw,cache,epoch,items,parts,Ks,cached);row.update(block=block,ABBA_index=index);rows.append(row);committed[cached]=state
   profiles=[];values=[];hidden=[]
   for cached in [False,True]:
    row,v,h,state=run_trial(raw,cache,epoch,items,parts,Ks,cached,True);profiles.append(row);values.append(v);hidden.append(h.detach().clone());assert bool(torch.isfinite(h).all())
   delta=hidden[1].float()-hidden[0].float();rms=float(delta.square().mean().sqrt()/hidden[0].float().square().mean().sqrt().clamp_min(1e-20));agreement=sum(a==b for x,y in zip(values[0],values[1])for a,b in zip(x,y))/sum(Ks);assert versions==[p._version for p in parts];raw,cache=committed[False],committed[True];assert all(raw.values[k][0]==cache.values[k][0]for k in raw.values)
   old=statistics.median(x['full_append_query_D2H_s']for x in rows if x['mode']=='fullctx');new=statistics.median(x['full_append_query_D2H_s']for x in rows if x['mode']=='cached');rounds.append({'round':number,'phase':'initial_fill'if number==0 else'append','append_rows_per_request':width,'active_C':len(order),'logical_ids':order,'K':K,'lengths':[raw.values[(epoch,i)][0]for i in order],'cold_initial':cold,'rows':rows,'profile':profiles,'full_median_s':old,'cached_median_s':new,'query_rate_ratio':old/new,'latency_reduction_fraction':1-new/old,'hidden_relativeRMS_diagnostic':rms,'proposal_prefix_agreement_diagnostic':agreement,'finite':True,'input_version_unchanged':True,'token_identity_required':False})
   if number==3 and C>1:
    for i in active[1::2]:raw.retire(epoch,i);cache.retire(epoch,i)
    active=active[::2]
   # Invalidpast is whole-batch rejected without mutating any stored cache state.
   bad=[{'request':i,'past':cache.values[(epoch,i)][0],'rows':1}for i in active];bad[-1]['past']+=1;before={k:(v[0],v[1])for k,v in cache.values.items()}
   try:cache.append_many(epoch,bad,[parts[0][:,:1]for i in active]);raise RuntimeError('badpast accepted')
   except AssertionError:pass
   assert before.keys()==cache.values.keys()and all(v[0]==cache.values[k][0]and v[1]is cache.values[k][1]for k,v in before.items())
  # One packed full-context reference, not per-request small GEMMs, outside measured windows.
  ids=active;whole=torch.cat([raw.values[(epoch,i)][1][0]for i in ids],0);expected=model.context_norm(model.context_proj(whole));actual=torch.cat([cache.values[(epoch,i)][1][0]for i in ids],0);diff=actual.float()-expected.float();ctx_rms=float(diff.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-20));assert bool(torch.isfinite(actual).all());raw.close();cache.close();persistent['raw']=raw;persistent['cache']=cache;d['cases'].append({'C':C,'prefill_rows':N,'epoch':epoch,'rounds':rounds,'projected_context_RMS_diagnostic':ctx_rms,'closed':True,'no_layer_KV_cache':True,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_free_bytes':torch.cuda.mem_get_info()[0]});save()
 with torch.inference_mode():
  torch.manual_seed(9200);epoch=0
  for C in [1,8,16,32]:
   for N in [128,384,512]:epoch+=1;one_geometry(C,N,epoch)
 d['context_weights_hash_after']=parameter_hash();assert d['context_weights_hash_after']==weights_hash;d.update(status='TRAINED_PROJECTED_CONTEXT_COMPONENT_COMPLETED',completed=True,query_geometry_cases=12,append_rounds=72,initial_fill_rounds=12,normal_closed=True)
except BaseException as e:d.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
 if loaded is not None:loaded.close()
 if head is not None:head.close()
 d['normal_close']=True;save()
print('KIMI_DRAFT_PROJECTED_CTX_R9200_NORMAL',flush=True)
