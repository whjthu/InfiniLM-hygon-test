"""Pinned Modal W16 offline compatibility on previously captured real six-tap target prefixes."""
import pathlib,json,hashlib,time,os,gc,statistics,signal,traceback
signal.alarm(500)
import numpy as np
import torch
from safetensors import safe_open
from transformers import DynamicCache
from modal_reference_r4258 import DFlashDraftModel,Qwen3Config
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('kimi-block16-draft-r9112-'+tag+'.json');assert not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();began=time.monotonic();report=dict(hardware='BW1000',level='REAL_CAPTURE_OFFLINE_DRAFT_ONLY',status='RUNNING',actual_handle=tag,completed=False,raw_activations_stay_remote=True,no_target_Model_load=True,no_online_acceptance_claim=True,no_speculative_speedup_claim=True,cases=[]);model=embed=head=cache=features=None

def save():
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
def timed(fn):
 torch.cuda.synchronize();start=time.perf_counter();x=fn();torch.cuda.synchronize();return x,time.perf_counter()-start
try:
 torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 report.update(torch_version=torch.__version__,HIP_version=torch.version.hip,GPU_name=torch.cuda.get_device_name(0),initial_free_total=list(torch.cuda.mem_get_info()));assert report['initial_free_total'][0]>16<<30
 verified=json.loads((r/'results/Modal-draft-r4255-data-verified.json').read_bytes());contract=json.loads((r/'incoming/modal_reference_contract_r4258.json').read_bytes());checkpoint=r/'models/Kimi-K3-Modal-DFlash-c192d15a-r4255/model.safetensors';assert checkpoint.stat().st_size==verified['bytes']==5196953912;assert verified['weights_sha256']=='0f0bd39d19c2dd0249de4f8216180f9729963402f902d50f19a156a8d358b733';assert sha(r/'incoming/modal_reference_r4258.py')==contract['module_sha256'];report['weight_binding']=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,mtime_ns=checkpoint.stat().st_mtime_ns,prior_verified_receipt_SHA256=sha(r/'results/Modal-draft-r4255-data-verified.json'),prior_full_SHA256=verified['weights_sha256'],current_full_SHA_reread=False)
 cfg=Qwen3Config(**contract['config']);cfg._attn_implementation='sdpa'
 with torch.device('meta'):model=DFlashDraftModel(cfg)
 assert {k:list(v.shape)for k,v in model.state_dict().items()}=={k:v['shape']for k,v in contract['tensor_metadata'].items()};assert model.target_layer_ids==[19,37,54,66,78,90]
 start=time.perf_counter();state={}
 with safe_open(checkpoint,framework='pt',device='cpu')as f:
  for name in f.keys():state[name]=f.get_tensor(name)
 model.load_state_dict(state,strict=True,assign=True);model.rotary_emb=type(model.rotary_emb)(cfg);model.eval().to('cuda');del state;torch.cuda.synchronize();report['load_backbone_seconds']=time.perf_counter()-start;assert all(v.dtype==torch.bfloat16 for v in model.parameters());assert all(v.dtype==torch.float32 for v in model.rotary_emb.buffers())
 target=pathlib.Path('/shared/Kimi-K3');mapping=json.loads((target/'model.safetensors.index.json').read_bytes())['weight_map'];tm=json.loads((r/'incoming/DFlash2-target-metadata-r9019.json').read_bytes());assert sha(target/'config.json')==tm['config_SHA256'];report['auxiliary_binding']={}
 def target_weight(key):
  shard=target/mapping[key];md=tm['tensors'][key]
  with shard.open('rb')as fp:n=int.from_bytes(fp.read(8),'little');hdr=fp.read(n)
  assert hashlib.sha256(hdr).hexdigest()==md['header_SHA256']
  with safe_open(shard,framework='pt',device='cpu')as f:w=f.get_tensor(key)
  assert list(w.shape)==[163840,7168]and w.dtype==torch.bfloat16;report['auxiliary_binding'][key]=dict(shard=str(shard),header_SHA256=md['header_SHA256'],shape=list(w.shape),dtype='BF16',identity_vocab=True);return w.to('cuda')
 start=time.perf_counter();embed=target_weight('language_model.model.embed_tokens.weight');head=target_weight('language_model.lm_head.weight');torch.cuda.synchronize();report['load_auxiliary_seconds']=time.perf_counter()-start;report['resident_allocated_bytes']=torch.cuda.memory_allocated();save()
 launch=json.loads((r/'draft-worker-r4813-launch.json').read_bytes());directory=r/'results'/('trained-draft-r4805-'+launch['session']);teacher=json.loads((r/'results/kimi-root-r4635-plan.json').read_bytes())['expected_AR_ids'];chunks=[];N=0;selected={1,10,30,60};digest_cache={};report['capture_source']=dict(session=launch['session'],requests_directory=str(directory),teacher_reference_SHA256=sha(r/'results/kimi-root-r4635-plan.json'),target_prefix_source='Oldactualcaptured6tap mixture; NOTcurrent997five-tap; not currentModeltransaction qualification.');report['feature_metadata']=[]
 with torch.inference_mode():
  for index in range(61):
   rp=directory/('request-'+str(index)+'.json');req=json.loads(rp.read_bytes());count=req['feature_tokens'];assert index==0 or not req['reset'];assert len(req['features'])==(6 if count else 0)
   if count:
    parts=[]
    for desc in req['features']:
     f=r/desc['file'];digest=digest_cache.get(str(f))
     if digest is None:digest=sha(f);digest_cache[str(f)]=digest
     assert digest==desc['sha256'];a=np.fromfile(f,np.uint16).reshape(-1,7168);assert a.shape[0]>=count;parts.append(torch.from_numpy(a[:count].copy()).view(torch.bfloat16).reshape(1,count,7168));report['feature_metadata'].append(dict(request=index,consumer=desc.get('consumer'),path=str(f),SHA256=digest,used_rows=count))
    chunks.append(torch.cat(parts,-1));N+=count
   assert req['anchor']==teacher[N-512]
   if index not in selected:continue
   host=torch.cat(chunks,1);assert list(host.shape)==[1,N,43008];features=host.to('cuda');assert torch.isfinite(features).all();assert len(teacher[N-512+1:N-512+16])==15
   for W in [8,16]:
    cache=DynamicCache();ids=torch.full((1,W),int(model.mask_token_id),device='cuda',dtype=torch.long);ids[0,0]=int(req['anchor']);noise=torch.nn.functional.embedding(ids,embed);positions=torch.arange(N+W,device='cuda').view(1,-1)
    def first():
     hidden=model(position_ids=positions,noise_embedding=noise,target_hidden=features,past_key_values=cache,use_cache=True);logits=model.compute_logits(hidden,lambda h:h@head.T);assert torch.isfinite(hidden).all()and torch.isfinite(logits).all();return logits[:,1:].argmax(-1)[0].cpu().tolist()
    initial,prime=timed(first);assert cache.get_seq_length()==N+W;cache.crop(N);empty=features[:,:0,:];samples=[];trials=[]
    def query():
     hidden=model(position_ids=positions[:,-W:],noise_embedding=noise,target_hidden=empty,past_key_values=cache,use_cache=True);logits=model.compute_logits(hidden,lambda h:h@head.T);assert torch.isfinite(hidden).all()and torch.isfinite(logits).all();return logits[:,1:].argmax(-1)[0].cpu().tolist()
    for rep in range(5):
     tokens,elapsed=timed(query);assert cache.get_seq_length()==N+W;cache.crop(N);samples.append(elapsed);trials.append(tokens)
    assert all(t==trials[0]for t in trials);expected=teacher[N-512+1:N-512+W];accepted=0
    for actual,gold in zip(trials[0],expected):
     if actual!=gold:break
     accepted+=1
    report['cases'].append(dict(request_index=index,prefix_tokens=N,anchor=req['anchor'],block_size=W,proposal_count=W-1,proposal_ids=trials[0],true_teacher_next_ids=expected,matched_prefix=accepted,useful_outputs_optimistic=accepted+1,cold_context_plus_query_seconds=prime,cached_query_seconds_samples=samples,cached_query_warm_median_seconds=statistics.median(samples[1:]),repeat_same_path_IDs=True,full_vs_cached_IDs_same=initial==trials[0],causal_layers=[bool(l.self_attn.is_causal)for l in model.layers],timing_scope='Hostcompletecacheddraftquery+head/finite/argmax/copy; excludesappend/capture/targetverify/commit/replay/emit. Offlineagreementnotservingacceptance.'));save();cache.crop(0);cache=None
   features=None;gc.collect();torch.cuda.empty_cache()
 report.update(completed=True,status='OFFLINE_W8_W16_REAL_OLD_CAPTURE_COMPLETE',new_target997_compatibility='MISSING_TAP54_CURRENT_WIRE5; private6tapfork required',Model_speedup_measured=False,peak_allocated_bytes=torch.cuda.max_memory_allocated());save()
except BaseException as e:report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
 model=embed=head=cache=features=None
 for name in ['state','host','parts','chunks','noise','ids','empty','a','positions']:globals().pop(name,None)
 gc.collect()
 if torch.cuda.is_initialized():torch.cuda.synchronize();torch.cuda.empty_cache()
 report['normal_owner_close']=True;report['elapsed_seconds']=time.monotonic()-began;save();print('KIMI_MODAL_BLOCK16_NORMAL',flush=True)
