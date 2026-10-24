import base64,collections,ctypes,hashlib,json,os,pathlib,time,traceback
import torch
from collections.abc import Mapping
import torch.multiprocessing as mp
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');stage=int(os.environ['SLURM_PROCID']);tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
lib=ctypes.CDLL(str(r/'build/libmodern_glm_full_model_probe.so'));ptr=lambda x:ctypes.c_void_p(x.data_ptr())
lib.modern_glm_model_error.restype=ctypes.c_char_p;lib.modern_glm_model_id_size.restype=ctypes.c_size_t;lib.modern_glm_model_id.argtypes=[ctypes.c_void_p]
lib.modern_glm_model_create.argtypes=[ctypes.c_int]*2+[ctypes.c_size_t]*2+[ctypes.c_char_p]+[ctypes.c_void_p]*3;lib.modern_glm_model_create.restype=ctypes.c_void_p
lib.modern_glm_model_keys.argtypes=[ctypes.c_void_p];lib.modern_glm_model_keys.restype=ctypes.c_char_p
lib.modern_glm_model_load_batch.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.modern_glm_model_finish.argtypes=[ctypes.c_void_p]
lib.modern_glm_model_forward.argtypes=[ctypes.c_void_p]+[ctypes.c_size_t]*2+[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p]
lib.modern_glm_model_control.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
lib.modern_glm_model_reset.argtypes=[ctypes.c_void_p];lib.modern_glm_model_destroy.argtypes=[ctypes.c_void_p]

def check(code):
 if code:raise RuntimeError(lib.modern_glm_model_error().decode())
def unique():
 raw=ctypes.create_string_buffer(lib.modern_glm_model_id_size());check(lib.modern_glm_model_id(raw));return base64.b64encode(raw.raw).decode()
def worker(rank,payload):
 torch.cuda.set_device(rank);stream=lambda:ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
 cfg=json.loads((model/'config.json').read_text());buffers=[ctypes.create_string_buffer(base64.b64decode(payload[k]if k!='tp'else payload[k][stage]))for k in ['world','tp','control']]
 ctx=lib.modern_glm_model_create(rank,stage,8,4096,json.dumps(cfg).encode(),*buffers)
 if not ctx:raise RuntimeError(lib.modern_glm_model_error().decode())
 report={'scope':'Actual78-layer GLM checkpoint execution, native TextModel direct CAPI with explicit TP8 PP4 and original FP8 bytes/BF16 nonquantized weights. Diagnostic greedy fixed16-token short-input repeated B1/B4; ordinary ModelFactory/Engine, independent model quality and maximum TPS remain unqualified.','step':tag,'stage':stage,'rank':rank,'cases':[],'pipeline_boundaries':[0,22,42,62,78]}
 sdk=sorted({line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfiniops.so'in line});assert len(sdk)==1 and'/install/hygon-runtime-fixes-r1/lib/'in sdk[0];report['sdk_sha256']=hashlib.sha256(pathlib.Path(sdk[0]).read_bytes()).hexdigest()
 keys=set(json.loads(lib.modern_glm_model_keys(ctx)));mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert keys<=set(mapping),sorted(keys-set(mapping))[:5]
 grouped=collections.defaultdict(list)
 for key in keys:grouped[mapping[key]].append(key)
 pending=[];loaded=0;source_bytes=0;begin=time.perf_counter()
 def flush():
  nonlocal loaded
  if not pending:return
  desc=[{'name':name,'pointer':w.data_ptr(),'shape':list(w.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]}for name,w in pending];check(lib.modern_glm_model_load_batch(ctx,json.dumps(desc).encode()));loaded+=len(pending);pending.clear();print('GLM78_LOAD_PROGRESS',stage,rank,loaded,len(keys),round(time.perf_counter()-begin,3),flush=True)
 for filename,names in sorted(grouped.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as shard:
   for name in sorted(names):
    w=shard.get_tensor(name).contiguous()
    if w.dtype==torch.float8_e4m3fn:w=w.view(torch.uint8)
    assert w.dtype in [torch.float32,torch.bfloat16,torch.uint8],(name,w.dtype);source_bytes+=w.numel()*w.element_size();pending.append((name,w))
    if len(pending)>=2048:flush()
   flush()
 check(lib.modern_glm_model_finish(ctx));report.update(loaded_parameters=loaded,source_parameter_bytes=source_bytes,loading_seconds=time.perf_counter()-begin,free_total_gpu_bytes_after_loading=torch.cuda.mem_get_info());print('GLM78_WEIGHTS_READY',stage,rank,report['loading_seconds'],report['free_total_gpu_bytes_after_loading'],flush=True)
 def control(x):check(lib.modern_glm_model_control(ctx,x.numel(),ptr(x),stream()))
 barrier=torch.ones(1,device='cuda');control(barrier);assert barrier.item()==32
 vocab=cfg['vocab_size']
 for batch in [1,4]:
  for repeat in [1,2]:
   phase=f'short_b{batch}_repeat{repeat}';prompt=payload['prompt'];length=len(prompt);check(lib.modern_glm_model_reset(ctx));barrier.fill_(1);control(barrier);generated=[[]for _ in range(batch)];steps=[];hashes=[];start=time.perf_counter()
   for step in range(16):
    t0=time.perf_counter();current=[prompt]*batch if step==0 else[[v[-1]]for v in generated];width=len(current[0]);past=0 if step==0 else length+step-1;ids=torch.tensor(sum(current,[]),device='cuda',dtype=torch.int64);pos=torch.arange(past,past+width,device='cuda',dtype=torch.int64).repeat(batch);requests=[(i+1,past,width)for i in range(batch)];desc=(ctypes.c_size_t*(3*batch))(*[v for q in requests for v in q]);logits=torch.full((batch,vocab),float('nan'),device='cuda',dtype=torch.bfloat16)
    f0=time.perf_counter();check(lib.modern_glm_model_forward(ctx,len(ids),batch,desc,ptr(ids),ptr(pos),ptr(logits),0,stream()));forward=time.perf_counter()-f0
    if stage==3:
     assert torch.isfinite(logits).all()
     if step<4:
      raw=logits.cpu().contiguous().view(torch.uint8).numpy().tobytes();hashes.append(hashlib.sha256(raw).hexdigest())
      if rank==0:(r/'results'/f'glm78-{tag}-{phase}-step{step}-logits-bf16.bin').write_bytes(raw)
    emitted=logits.argmax(-1).float()if stage==3 and rank==0 else torch.zeros(batch,device='cuda');control(emitted);next_ids=[int(x)for x in emitted.cpu().tolist()];assert all(0<=v<vocab for v in next_ids)
    for i,v in enumerate(next_ids):generated[i].append(v)
    row={'step':step,'forward_seconds':forward,'wall_seconds':time.perf_counter()-t0};steps.append(row)
    print('GLM78_STEP',stage,rank,phase,step,round(row['wall_seconds'],6),next_ids,flush=True)
   report['cases'].append({'phase':phase,'batch':batch,'input_tokens_per_request':length,'output_tokens_per_request':16,'generated_ids':generated,'steps':steps,'elapsed_seconds':time.perf_counter()-start,'first4_logits_sha256':hashes,'free_total_gpu_bytes_after':torch.cuda.mem_get_info()})
   (r/'results'/f'modern-glm78-real-{tag}-stage{stage}-rank{rank}.json').write_text(json.dumps(report,indent=2))
 pairs=[]
 for i in [0,2]:
  a,b=report['cases'][i:i+2];pairs.append({'batch':a['batch'],'all_16_generated_ids_equal':a['generated_ids']==b['generated_ids'],'first4_logits_bits_exact':a['first4_logits_sha256']==b['first4_logits_sha256']if stage==3 else None})
 report['repeat_pairs']=pairs;check(lib.modern_glm_model_destroy(ctx));check(lib.modern_glm_model_shutdown());report.update(completed=True,worker_shutdown_completed=True,all_repeats_passed=all(v['all_16_generated_ids_equal']and v['first4_logits_bits_exact']is not False for v in pairs),complete_model_quality_qualified=False,maximum_model_tps_established=False)
 (r/'results'/f'modern-glm78-real-{tag}-stage{stage}-rank{rank}.json').write_text(json.dumps(report,indent=2))
def guarded(rank,payload):
 try:worker(rank,payload)
 except BaseException:traceback.print_exc();os._exit(2)
if __name__=='__main__':
 directory=r/'control';directory.mkdir(exist_ok=True);p=directory/f'glm78-{tag}-ids.json'
 if stage==0:
  tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True);prompt=tokenizer.apply_chat_template([{'role':'user','content':'用一句话介绍北京。'}],tokenize=True,add_generation_prompt=True)
  if isinstance(prompt,Mapping):prompt=prompt['input_ids']
  if hasattr(prompt,'tolist'):prompt=prompt.tolist()
  if len(prompt)==1 and isinstance(prompt[0],(list,tuple)):prompt=prompt[0]
  prompt=[int(v)for v in prompt]
  assert 1<=len(prompt)<=512 and all(0<=v<154880 for v in prompt)
  payload={'world':unique(),'tp':[unique()for _ in range(4)],'control':unique(),'prompt':prompt};tmp=p.with_suffix('.tmp');assert not p.exists()
  tmp.write_text(json.dumps(payload));tmp.replace(p)
 begin=time.monotonic()
 while not p.exists():
  if time.monotonic()-begin>300:raise TimeoutError('GLM control identities missing')
  time.sleep(.25)
 payload=json.loads(p.read_text());mp.spawn(guarded,args=(payload,),nprocs=8,join=True);print('GLM78_NODE_COMPLETE',stage,flush=True)
