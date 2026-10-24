"""Finite actual-checkpoint C128 component; synthetic hidden, not Model TPS."""
import ctypes, pathlib, json, hashlib, os, time, threading, concurrent.futures, traceback
import torch
from safetensors import safe_open

ROOT=pathlib.Path('/refactor')
MODEL=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8')
LIB=ROOT/'candidates/deep-C128-component-r8694/libdeep_C128_component_r8694.so'
LIB_SHA='b4c0fcff7657784c50ef030452f2cb2b2f71ffb0e8eb010a839ffde6b7fa0708'
FLAG='INFINILM_DEEP_C128_F32_WEIGHT_CACHE_R8681'
ARMS=['O','N']+['O','N','N','O']*3
ROWS=32; RING=128; DIM=512; CAPACITY=8
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
assert torch.cuda.device_count()==8
torch.cuda.set_device(0)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(LIB)==LIB_SHA
receipt=ROOT/'candidates/deep-C128-component-r8694/CPU-build.json'
assert sha(receipt)=='03414aabcbd7b9a7b7148cf07a73d12fdd8842f7460a5e90e82565623a18223d'
build=json.loads(receipt.read_text());assert build['completed']and build['CPU_only']and not build['GPU']and not build['dlopen']
for p,h in build['link_objects'].items():assert sha(p)==h
cfg=json.loads((MODEL/'config.json').read_text());layer=next(i for i,x in enumerate(cfg['compress_ratios'])if x==128)
index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
weights={};weight_receipts={}
for name in ['wkv.weight','wgate.weight','ape','norm.weight']:
 key=f'layers.{layer}.attn.compressor.{name}'
 with safe_open(MODEL/index[key],framework='pt',device='cpu')as f:w=f.get_tensor(key).contiguous()
 assert w.dtype in [torch.bfloat16,torch.float32]
 weights[name]=w;weight_receipts[name]={'key':key,'shape':list(w.shape),'dtype':str(w.dtype),'bytes':w.numel()*w.element_size()}
assert weights['wkv.weight'].shape==weights['wgate.weight'].shape==(512,4096)
lib=ctypes.CDLL(str(LIB));P=ctypes.c_void_p;S=ctypes.c_size_t
lib.modern_real_compress_error.restype=ctypes.c_char_p
lib.modern_real_compress_create.argtypes=[ctypes.c_int]+[S]*4+[ctypes.c_char_p];lib.modern_real_compress_create.restype=P
lib.modern_real_compress_keys.argtypes=[P];lib.modern_real_compress_keys.restype=ctypes.c_char_p
lib.modern_real_compress_load.argtypes=[P,ctypes.c_char_p,P,S,P,ctypes.c_int]
lib.modern_real_compress_finish.argtypes=[P]
lib.modern_real_compress_forward.argtypes=[P,S,S]+[P]*6+[ctypes.c_int,P]
lib.modern_real_compress_bytes.argtypes=[P,ctypes.c_int];lib.modern_real_compress_bytes.restype=S
lib.modern_real_compress_snapshot.argtypes=[P,ctypes.c_int,P,P]
lib.modern_real_compress_gather.argtypes=[P,S,P,P,P]
lib.modern_real_compress_reset.argtypes=[P];lib.modern_real_compress_destroy.argtypes=[P]
lib.modern_real_compress_release_workers.argtypes=[]
lib.modern_real_compress_timing_r8694.argtypes=[ctypes.POINTER(ctypes.c_double),S];lib.modern_real_compress_timing_r8694.restype=ctypes.c_int
ptr=lambda x:P(x.data_ptr())
def check(v):
 if v:raise RuntimeError(lib.modern_real_compress_error().decode())
barrier=threading.Barrier(8,timeout=180)
def rendezvous():barrier.wait()
def emit(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
run=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
target=ROOT/'results'/f'Deep-C128-component-r8695-{run}.json'
assert not target.exists()
report={'scope':'Actual checkpoint C128 compressor, synthetic BF16 hidden, 8 independent replicated device owners; not full Model or final token qualification. Component CAPI includes original output copies/drain. Rank samples are not independent repetitions.',
 'actual_handle':run,'library_SHA256':LIB_SHA,'CPU_receipt_SHA256':sha(receipt),'layer':layer,'C128_layer_count':sum(x==128 for x in cfg['compress_ratios']),
 'geometry':{'rows':ROWS,'ring':RING,'dim':DIM,'compressed_capacity_per_row':CAPACITY,'history_tokens':640,'seed_tokens_per_row':575,'selected_past':[575,639],'arms':ARMS,'warmup_trials':[0,1],'ordinary_trials':list(range(2,14))},
 'extra_F32_cached_weight_bytes_per_C128_per_GPU':2*512*4096*4,'weights':weight_receipts,'owners':[],'all8normal':False,'final_tokens':'NOT_MEASURED','Model_gain':'UNKNOWN'}
emit(target,report)

class Owner:
 def __init__(self,device):
  self.device=device;self.contexts={};self.history=None;self.gold={};self.records=[];self.load=[];self.minimum_free=None;self.closed=False;self.native_slot=(ctypes.c_double*3)();self.native_sequence=0
 def activate(self):torch.cuda.set_device(self.device)
 def stream(self):return P(torch.cuda.current_stream(self.device).cuda_stream)
 def memory(self,admit=False):
  free,total=torch.cuda.mem_get_info(self.device);self.minimum_free=free if self.minimum_free is None else min(free,self.minimum_free)
  assert free>=(6*1024**3 if admit else 512*1024**2),(self.device,free,'headroom')
  return free
 def create(self,mode):
  self.activate();self.memory(True);assert os.environ.get(FLAG)==str(mode)
  start=time.monotonic();c=lib.modern_real_compress_create(self.device,layer,DIM,ROWS,CAPACITY,json.dumps(cfg).encode())
  if not c:raise RuntimeError(lib.modern_real_compress_error().decode())
  self.contexts[mode]=c
  assert set(json.loads(lib.modern_real_compress_keys(c)))==set(weights)
  for n,w in weights.items():
   shape=(S*w.ndim)(*w.shape);check(lib.modern_real_compress_load(c,n.encode(),ptr(w),w.ndim,shape,int(w.dtype==torch.float32)))
  check(lib.modern_real_compress_finish(c));self.load.append({'mode':mode,'seconds':time.monotonic()-start,'free_after':self.memory(True)})
 def forward(self,c,x,positions,requests):
  desc=(S*(len(requests)*3))(*[v for q in requests for v in q]);n=len(x)
  score=torch.empty((n,2*DIM),dtype=torch.bfloat16,device=self.device);pooled=torch.empty((n,DIM),dtype=torch.bfloat16,device=self.device);rotated=torch.empty_like(pooled)
  # Torch producer work is excluded from native-call wall for both arms.
  torch.cuda.current_stream(self.device).synchronize()
  t=time.monotonic();cpu=time.thread_time()
  check(lib.modern_real_compress_forward(c,n,len(requests),desc,ptr(x),ptr(positions),ptr(score),ptr(pooled),ptr(rotated),0,self.stream()))
  wall=time.monotonic()-t;thread_cpu=time.thread_time()-cpu
  check(lib.modern_real_compress_timing_r8694(self.native_slot,3))
  native,sequence,native_tokens=list(self.native_slot);assert sequence==self.native_sequence+1 and native_tokens==n and 0<native<=wall
  self.native_sequence=int(sequence)
  return [score,pooled,rotated],wall,thread_cpu,native
 def snapshot(self,c,cache):
  n=lib.modern_real_compress_bytes(c,int(cache));x=torch.empty(n,dtype=torch.uint8,device=self.device)
  check(lib.modern_real_compress_snapshot(c,int(cache),ptr(x),self.stream()));return x.cpu()
 def validation(self,c,outputs,past,trial):
  # No readback occurs inside the timed native call. Cache raw-byte equality is diagnostic.
  entries=[(row,i)for row in range(ROWS)for i in range((past+1)//128)]
  desc=(S*(len(entries)*2))(*[v for q in entries for v in q]);g=torch.empty((len(entries),512),dtype=torch.bfloat16,device=self.device)
  check(lib.modern_real_compress_gather(c,len(entries),desc,ptr(g),self.stream()))
  data=[x.cpu()for x in outputs]+[self.snapshot(c,False).view(torch.bfloat16),g.cpu()]
  raw=self.snapshot(c,True)
  assert all(torch.isfinite(x).all().item()for x in data)
  if past not in self.gold:
   assert trial==0;self.gold[past]=(data,raw);return {'finite':True,'baseline_frozen':True}
  old,oldraw=self.gold[past];metrics=[]
  for a,b in zip(data,old):
   af=a.float();bf=b.float();delta=af-bf;nr=float(delta.square().mean().sqrt()/bf.square().mean().sqrt().clamp_min(1e-9));ma=float(delta.abs().max())
   assert nr<=.005,(self.device,past,nr,ma)
   metrics.append({'NRMSE':nr,'max_abs':ma,'byte_equal':torch.equal(a.view(torch.uint8),b.view(torch.uint8))})
  return {'finite':True,'components_score_pool_rotated_ring_gather':metrics,'raw_cache_byte_equal_diagnostic':torch.equal(raw,oldraw)}
 def run(self):
  self.activate()
  try:
   torch.manual_seed(868800+self.device)
   self.history=torch.randn((640,ROWS,4096),dtype=torch.bfloat16,device=self.device)*.1
   self.memory(True)
   warmup=time.monotonic();warmup_seconds=None
   for trial,arm in enumerate(ARMS):
    c=self.contexts[int(arm=='N')];check(lib.modern_real_compress_reset(c));seed=time.monotonic()
    for begin in range(0,ROWS,7):
     rows=list(range(begin,min(begin+7,ROWS)));x=self.history[:575,rows].permute(1,0,2).contiguous().reshape(-1,4096)
     pos=torch.arange(575,dtype=torch.int64,device=self.device).repeat(len(rows))
     self.forward(c,x,pos,[(row,0,575)for row in rows])
    seed_seconds=time.monotonic()-seed
    for past in range(575,640):
     x=self.history[past].contiguous();pos=torch.full((ROWS,),past,dtype=torch.int64,device=self.device)
     rendezvous();out,wall,cpu,native=self.forward(c,x,pos,[(row,past,1)for row in range(ROWS)])
     if past in [575,639]:
      v=self.validation(c,out,past,trial);self.records.append({'trial':trial,'arm':arm,'warmup':trial<2,'past':past,'next_past':past+1,'C128_boundary':(past+1)%128==0,'native_enclosing_wall_s':wall,'owner_thread_CPU_s':cpu,'native_inner_wall_s':native,'outer_minus_inner_s':wall-native,'validation':v,'seed_seconds':seed_seconds,'free_after':self.memory()})
     rendezvous()
    if trial==1:warmup_seconds=time.monotonic()-warmup
   return {'device':self.device,'normal':True,'load':self.load,'warmup_seconds':warmup_seconds,'minimum_sampled_free':self.minimum_free,'records':self.records}
  except BaseException:
   barrier.abort();raise
 def close(self):
  self.activate();errors=[]
  for mode,c in list(self.contexts.items()):
   try:check(lib.modern_real_compress_destroy(c))
   except BaseException as e:errors.append(repr(e))
   finally:del self.contexts[mode]
  try:check(lib.modern_real_compress_release_workers())
  except BaseException as e:errors.append(repr(e))
  self.history=None;self.gold.clear();torch.cuda.synchronize(self.device);self.closed=not errors
  if errors:raise RuntimeError(errors)
  return {'device':self.device,'normal_close':True}

pools=[concurrent.futures.ThreadPoolExecutor(max_workers=1)for _ in range(8)]
owners=[Owner(i)for i in range(8)]
failure=None
try:
 for mode in [0,1]:
  os.environ[FLAG]=str(mode)
  futures=[p.submit(o.create,mode)for p,o in zip(pools,owners)]
  for f in futures:f.result(timeout=180)
 # No later getenv dependency: native freezes dtype/cache at each finish.
 futures=[p.submit(o.run)for p,o in zip(pools,owners)]
 for f in futures:report['owners'].append(f.result(timeout=450))
 report['all8normal']=len(report['owners'])==8
except BaseException as e:
 barrier.abort();failure=traceback.format_exc();report['error']=failure
finally:
 os.environ[FLAG]='0';close=[]
 for p,o in zip(pools,owners):
  try:close.append(p.submit(o.close).result(timeout=90))
  except BaseException:close.append({'device':o.device,'error':traceback.format_exc()})
 for p in pools:p.shutdown(wait=True)
 report['close']=close;report['all8normal_close']=all(x.get('normal_close',False)for x in close)
 report['completed']=failure is None and report['all8normal']and report['all8normal_close'];emit(target,report)
assert report['completed'],report.get('error',report['close'])
print(json.dumps({'result':str(target),'completed':True,'all8normal':True,'all8normal_close':True}),flush=True)
