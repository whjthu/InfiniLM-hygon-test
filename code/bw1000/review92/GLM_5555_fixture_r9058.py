import pathlib,json,ctypes,os,time,hashlib,traceback
import torch,torch.multiprocessing as mp
r=pathlib.Path('/refactor');base=r/'candidates/glm-567-cohort-r9057';V=ctypes.c_void_p;Z=ctypes.c_size_t;I=ctypes.c_int
DSO_SHA='fbd07490dd9887868e4a6f5303170969dfa188e9cbcb39f7dca708b3a303741a'
def api():
 lib=ctypes.CDLL(str(base/'libglm_567_cohort_r9057.so'))
 lib.glm_layer_id_size_r5670.restype=Z;lib.glm_layer_id_r5670.argtypes=[V]
 lib.glm_layer_create_r5670.argtypes=[I,ctypes.c_char_p,V];lib.glm_layer_create_r5670.restype=V
 lib.glm_layer_destroy_r5670.argtypes=[V,ctypes.POINTER(I)]
 lib.glm_layer_error_r5670.restype=ctypes.c_char_p
 lib.glm_layer_report_r5670.restype=ctypes.c_char_p
 lib.glm_attention_fixture_r9057.argtypes=[V,Z,Z,V,V,V,V]
 lib.glm_attention_cohort_policy_r9054.argtypes=[I,I]
 lib.glm_attention_cohort_counts_r9054.argtypes=[I,ctypes.POINTER(ctypes.c_uint64),Z]
 return lib

def worker(rank,uid):
 torch.set_num_threads(1);torch.cuda.set_device(rank);cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=64;os.sched_setaffinity(0,cpus[rank*8:rank*8+8])
 p=r/'results'/f'GLM-5555-fixture-r9058-{os.environ["SLURM_JOB_ID"]}.{os.environ["SLURM_STEP_ID"]}-rank{rank}.json';assert not p.exists()
 q={'completed':False,'rank':rank,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'library_SHA256':DSO_SHA,'model_weights_loaded':False,'entry':5555,'B':16,'P':513,'trials':[]};ctx=None;lib=api()
 def save():t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
 def ck(c):
  if c:raise RuntimeError((c,lib.glm_layer_error_r5670().decode()))
 def counts():
  a=(ctypes.c_uint64*4)();ck(lib.glm_attention_cohort_counts_r9054(rank,a,4));return list(a)
 try:
  proof=json.loads((base/'CPU-build.json').read_bytes());assert proof['completed']and proof['library_SHA256']==DSO_SHA==hashlib.sha256((base/'libglm_567_cohort_r9057.so').read_bytes()).hexdigest();assert proof['parent567_SHA256']=='68bdbddc918063dee48b4da21bb3ca24a55abdc4a4a3c0d7c415e3e167de100e'
  cfg=json.loads((pathlib.Path('/old/models/ZhipuAI-GLM-5.3')/'config.json').read_bytes());cfg['probe_layers_r5680']=[38];u=ctypes.create_string_buffer(uid,len(uid));ctx=lib.glm_layer_create_r5670(rank,json.dumps(cfg).encode(),u);assert ctx,lib.glm_layer_error_r5670()
  for seed in [9058,9059]:
   g=torch.Generator().manual_seed(seed)
   tensors=[torch.randn(shape,generator=g,dtype=torch.float32).to(torch.bfloat16).cuda()for shape in [(16,8,256),(16,513,8,256),(16,513,8,256)]];snap=[x.clone()for x in tensors];guard=torch.full((16*8*256+128,),123,dtype=torch.bfloat16,device='cuda');out=guard[64:-64].view(16,8,256);reference=None
   for mode in [0,1,1,0,0]:
    ck(lib.glm_attention_cohort_policy_r9054(rank,mode));before=counts();torch.cuda.synchronize();t=time.perf_counter();ck(lib.glm_attention_fixture_r9057(ctx,16,513,*[V(x.data_ptr())for x in tensors],V(out.data_ptr())));wall=time.perf_counter()-t;after=counts();delta=[a-b for a,b in zip(after,before)];assert delta==([0,1,0,0]if mode else[1,0,0,0]),('actual5555path',delta);actual=out.cpu()
    if reference is None:reference=actual.clone()
    d=actual.float()-reference.float();finite=bool(torch.isfinite(actual.float()).all());exact=torch.equal(actual.view(torch.int16),reference.view(torch.int16));rr=float(d.square().mean().sqrt())/max(float(reference.float().square().mean().sqrt()),1e-30);readonly=all(torch.equal(a.view(torch.uint8),b.view(torch.uint8))for a,b in zip(tensors,snap));canary=bool((guard[:64]==123).all()and(guard[-64:]==123).all());assert finite and readonly and canary and rr<=.005 and float(d.abs().max())<=.05*max(1.,float(reference.float().abs().max()))
    q['trials'].append({'seed':seed,'mode':mode,'delta':delta,'finite':finite,'readonly':readonly,'canary':canary,'bits_exact':exact,'relRMS':rr,'maxabs':float(d.abs().max()),'CAPI_wall_s_not_perf_qualified':wall,'report':json.loads(lib.glm_layer_report_r5670())});save()
  ck(lib.glm_attention_cohort_policy_r9054(rank,0));expired=I();ck(lib.glm_layer_destroy_r5670(ctx,ctypes.byref(expired)));ctx=None;assert expired.value==1;torch.cuda.synchronize();assert (torch.ones(32,device='cuda')+1).sum().item()==64;q.update(completed=True,normal_reset_close=True,Runtime_weak_expired=True,Torch_postclose=True,all_outputs_finite=True,actual5555_ON_calls=4);save()
 except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc();save();raise
 finally:
  if ctx:
   e=I();q['finally_destroy_code']=lib.glm_layer_destroy_r5670(ctx,ctypes.byref(e));q['finally_Runtime_expired']=bool(e.value);save()

if __name__=='__main__':
 assert torch.cuda.device_count()==8;torch.cuda.set_device(0);lib=api();u=ctypes.create_string_buffer(lib.glm_layer_id_size_r5670());assert lib.glm_layer_id_r5670(u)==0;mp.spawn(worker,args=(bytes(u),),nprocs=8,join=True)
