import pathlib,json,torch,ctypes,hashlib,tempfile,os,socket,time
from safetensors import safe_open
from safetensors.torch import save_file
import parameter_loader_original as old
import parameter_loader_profile_r9204 as new
assert not torch.cuda.is_initialized()
class Recorder:
 def __init__(self):self.calls=[];self.finishes=0
 def modern_kimi_engine_load_r2475(self,ctx,payload):
  packet=json.loads(payload);row=[]
  for e in packet:
   size={0:4,1:2,2:1}[e['dtype']]
   for x in e['shape']:size*=x
   row.append({'name':e['name'],'shape':e['shape'],'dtype':e['dtype'],'bytes_SHA256':hashlib.sha256(ctypes.string_at(e['pointer'],size)).hexdigest()})
  self.calls.append(row);return 0
 def modern_kimi_engine_finish_r2475(self,ctx):self.finishes+=1;return 0
def check(rc):assert rc==0
with tempfile.TemporaryDirectory(prefix='loader-phase-CPU-')as tmp:
 p=pathlib.Path(tmp);first={f'language_model.layers.0.fixture_{i:05d}':torch.tensor([i],dtype=torch.float32)for i in range(16385)}
 second={'language_model.layers.0.self_attn.A_log':torch.arange(24,dtype=torch.float32).reshape(6,4),'z_bf16':torch.arange(12,dtype=torch.bfloat16).reshape(3,4),'z_uint8':torch.arange(17,dtype=torch.uint8)}
 save_file(first,str(p/'a.safetensors'));save_file(second,str(p/'b.safetensors'));mapping={n:f for f,ts in [('a.safetensors',first),('b.safetensors',second)]for n in ts};(p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':mapping}))
 results=[];recorders=[]
 for module in [old,new]:
  rec=Recorder();out=module.load_parameters(rec,1,p,{'linear_attn_config':{'num_heads':4}},set(mapping),safe_open,torch,check,lambda v:None);results.append(out);recorders.append(rec)
 assert recorders[0].calls==recorders[1].calls and all(x.finishes==1 for x in recorders)
 assert [len(x)for x in recorders[1].calls]==[16384,1,3]
 rows=[x for row in recorders[1].calls for x in row];a=next(x for x in rows if x['name'].endswith('.A_log'));assert a['shape']==[4,4]
 assert results[0]['loaded_keys']==results[1]['loaded_keys']==16388 and results[0]['loader_flushes']==results[1]['loader_flushes']==3
 phases=results[1]['host_loader_phase_profile'];assert phases['native_load_inclusive']['calls']==3 and phases['descriptor_json']['calls']==3 and phases['safe_open_enter']['calls']==2 and phases['get_tensor_inclusive']['calls']==16388
 assert all(v['seconds']>=0 for v in phases.values())
 assert sum(v['seconds']for v in phases.values())<=results[1]['loading_seconds']+0.001
 noalloc=Recorder()
 try:new.load_parameters(noalloc,1,p,{'linear_attn_config':{'num_heads':4}},set(mapping)|{'not_in_checkpoint'},safe_open,torch,check,lambda v:None);raise RuntimeError('missing key not rejected')
 except AssertionError:pass
 assert not noalloc.calls and noalloc.finishes==0
 assert not torch.cuda.is_initialized()
 q={'CPU_only':True,'GPU_submitted':False,'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'Torch_GPU_uninitialized':True,'passed':True,'actual_safetensors_and_CPU_tensors':True,'captured_native_payload_buffers_equal':True,'dtypes':['F32','BF16','U8'],'checked_keys':16388,'flush_boundaries':[16384,1,3],'A_log_real_trim':[6,4,4,4],'missing_key_fail_before_native':True,'phases':phases,'scope':'NativeGPU CAPI mocked only to read actualCPU payload bytes; no head giantbuffer or 2GiB threshold/GPUtiming qualification','source_SHA256':{n:hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()for n in ['parameter_loader_original.py','parameter_loader_profile_r9204.py','shard_head_contract_r9183.py']}}
 pathlib.Path('/refactor/results/loader-phase-CPU-r9204.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
