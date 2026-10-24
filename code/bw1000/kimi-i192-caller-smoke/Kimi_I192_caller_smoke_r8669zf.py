import os,re,pathlib,json,ctypes,hashlib,datetime
# The native smoke deliberately uses virtual device1 and currentdevice0.
allocation=os.environ.get('SLURM_STEP_GPUS','')
assert re.fullmatch(r'[0-7],[0-7]',allocation) and len(set(allocation.split(',')))==2,('twoallocatedGPUidsrequired',allocation)
os.environ['CUDA_VISIBLE_DEVICES']=allocation
import torch
torch.set_num_threads(1);assert torch.cuda.device_count()==2;torch.cuda.set_device(0)
r=pathlib.Path('/refactor');label='Kimi-I192-caller-smoke-r8669zf';m=json.loads((r/'incoming/Kimi-I192-caller-smoke-manifest-r8669zf.json').read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();receipt=r/m['CPU_receipt'];assert sha(receipt)==m['CPU_receipt_SHA256'];b=json.loads(receipt.read_text());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen'];assert sha(b['native'])==b['native_SHA256']==m['native_DSO_SHA256']
lib=ctypes.CDLL(b['native']);f=lib.modern_kimi_I192_caller_smoke_r8669zf;f.argtypes=[];f.restype=ctypes.c_char_p
class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int;provider=Dl();assert dl(ctypes.cast(f,ctypes.c_void_p),ctypes.byref(provider));assert pathlib.Path(provider.fname.decode()).resolve()==pathlib.Path(b['native']).resolve()
out={'completed':False,'scope':'SYNTHETIC_ACTUAL_SHARED_CALLER_CONTRACT; noactivationreplay/TPcommunication/Model/performance','allocation':allocation,'native_SHA256':b['native_SHA256'],'CPU_receipt_SHA256':sha(receipt),'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
 out['native']=json.loads(f());q=out['native'];assert q['completed']and q['normal_runtime_close']and q['finite']and q['nonzero_outputs']>0 and q['full_output_bit_differences']==0 and q['output_elements']==57344 and q['current_after_helper']==0 and q['current_after_copy']==0 and q['canary_intact']and q['shared_helper_success_delta']==1 and q['compact_workspace_bytes']==4468480
 out['completed']=True;out['normal_CUDA_cleanup']=True
except BaseException as e:out['error']=repr(e);raise
finally:
 out['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=r/'results'/(label+'-'+out['step']+'.json');assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
