"""Current8df one-layer TP16 MoE strategy; historical rows expanded to geometry512.

Import/build_payload/dispatch are CPU-safe. Only main.run opens weights or dlopens.
No full-model, real-prefill traffic, HTTP or service qualification is implied.
"""
from pathlib import Path,PurePosixPath
import ctypes,hashlib,json,os,time,gc
from plan import records,row_order
R=Path('/refactor');D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
NATIVE='/refactor/candidates/Kimi-append512-r247/libKimi_append512_r247.so'
NATIVE_SHA='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
def build_payload(config,weights,input_pointers,node,host,port,raw,deadline):
 if node not in [0,1]or type(node)is not int:raise ValueError('actualtwoTPnodes')
 if len(input_pointers)!=2 or any(type(p)is not int or p<=0 for p in input_pointers):raise ValueError('twoactual512matrices')
 if not isinstance(weights,list)or not weights or len({w['name']for w in weights})!=len(weights):raise ValueError('uniqueactualweights')
 for w in weights:
  if not isinstance(w.get('name'),str)or not w['name']or type(w.get('pointer'))is not int or w['pointer']<=0 or w.get('dtype')not in [0,1,2]or not isinstance(w.get('shape'),list)or not w['shape']or any(type(n)is not int or n<=0 for n in w['shape']):raise ValueError('actualweightdescriptor')
 if not PurePosixPath(raw).is_absolute()or not isinstance(host,str)or not host or not 1024<=port<65536:raise ValueError('absoluteoutputandactualrendezvous')
 if not isinstance(deadline,(float,int))or deadline<=time.time():raise ValueError('originaldeadline')
 return dict(schema=1,config=config,weights=weights,input_pointers=input_pointers,node_rank=node,master_addr=host,master_port=port,raw_directory=str(raw),deadline_unix=deadline,plan=list(records()))
def dispatch(payload,invoke,read_result):
 """Actual adapter glue; CPU receipt may record its CAPI leaf, never claim Native."""
 encoded=json.dumps(payload,separators=(',',':')).encode();rc=invoke(encoded);q=json.loads(read_result())
 if rc or not q.get('completed')or not q.get('all8normal_before_communicators'):raise RuntimeError('Original adapter failed; no reset/retry: '+str(q.get('error')))
 return q
def run(manifest,output):
 import numpy as np
 import torch
 from safetensors import safe_open
 torch.set_num_threads(1);assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 m=json.loads(Path(manifest).read_bytes());cpu=Path(m['CPU_receipt']);assert sha(cpu)==m['CPU_receipt_SHA256'];c=json.loads(cpu.read_bytes());assert c['completed']and c['passed']and not c['Native_dlopen']and c['HOST_TUs_compiled']==1
 assert sha(c['library'])==c['library_SHA256']==m['library_SHA256'];assert sha(NATIVE)==NATIVE_SHA==c['native_SHA256']
 out=Path(output);assert not out.exists();raw=out.with_suffix('');raw.mkdir();node=int(os.environ['SLURM_PROCID']);assert node in [0,1]
 q=dict(completed=False,scope=__doc__,node=node,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),CPU_receipt_SHA256=sha(cpu),library_SHA256=c['library_SHA256'],Native_SHA256=NATIVE_SHA,phase='beforeweights')
 def save():
  t=out.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(out)
 save()
 try:
  lib=ctypes.CDLL(c['library']);lib.kimi_moe512_micro64_r391.argtypes=[ctypes.c_char_p];lib.kimi_moe512_micro64_r391.restype=ctypes.c_int;lib.kimi_moe512_micro64_result_r391.restype=ctypes.c_char_p
  class Dl(ctypes.Structure):_fields_=[('path',ctypes.c_char_p),('base',ctypes.c_void_p),('name',ctypes.c_char_p),('addr',ctypes.c_void_p)]
  dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int;q['providers']={}
  for symbol,expected in [('modern_kimi_I192_counts_r8669y',NATIVE),('modern_kimi_I192_T_counts_r9129',NATIVE),('modern_kimi_grouped_I192_r8669y',NATIVE),('modern_kimi_set_grouped_prefill',NATIVE),('modern_grouped_mxfp4_owned','/refactor/candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so'),('modern_grouped_metadata_counts_r5858','/refactor/candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so')]:
   v=Dl();assert dl(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(v));p=Path(v.path.decode()).resolve();assert p==Path(expected).resolve();q['providers'][symbol]=dict(path=str(p),SHA256=sha(p))
  model=Path('/shared/Kimi-K3');config=json.loads((model/'config.json').read_bytes());index=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];prefix='language_model.model.layers.1.block_sparse_moe.';groups={}
  for key,fn in index.items():
   if key.startswith(prefix):groups.setdefault(fn,[]).append(key)
  assert groups;q.update(phase='loading_one_MoE_layer',config_SHA256=sha(model/'config.json'),index_SHA256=sha(model/'model.safetensors.index.json'));save();held=[];descriptors=[];tick=time.monotonic()
  for filename,keys in sorted(groups.items()):
   if time.time()>=m['deadline_unix']:raise TimeoutError('Originalresidencydeadline duringload')
   with safe_open(str(model/filename),framework='pt',device='cpu')as f:
    for key in sorted(keys):
     value=f.get_tensor(key).clone().contiguous();assert value.dtype in [torch.float32,torch.bfloat16,torch.uint8];held.append(value);descriptors.append(dict(name=key[len(prefix):],pointer=value.data_ptr(),shape=list(value.shape),dtype={torch.float32:0,torch.bfloat16:1,torch.uint8:2}[value.dtype]))
  q.update(weight_load_seconds=time.monotonic()-tick,CPU_weight_bytes=sum(v.nbytes for v in held),weight_count=len(held));fixture=Path(m['teacher_fixture']);assert sha(fixture)=='0d0f2385a5006dfe0ac1b6f9ad6539fe5793393aca4bc4125eaf135a0ae47e4a';source=np.frombuffer(fixture.read_bytes(),dtype=np.uint16).reshape(16,7168);inputs=[np.ascontiguousarray(source[row_order(i)])for i in range(2)]
  q['inputs']=[dict(fixture=i,shape=list(x.shape),SHA256=hashlib.sha256(x.tobytes()).hexdigest(),geometry_only=True)for i,x in enumerate(inputs)];payload=build_payload(config,descriptors,[int(x.ctypes.data)for x in inputs],node,m['master_addr'],m['master_port'],str(raw),m['deadline_unix']);q.update(phase='oneactualNativeadapter',planned_formal=96,planned_warm=6);save()
  q['native']=dispatch(payload,lib.kimi_moe512_micro64_r391,lib.kimi_moe512_micro64_result_r391);q['warm_raw_proof']=[dict(file=p.name,bytes=p.stat().st_size,SHA256=sha(p))for p in sorted(raw.glob('*-warm.bin'))];assert len(q['warm_raw_proof'])==48
  del held,descriptors,inputs,source;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();test=torch.ones(32,device='cuda');assert(test+1).sum().item()==64;torch.cuda.synchronize();q.update(completed=True,normal_CUDA_cleanup=True,Torch_after_Runtime_close=True,phase='completed');save()
 except BaseException as error:q['error']=repr(error);save();raise
if __name__=='__main__':
 import sys
 run(sys.argv[1],sys.argv[2])
