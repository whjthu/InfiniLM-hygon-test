"""One node's actual constructor -> embedding-only Native consumer -> destroy.

Metadata modes import neither Torch nor Native/SDK. No full-Model forward.
"""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,os,socket,struct,sys,time,traceback
from constructor import check_metadata,resident_constructor
import embedding
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkpoint_header(q):
 root=Path(q['model_directory']);pin=q['checkpoint_embedding'];p=root/pin['filename']
 with p.open('rb')as f:
  n=struct.unpack('<Q',f.read(8))[0];assert n==pin['header_bytes'];raw=f.read(n)
 assert hashlib.sha256(raw).hexdigest()==pin['header_SHA256']
 actual=json.loads(raw)[embedding.KEY];assert actual==pin['tensor']
 st=p.stat();assert st.st_size==pin['file_bytes']and st.st_mtime_ns==pin['file_mtime_ns']
 return dict(filename=pin['filename'],header_SHA256=pin['header_SHA256'],tensor=actual,file_bytes=st.st_size,file_mtime_ns=st.st_mtime_ns)
def metadata(node=None,hostname=None,require_gpu_entry_mask=True):
 q=json.loads((D/'config.json').read_bytes());n=int(os.environ['SLURM_PROCID'])if node is None else node
 h=socket.gethostname().split('.')[0]if hostname is None else hostname
 assert n in range(4)and h==q['nodes'][n]
 if require_gpu_entry_mask:assert len(os.sched_getaffinity(0))>=32,'Original entry>=32; no early pin'
 for key,value in q['environment'].items():assert os.environ.get(key)==value,(key,os.environ.get(key))
 root=Path(q['model_directory']);assert sha(root/'config.json')==q['config_SHA256']and sha(root/'model.safetensors.index.json')==q['index_SHA256']
 cfg=json.loads((root/'config.json').read_bytes());wm=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map'];check_metadata(cfg,wm)
 assert wm[embedding.KEY]==q['checkpoint_embedding']['filename'];checkpoint_header(q)
 assert [len(v)for v in embedding.queries()]==q['lookup_widths']==[1,1,64,512]
 return q,n,cfg,wm
def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('--metadata-only',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--metadata-all-recording-hosts',action='store_true');args=parser.parse_args(argv)
 assert sum([args.metadata_only,args.execute,args.metadata_all_recording_hosts])==1
 if args.metadata_all_recording_hosts:
  q=json.loads((D/'config.json').read_bytes());roles=[]
  for n,h in enumerate(q['nodes']):metadata(n,h,False);roles.append(dict(node=n,hostname_leaf='recording:'+h))
  print(json.dumps(dict(passed=True,roles=roles,Native_dlopen=False,GPU=0,real_affinity_count=len(os.sched_getaffinity(0)),GPU_entry_mask_qualified=False,
   fixed_query_ID_SHA256=[embedding.ID_SHA(v)for v in embedding.queries()],checkpoint_header=checkpoint_header(q))));return
 q,n,cfg,wm=metadata()
 if args.metadata_only:print(json.dumps(dict(passed=True,node=n,Native_dlopen=False,GPU=0)));return
 epoch=float(os.environ['K461_EPOCH']);deadline=epoch+q['Native_seconds'];assert time.time()<deadline
 assert sys.byteorder=='little'
 out=Path(q['runtime_directory']);out.mkdir(exist_ok=True);report=out/('node%d.json'%n);assert not report.exists()
 result=dict(node=n,hostname=q['nodes'][n],run=q['run'],source_manifest_SHA256=os.environ['K461_SOURCE_SHA'],epoch=epoch,Native_deadline=deadline,
  actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],PID=os.getpid(),UID=os.getuid(),PGID=os.getpgrp(),entry_mask=sorted(os.sched_getaffinity(0)),
  completed=False,passed=False,weight_keys=0,fullModel_forward=0,HTTP=0,reset=0,create_calls=0,destroy_calls=0,events=[])
 def save():
  t=report.with_suffix('.tmp');t.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');t.replace(report)
 def store(name,raw):
  path=out/('node%d-'%n+name)
  with path.open('xb')as f:f.write(raw)
  return dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),dtype='BF16',endianness='little',scope='raw CPU reference/output; no activation export to Git')
 def record(event,payload):
  # Actual full key list is validated by the unchanged consumer but stored only
  # once, avoiding repeated 40MB metadata in every memory/query event.
  if event=='constructor_observation'and'keys'in payload:
   count=len(payload['keys']);keys=json.dumps(payload['keys'],separators=(',',':')).encode();keyrecord=store('constructor-keys.json',keys);keyrecord.pop('dtype');keyrecord.pop('endianness')
   keyrecord['scope']='Actual constructor key JSON; no tensor payload'
   payload={k:v for k,v in payload.items()if k!='keys'};payload['keys_manifest']=dict(**keyrecord,count=count)
  result['events'].append(dict(event=event,payload=payload,unix=time.time()))
  if event=='before_create':result['create_calls']+=1
  if event=='destroy_returned':result['destroy_calls']+=1
  if event.startswith('memory_'):result[event]=payload
  save()
 save()
 try:
  import torch
  import numpy as np
  from safetensors import safe_open
  assert torch.cuda.device_count()==8
  cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int];cuda.cudaDeviceGetPCIBusId.restype=C.c_int
  hardware=[]
  for d in range(8):
   prop=torch.cuda.get_device_properties(d);bus=C.create_string_buffer(64)
   assert 'gfx936'in prop.gcnArchName and cuda.cudaDeviceGetPCIBusId(bus,64,d)==0
   hardware.append(dict(device=d,PCI=bus.value.decode().lower(),arch=prop.gcnArchName))
  assert len({v['PCI']for v in hardware})==8 and {v['PCI']for v in hardware}=={p.resolve().name for p in Path('/sys/class/drm').glob('renderD*/device')}
  result['hardware']=hardware;torch.cuda.set_device(0)
  memory=lambda:[list(torch.cuda.mem_get_info(d))for d in range(8)]
  record('memory_before_create',memory());assert all(t==q['physical_VRAM_bytes']and f>=q['floor_bytes']for f,t in result['memory_before_create'])
  assert sha(q['native'])==q['Native_SHA256'];lib=C.CDLL(q['native'],mode=os.RTLD_NOW|os.RTLD_LOCAL)
  class Info(C.Structure):_fields_=[('file',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
  dl=C.CDLL(None);dl.dladdr.argtypes=[C.c_void_p,C.POINTER(Info)];dl.dladdr.restype=C.c_int;providers={}
  for name in ['modern_kimi_engine_create_vocab_parallel_r461','modern_kimi_embedding_probe_load_r461','modern_kimi_embedding_probe_r461','modern_kimi_engine_destroy_r2475']:
   info=Info();assert dl.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(info))
   path=Path(info.file.decode()).resolve();assert path==Path(q['native']).resolve()and sha(path)==q['Native_SHA256'];providers[name]=dict(path=str(path),SHA256=sha(path),address=int(info.address))
  maps=Path('/proc/self/maps').read_text();result.update(dladdr=providers,Native_maps=maps)
  for basename,pin in q['dependency_libraries'].items():
   found={line.split()[-1]for line in maps.splitlines()if line.split()and line.split()[-1].split('/')[-1]==basename}
   assert len(found)==1 and all(sha(v)==pin['SHA256']for v in found),(basename,found)
  save()
  with resident_constructor(lib,cfg,wm,node=n,address=q['master_address'],port=q['native_port'],record=record)as(ctx,_):
   try:
    record('memory_after_create',memory());assert all(t==q['physical_VRAM_bytes']and f>=q['floor_bytes']for f,t in result['memory_after_create']),'Original 4GiB constructor floor'
    assert time.time()<deadline;header=checkpoint_header(q);source_started=time.perf_counter()
    with safe_open(Path(q['model_directory'])/header['filename'],framework='pt',device='cpu')as shard:
     tensor=shard.get_tensor(embedding.KEY) # ONLY this checkpoint key; no full-model loader/finish
     assert tensor.device.type=='cpu'and tensor.dtype==torch.bfloat16 and list(tensor.shape)==[163840,7168]and tensor.is_contiguous()
     words=tensor.view(torch.int16).numpy().view(np.uint16)
     tensor_SHA256=hashlib.sha256(memoryview(words).cast('B')).hexdigest()
     class Source:
      descriptor=dict(name=embedding.KEY,pointer=tensor.data_ptr(),shape=[163840,7168],dtype=1)
      metadata=dict(CPU_tensor_bytes=tensor.numel()*tensor.element_size(),tensor_SHA256=tensor_SHA256,checkpoint=header,
       actual_checkpoint_payload=True,CPU_source_prepare_and_hash_s=time.perf_counter()-source_started)
      @staticmethod
      def reference(ids):return tensor.index_select(0,torch.tensor(ids,dtype=torch.long,device='cpu')).contiguous().view(torch.int16).numpy().view(np.uint16)
     result['work']=embedding.run(lib,ctx,Source(),node=n,configuration=q,record=record,memory=memory,store=store,deadline=deadline)
     result['weight_keys']=1
     del words,tensor
    record('memory_before_destroy',memory())
    result['affinity_after_queries']=json.loads(lib.modern_owner_affinity_report_r2392())
    save()
   except BaseException as error:
    # Persist before the context manager attempts destroy; keeper can stop
    # peers even when an erroring node is waiting in a real collective drain.
    result.update(error=repr(error),traceback=traceback.format_exc());save();raise
  result['embedding_after_destroy']=json.loads(lib.modern_kimi_vocab_embedding_info_r461())
  assert all(owner['destroyed']for owner in result['embedding_after_destroy']['owners'])
  record('memory_after_destroy',memory())
  assert result['create_calls']==result['destroy_calls']==1 and not any(e['event']=='cleanup_error'for e in result['events'])
  result['passed']=True
 except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc());raise
 finally:result['completed']=True;result['finished_unix']=time.time();save()
if __name__=='__main__':main()
