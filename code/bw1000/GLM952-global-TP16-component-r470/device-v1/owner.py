"""One actual owner process; no Model/cache/weight/HTTP or Ex::submit calls."""
from pathlib import Path
import ctypes,hashlib,json,os,sys,time,socket,threading,runpy,traceback,base64
from shared import D,R,OUT,NODES,LABEL,H,actual,save,wait,verify
class DlInfo(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('symbol',ctypes.c_char_p),('address',ctypes.c_void_p)]
def owned_sockets():
 result=[]
 for f in Path('/proc/self/fd').iterdir():
  try:
   value=os.readlink(str(f))
   if value.startswith('socket:['):result.append(value[8:-1])
  except OSError:pass
 return sorted(set(result))
def main(node,device):
 epoch=float(os.environ['GLM470_EPOCH']);globalrank=node*8+device;report=OUT/('owner-%02d.json'%globalrank);q=dict(completed=False,passed=False,node=socket.gethostname().split('.')[0],node_index=node,device=device,global_rank=globalrank,PID=os.getpid(),UID=os.getuid(),startticks=Path('/proc/self/stat').read_text().split()[21],owner_TID=threading.get_native_id(),entry_CPUmask=sorted(os.sched_getaffinity(0)),Model_READY=False,weights=False,HTTP=0,cache_allocated=False,phase='before_library',events=[]);ctx=None;lib=None
 def mark(name):q['phase']=name;q['events'].append(dict(phase=name,epoch=time.time()));save(report,q);assert time.time()<epoch+150
 try:
  assert q['node']==NODES[node]and len(q['entry_CPUmask'])>=32 and q['PID']==q['owner_TID'];m,cpu=verify(os.environ['GLM470_GPU_MF']);q.update(actual_GPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],gpu_epoch=epoch,source_manifest_SHA256=m['source_manifest_SHA256'],library_SHA256=cpu['library_SHA256']);mark('library_load')
  rt=ctypes.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12');rt.cudaSetDevice.argtypes=[ctypes.c_int];rt.cudaSetDevice.restype=ctypes.c_int;rt.cudaGetDevice.argtypes=[ctypes.POINTER(ctypes.c_int)];rt.cudaGetDevice.restype=ctypes.c_int;rt.cudaMemGetInfo.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];rt.cudaMemGetInfo.restype=ctypes.c_int;rt.cudaDeviceGetPCIBusId.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];rt.cudaDeviceGetPCIBusId.restype=ctypes.c_int;assert rt.cudaSetDevice(device)==0;cur=ctypes.c_int(-1);assert rt.cudaGetDevice(ctypes.byref(cur))==0 and cur.value==device;pcib=ctypes.create_string_buffer(64);assert rt.cudaDeviceGetPCIBusId(pcib,64,device)==0;q['physical_PCI']=pcib.value.decode().lower();q['CUDA_runtime_actual_SHA256']=H('/opt/dtk/cuda/cuda/lib64/libcudart.so.12')
  def memory(floor):
   f=ctypes.c_size_t();t=ctypes.c_size_t();assert rt.cudaMemGetInfo(ctypes.byref(f),ctypes.byref(t))==0;assert f.value>=floor*1024**3;return dict(free=f.value,total=t.value,floor_GiB=floor)
  q['before_create']=memory(8);lib=ctypes.CDLL(str(actual(cpu['library'])));api=runpy.run_path(str(D.parent/'binding.py'));api['declare'](lib);dl=ctypes.CDLL('libdl.so.2');dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int;info=DlInfo();assert dl.dladdr(ctypes.cast(lib.glm_global_tp16_create_r470,ctypes.c_void_p),ctypes.byref(info))and H(info.name.decode())==cpu['library_SHA256'];q['CAPI_dladdr']=info.name.decode()
  maps={v.split()[-1]for v in Path('/proc/self/maps').read_text().splitlines()if '/'in v and v.split()[-1].startswith('/')};q['actual_provider_maps']={}
  for p,h in {cpu['library']:cpu['library_SHA256'],**cpu['verified_libraries']}.items():
   matches=[v for v in maps if Path(v).name==Path(p).name];identities={(os.stat(v).st_dev,os.stat(v).st_ino)for v in matches};assert len(identities)==1 and all(H(v)==h for v in matches),(p,matches);q['actual_provider_maps'][p]=dict(SHA256=h,paths=sorted(matches),unique_loaded_inode_count=1)
  size=lib.glm_global_tp16_id_size_r470();assert 0<size<=4096
  if globalrank==0:
   buffers=[ctypes.create_string_buffer(size)for _ in range(2)]
   for b in buffers:assert lib.glm_global_tp16_id_r470(b,size)==0
   data=[b.raw for b in buffers];assert data[0]!=data[1];save(OUT/'UIDS.json',dict(actual_GPU=q['actual_GPU'],epoch=epoch,source=m['source_manifest_SHA256'],UID_producer_PID=q['PID'],UID_producer_startticks=q['startticks'],world=base64.b64encode(data[0]).decode(),tp=base64.b64encode(data[1]).decode()))
  uid=wait(OUT/'UIDS.json',epoch);assert uid['actual_GPU']==q['actual_GPU']and uid['epoch']==epoch and uid['source']==m['source_manifest_SHA256'];data=[base64.b64decode(uid[k],validate=True)for k in ['world','tp']];assert len(data[0])==len(data[1])==size and data[0]!=data[1];q['UID_SHA256']=[hashlib.sha256(v).hexdigest()for v in data]
  raw=Path('/old/models/ZhipuAI-GLM-5.3/config.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69';cfg=json.loads(raw);cfg['glm_global_tp16_r470']=True;mark('create_global16');ctx=api['create'](lib,node,device,cfg,*data);q['after_create']=memory(8);q['after_create_CPUmask']=sorted(os.sched_getaffinity(0));assert q['after_create_CPUmask']==q['entry_CPUmask'];geo=json.loads(lib.glm_global_tp16_geometry_r470(ctx));assert geo['tp_rank']==geo['world_rank']==globalrank and geo['TP']==16 and geo['PP']==1 and geo['local_attention_heads']==4 and geo['local_index_heads']==2 and geo['layer_end']==78 and geo['capacity_per_row']==32768 and not geo['Model_READY']and not geo['cache_allocated'];q['geometry']=geo;q['owned_socket_inodes']=owned_sockets();save(OUT/('created-%02d.json'%globalrank),dict(global_rank=globalrank,epoch=epoch))
  for i in range(16):assert wait(OUT/('created-%02d.json'%i),epoch)==dict(global_rank=i,epoch=epoch)
  mark('one_collective_proof');rc=lib.glm_global_tp16_probe_r470(ctx);assert rc==0,lib.glm_global_tp16_error_r470().decode();geo=json.loads(lib.glm_global_tp16_geometry_r470(ctx));assert geo['collective_proof']and geo['TP_SUM']==136 and geo['TP_order']==list(range(1,17));q['proof']=geo;q['after_probe']=memory(8);q['after_probe_CPUmask']=sorted(os.sched_getaffinity(0));assert q['after_probe_CPUmask']==q['entry_CPUmask'];q['SDK_first_submit_qualification']=False;q['owned_socket_inodes']=sorted(set(q['owned_socket_inodes']+owned_sockets()));save(OUT/('proved-%02d.json'%globalrank),dict(global_rank=globalrank,epoch=epoch))
  for i in range(16):assert wait(OUT/('proved-%02d.json'%i),epoch)==dict(global_rank=i,epoch=epoch)
  mark('normal_close');assert lib.glm_global_tp16_close_r470(ctx)==0;ctx=None;q.update(normal_close=True,Runtime_weak_expired_CABI_assertion=True,after_close=memory(4),passed=True)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  # A failed collective cannot be safely re-entered or recreated. Owning step's
  # original bounded fail-stop controls peers; never invent normal destruction.
  if ctx:q['normal_close']=False
  q.update(completed=True,phase='normal_closed'if q['passed']else'failed');save(report,q)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
