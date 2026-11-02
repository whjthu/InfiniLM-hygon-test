"""Two T1 leases, numeric/state closure only. No ABBA or Model/performance result."""
from pathlib import Path
import os,json,hashlib,time,threading,concurrent.futures,traceback,ctypes as C,sys,subprocess
import numpy as np
R=Path('/refactor');D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():return [8575,8576]
def score_then_commit(rank,packet,score,commit,barrier,shared,precheck):
 # The same function is exercised by the CPU recording test and actual owners.
 # A failed score or numerical consumer aborts all owners before any commit.
 try:
  barrier.wait(120);score(0);barrier.wait(120);score(1);shared[rank]=packet;barrier.wait(120)
  if rank==0:precheck(shared)
  barrier.wait(120);commit();shared[rank]=packet;barrier.wait(120)
 except BaseException:
  barrier.abort();raise
def main():
 assert os.environ.get('DEEP390_GPU_GO')=='FROZEN_NUMERIC_CLAIM'
 mp=D/'GPU_MANIFEST.json';assert sha(mp)==os.environ['DEEP390_GPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert m['GPU_GO']is True
 assert all(sha(D/n)==h for n,h in m['source_SHA256'].items())
 S=R/'incoming/Deep-C4-same-cache-numeric-r390/cpu-v1';assert sha(S/'SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256'];sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
 cp=Path(m['CPU_receipt']);assert sha(cp)==m['CPU_receipt_SHA256'];build=json.loads(cp.read_bytes());assert build['passed']and build['GPU_GO']is False and sha(build['library'])==m['bridge_SHA256']==build['library_SHA256']
 sys.path.insert(0,str(S));import interface as api,consumer,reference as oracle
 import torch
 from safetensors import safe_open
 torch.set_num_threads(1);assert torch.cuda.device_count()==8 and len(os.sched_getaffinity(0))>=32
 deadline=float(os.environ['DEEP390_ABSOLUTE_DEADLINE']);started=time.monotonic()
 def remaining():assert time.time()<deadline-10,'Original numeric GPU deadline exhausted'
 model=Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');assert sha(model/'config.json')=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'
 cfg=json.loads((model/'config.json').read_bytes());mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];assert cfg['compress_ratios'][16]==4
 weights={};headers={}
 for fn in sorted({fn for k,fn in mapping.items()if k.startswith('layers.16.attn.indexer.')}):
  remaining();p=model/fn
  with p.open('rb')as f:size=int.from_bytes(f.read(8),'little');header=f.read(size)
  headers[fn]={'size':p.stat().st_size,'header_SHA256':hashlib.sha256(header).hexdigest()}
  with safe_open(p,framework='pt',device='cpu')as reader:
   for key in sorted(k for k,v in mapping.items()if v==fn and k.startswith('layers.16.attn.indexer.')):weights[key]=reader.get_tensor(key).contiguous()
 seed=torch.Generator().manual_seed(3598496);hidden=torch.randn((8577,4096),generator=seed).to(torch.bfloat16);query=torch.randn((8577,1024),generator=seed).to(torch.bfloat16)
 lib=C.CDLL(build['library']);f=api.configure(lib)
 cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int];hardware=[]
 for device in range(8):
  bus=C.create_string_buffer(64);assert cuda.cudaDeviceGetPCIBusId(bus,64,device)==0;hardware.append({'device':device,'PCI':bus.value.decode()})
 assert len({v['PCI']for v in hardware})==8
 class Info(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('saddr',C.c_void_p)]
 dl=C.CDLL(None).dladdr;dl.argtypes=[C.c_void_p,C.POINTER(Info)];dl.restype=C.c_int;providers={}
 undef=subprocess.check_output(['nm','-D','--undefined-only',build['library']],text=True);c4symbols=[line.split()[-1]for line in undef.splitlines()if'C4Indexer7prepare'in line or'C4Indexer6commit'in line];assert len(c4symbols)==2
 for symbol in ['deep_c4_lease_prepare_r390','deep_c4_lease_score_r390','deep_c4_lease_commit_r390','deep_paged_scores_r359']+c4symbols:
  info=Info();assert dl(C.cast(getattr(lib,symbol),C.c_void_p),C.byref(info));path=Path(info.fname.decode()).resolve();expected=build['current_Native_SHA256']if symbol in c4symbols else build['library_SHA256'];assert sha(path)==expected;providers[symbol]={'path':str(path),'SHA256':sha(path)}
 def check(rc):
  if rc:raise RuntimeError(f('error')().decode())
 def mapped_providers():
  paths=sorted({line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if'/'in line and '.so'in line and any(tag in line for tag in ['libDeep_','libDeepNative','libdeep_native'])})
  rows=[{'path':p,'SHA256':sha(p)}for p in paths]
  assert any(v['SHA256']==build['current_Native_SHA256']for v in rows)
  assert any(v['SHA256']==build['library_SHA256']for v in rows)
  assert not any('libDeep_C4_runtime_r359' in v['path']or v['SHA256']=='a4e6208eef89bd63045c01bcab5ccc358e6a36cd6ea034aa286b2934ba82d91b'for v in rows),'Old private C4 Native must not be loaded'
  return rows
 uid=C.create_string_buffer(f('id_size')());check(f('id')(uid));barrier=threading.Barrier(8);lock=threading.Lock();shared=[None]*8;previous=[None];metrics={}
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];folder=R/'results'/('Deep390-samecache-'+step);folder.mkdir(exist_ok=False);out=folder.parent/(folder.name+'.json');assert not out.exists()
 report=dict(hardware=hardware,actual_providers=providers,selected_weight_payload_SHA256={k:hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()for k,v in weights.items()},completed=False,passed=False,normal=False,actual_step=step,plan=plan(),performance_trials=0,Model_loaded=False,Model_gain='UNKNOWN',CPU_receipt_SHA256=sha(cp),bridge_SHA256=sha(build['library']),current_Native_SHA256=build['current_Native_SHA256'],old_HIP_object_SHA256=build['old_HIP_object_SHA256'],source_manifest_SHA256=sha(mp),checkpoint={'config_SHA256':sha(model/'config.json'),'index_SHA256':sha(model/'model.safetensors.index.json'),'headers':headers},owners=[dict(rank=i,normal_close=False)for i in range(8)],steps=[],raw_archives=[],scope='Actual original C4 prepare once per step; same frozen valid cache and Pending inputs; SDK and old paged replay; CPU oracle; one original commit; next8576 consumes newslot2143. Not Model, not performance, not original684 numerical qualification.')
 def save():
  with lock:tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out)
 def retain(packets,past,phase):
  # All evidence stays remote. Publish only hashes, metrics and source code.
  for packet in packets:
   rank=packet['rank'];arrays={};meta={k:v for k,v in packet.items()if k.startswith('identity_')};meta['fields']={}
   for stage in (['before','after_sdk','after_paged']if phase=='precommit'else['post_commit']):
    meta['fields'][stage]={}
    for key,value in packet[stage].items():
     if value is None:meta['fields'][stage][key]=None;continue
     name=stage+'__'+str(key);arrays[name]=value['raw'];meta['fields'][stage][key]={'shape':value['shape'],'dtype':value['dtype'],'array':name,'SHA256':hashlib.sha256(value['raw'].tobytes()).hexdigest()}
   npz=folder/(f'owner{rank}-past{past}-{phase}.npz');np.savez_compressed(npz,**arrays);jp=npz.with_suffix('.json');jp.write_text(json.dumps(meta,indent=2)+'\n');report['raw_archives'].append({'rank':rank,'past':past,'phase':phase,'npz':str(npz),'npz_SHA256':sha(npz),'metadata':str(jp),'metadata_SHA256':sha(jp)})
 def owner(rank):
  ctx=None;row=report['owners'][rank];initial=sorted(os.sched_getaffinity(0));assigned=initial[rank*4:rank*4+4]
  try:
   remaining();torch.cuda.set_device(rank);props=torch.cuda.get_device_properties(rank);assert'gfx936'in props.gcnArchName
   ctx=f('create')(rank,json.dumps(cfg).encode(),uid);assert ctx,f('error')().decode();names=json.loads(f('keys')(ctx));assert set(names)<=set(weights)
   row.update(TID=threading.get_native_id(),initial_affinity=initial,expected_affinity=assigned,loaded_weight_keys=names,arch=props.gcnArchName)
   for name in names:
    w=weights[name];shape=(api.Z*w.ndim)(*w.shape);dtype={torch.int8:0,torch.float32:1,torch.bfloat16:2}[w.dtype];check(f('load')(ctx,name.encode(),api.P(w.data_ptr()),w.ndim,shape,dtype))
   check(f('finish')(ctx));check(f('reset')(ctx));barrier.wait(120)
   for past in range(0,8575,1024):
    count=min(1024,8575-past);x=hidden[past:past+count].to(rank);q=query[past:past+count].to(rank);pos=torch.arange(past,past+count,dtype=torch.int64,device=rank)
    check(f('prefix')(ctx,past,count,api.P(x.data_ptr()),api.P(q.data_ptr()),api.P(pos.data_ptr()),api.P(torch.cuda.current_stream(rank).cuda_stream)))
   row['actual_affinity']=sorted(os.sched_getaffinity(0));assert row['actual_affinity']==assigned
   for past in plan():
    remaining();x=hidden[past:past+1].to(rank);q=query[past:past+1].to(rank);pos=torch.tensor([past],dtype=torch.int64,device=rank);xc=x.clone();qc=q.clone();pc=pos.clone();torch.cuda.synchronize(rank)
    token,packet=api.begin(f,check,ctx,past,api.P(x.data_ptr()),api.P(q.data_ptr()),api.P(pos.data_ptr()),api.P(torch.cuda.current_stream(rank).cuda_stream));packet['rank']=rank
    def precheck(packets):
     retain(packets,past,'precommit');save();metrics[past]=consumer.validate_same_cache(packets,oracle,previous[0],committed=False)
    score_then_commit(rank,packet,lambda mode:api.score(f,check,ctx,token,mode,packet),lambda:api.commit(f,check,ctx,token,packet),barrier,shared,precheck)
    assert torch.equal(x,xc)and torch.equal(q,qc)and torch.equal(pos,pc)
    if rank==0:
     retain(shared,past,'postcommit');save();final=consumer.validate_same_cache(shared,oracle,previous[0]);report['steps'].append(final);previous[0]=list(shared);save()
    barrier.wait(120)
  except BaseException as exc:row.update(error=repr(exc),traceback=traceback.format_exc());barrier.abort();raise
  finally:
   if ctx:check(f('destroy')(ctx))
   check(f('shutdown')());row['normal_close']=True;save()
 save()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
  maps=mapped_providers();report.update(completed=True,passed=len(report['steps'])==2,normal=all(r['normal_close']for r in report['owners']),final_mapped_providers=maps)
 except BaseException as exc:report.update(completed=False,passed=False,error=repr(exc),traceback=traceback.format_exc());raise
 finally:report['driver_wall_s']=time.monotonic()-started;save()
 print(json.dumps({'completed':report['completed'],'passed':report['passed'],'normal':report['normal'],'path':str(out)}))
if __name__=='__main__':main()
