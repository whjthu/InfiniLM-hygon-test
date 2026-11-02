"""One current C4 layer TP8: two warm queries and OFF/ON/OFF, timing diagnosis only."""
from pathlib import Path
import os,json,hashlib,time,threading,concurrent.futures,traceback,ctypes as C,sys,subprocess,contextlib
import numpy as np
R=Path('/refactor');D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('DEEP406_GPU_GO')=='FROZEN_STATUS_CLAIM'
 mp=D/'GPU_MANIFEST.json';assert sha(mp)==os.environ['DEEP406_GPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert m['GPU_GO']is True
 assert all(sha(D/n)==h for n,h in m['source_SHA256'].items())
 S=R/'incoming/Deep-C4-status-enclosure-r406/cpu-v2';assert sha(S/'SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256'];sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
 cp=Path(m['CPU_receipt']);assert sha(cp)==m['CPU_receipt_SHA256'];build=json.loads(cp.read_bytes());assert build['passed']and build['completed']and sha(build['library'])==m['bridge_SHA256']==build['library_SHA256']and build['HOST_compile_count']==2
 sys.path.insert(0,str(S));import driver_core,reference as oracle
 import interface as api,consumer
 import torch
 from safetensors import safe_open
 torch.set_num_threads(1);assert torch.cuda.device_count()==8 and len(os.sched_getaffinity(0))>=32
 deadline=float(os.environ['DEEP406_ABSOLUTE_DEADLINE']);started=time.monotonic()
 def remaining():assert time.time()<deadline-8,'Original GPU deadline exhausted'
 model=Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');assert sha(model/'config.json')=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'
 fixture=json.loads((D/'FIXTURE_PINS.json').read_bytes());assert sha(model/'model.safetensors.index.json')==fixture['index_SHA256']
 cfg=json.loads((model/'config.json').read_bytes());mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];assert cfg['compress_ratios'][16]==4 and cfg['index_topk']==512
 weights={};headers={};loadstart=time.perf_counter()
 for fn in sorted({fn for k,fn in mapping.items()if k.startswith('layers.16.attn.indexer.')}):
  remaining();p=model/fn
  with p.open('rb')as f:size=int.from_bytes(f.read(8),'little');header=f.read(size)
  headers[fn]={'size':p.stat().st_size,'header_SHA256':hashlib.sha256(header).hexdigest(),'full_payload_rehashed':False};assert headers[fn]==fixture['headers'][fn]
  with safe_open(p,framework='pt',device='cpu')as reader:
   for key in sorted(k for k,v in mapping.items()if v==fn and k.startswith('layers.16.attn.indexer.')):
    w=reader.get_tensor(key).contiguous();assert list(w.shape)==fixture['selected_parameters'][key]['shape']and hashlib.sha256(w.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()==fixture['selected_parameters'][key]['payload_SHA256'];weights[key]=w
 seed=torch.Generator().manual_seed(3598496);hidden=torch.randn((8497,4096),generator=seed).to(torch.bfloat16);query=torch.randn((8497,1024),generator=seed).to(torch.bfloat16)
 lib=C.CDLL(build['library']);f=api.configure(lib);dtypes={name:f('dtype')(i)for i,name in enumerate(['F32','BF16','I64','U8'])}
 def check(rc):
  if rc:raise RuntimeError(f('error')().decode())
 cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int];hardware=[]
 for device in range(8):
  props=torch.cuda.get_device_properties(device);bus=C.create_string_buffer(64);assert'gfx936'in props.gcnArchName and cuda.cudaDeviceGetPCIBusId(bus,64,device)==0
  hardware.append(dict(device=device,PCI=bus.value.decode(),arch=props.gcnArchName))
 assert len({x['PCI']for x in hardware})==8
 def mapped_providers():
  paths=sorted({line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if'/'in line and '.so'in line and any(tag in line for tag in ['libDeep_','libDeepNative','libdeep_native'])});rows=[dict(path=p,SHA256=sha(p))for p in paths]
  assert any(v['SHA256']==build['current_Native_SHA256']for v in rows)and any(v['SHA256']==build['library_SHA256']for v in rows)
  assert len([v for v in rows if'HC_T1_Model'in v['path']])==1 and not any('libDeep_C4_runtime_r359'in v['path']or'libDeep_latest_native'in v['path']or'libDeep_C4_same_cache'in v['path']for v in rows)
  maps=Path('/proc/self/maps').read_text();sdk_paths=sorted({line.split()[-1]for line in maps.splitlines()if'/'in line and any(tag in line for tag in ['libinfiniops.so','libinfinilm_runtime.so','libinfiniccl.so','libinfinirt.so','_infinilm.cpython'])})
  sdk=[dict(path=p,SHA256=sha(p))for p in sdk_paths];assert all(any(v['SHA256']==expected['SHA256']for v in sdk)for expected in build['SDK_bindings'].values())
  rows.extend(sdk)
  return rows
 providers=mapped_providers();uid=C.create_string_buffer(f('id_size')());check(f('id')(uid));barrier=threading.Barrier(8);lock=threading.RLock();shared=[None]*8;previous=[None];metrics={}
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];folder=R/'results'/('Deep406-status-'+step);folder.mkdir(exist_ok=False);out=folder.parent/(folder.name+'.json');assert not out.exists()
 report=dict(completed=False,passed=False,normal=False,actual_step=step,hardware=hardware,CPU_receipt_SHA256=sha(cp),source_manifest_SHA256=sha(mp),bridge_SHA256=build['library_SHA256'],current_Native_SHA256=build['current_Native_SHA256'],initial_mapped_providers=providers,plan=driver_core.plan(),owners=[dict(rank=i,normal_close=False,calls=[])for i in range(8)],checkpoint=dict(config_SHA256=sha(model/'config.json'),index_SHA256=sha(model/'model.safetensors.index.json'),headers=headers),selected_weight_payload_SHA256={k:hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()for k,v in weights.items()},input=dict(kind='Synthetic BF16 hidden/q_lora, real layer16 checkpoint weights; not Model activations',seed=3598496,hidden_SHA256=hashlib.sha256(hidden.view(torch.uint8).numpy().tobytes()).hexdigest(),q_lora_SHA256=hashlib.sha256(query.view(torch.uint8).numpy().tobytes()).hexdigest()),weight_read_input_prepare_s=time.perf_counter()-loadstart,raw_archives=[],timing_scope='Status HOST completion enclosure includes upstream GPU completion wait and CPU/spin; not pure D2H. ThreadCPU separate. CPP prepare/commit excludes added state snapshots; CAPI includes snapshots, producer sync; checked wall additionally H2D/input validation/proofs/CPU oracle/archive. No multiply21/43 or Model attribution.',Model_loaded=False,Model_gain='UNKNOWN',synchronization_and_failure_containment_unchanged=True)
 def save():
  with lock:temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out)
 def retain(packet):
  rank=packet['rank'];i=packet['item']['index'];arrays={};meta={}
  for k,v in packet['fields'].items():
   if v is None:meta[k]=None;continue
   arrays[str(k)]=v['raw'];meta[k]=dict(shape=v['shape'],dtype=v['dtype'],array=str(k),SHA256=hashlib.sha256(v['raw'].tobytes()).hexdigest())
  p=folder/f'query{i}-owner{rank}.npz';np.savez(p,**arrays);j=p.with_suffix('.json');j.write_text(json.dumps(dict(fields=meta,stats=packet['stats'],item=packet['item']),indent=2)+'\n');report['raw_archives'].append(dict(rank=rank,query=i,npz=str(p),npz_SHA256=sha(p),metadata=str(j),metadata_SHA256=sha(j)))
 def owner(rank):
  row=report['owners'][rank];initial=sorted(os.sched_getaffinity(0));assigned=initial[rank*4:rank*4+4]
  def create(device):
   remaining();torch.cuda.set_device(device);ctx=f('create')(device,json.dumps(cfg).encode(),uid);assert ctx,f('error')().decode();names=json.loads(f('keys')(ctx));assert set(names)<=set(weights)
   row.update(TID=threading.get_native_id(),initial_affinity=initial,expected_affinity=assigned,loaded_weight_keys=names)
   try:
    for name in names:
     w=weights[name];shape=(api.Z*w.ndim)(*w.shape);dtype={torch.int8:0,torch.float32:1,torch.bfloat16:2}[w.dtype];check(f('load')(ctx,name.encode(),api.P(w.data_ptr()),w.ndim,shape,dtype))
    check(f('finish')(ctx));barrier.wait(90);return ctx
   except BaseException:
    try:check(f('destroy')(ctx))
    finally:check(f('shutdown')())
    raise
  def trial(ctx,item):
   remaining();barrier.wait(90);check(f('reset')(ctx));prefix_start=time.perf_counter()
   for past in range(0,8496,1024):
    remaining();count=min(1024,8496-past);x=hidden[past:past+count].to(rank);q=query[past:past+count].to(rank);pos=torch.arange(past,past+count,dtype=torch.int64,device=rank)
    check(f('forward')(ctx,past,count,api.P(x.data_ptr()),api.P(q.data_ptr()),api.P(pos.data_ptr()),api.P(torch.cuda.current_stream(rank).cuda_stream),0,0))
   prefix_s=time.perf_counter()-prefix_start;row['actual_affinity']=sorted(os.sched_getaffinity(0));assert row['actual_affinity']==assigned
   barrier.wait(90);begin=time.perf_counter();x=hidden[8496:8497].to(rank);q=query[8496:8497].to(rank);pos=torch.tensor([8496],dtype=torch.int64,device=rank);xc=x.clone();qc=q.clone();pc=pos.clone();torch.cuda.synchronize(rank);input_ready=time.perf_counter()
   tick=time.perf_counter();check(f('forward')(ctx,8496,1,api.P(x.data_ptr()),api.P(q.data_ptr()),api.P(pos.data_ptr()),api.P(torch.cuda.current_stream(rank).cuda_stream),item['timing'],1));CAPI_wall=time.perf_counter()-tick
   assert torch.equal(x,xc)and torch.equal(q,qc)and torch.equal(pos,pc);st=api.stats(f,check,ctx)
   if row['calls']:assert st['owner_thread_hash']==row['calls'][0]['stats']['owner_thread_hash']
   fields={k:api.proof(f,check,ctx,k)for k in range(11)};proof_end=time.perf_counter()
   packet=dict(rank=rank,item=item,stats=st,fields=fields);shared[rank]=packet;barrier.wait(90)
   if rank==0:
    metrics[item['index']]=consumer.qualify(shared,oracle,dtypes,previous[0]);previous[0]=list(shared)
    for packet in shared:retain(packet)
    save()
   barrier.wait(90);end=time.perf_counter();rec=dict(**item,prefix_rebuild_s=prefix_s,measurement_begin_s=begin,input_H2D_clone_s=input_ready-begin,CAPI_wall_s=CAPI_wall,proof_read_end_s=proof_end,measurement_end_s=end,checked_local_wall_s=end-begin,stats=st,numeric=metrics[item['index']][rank]);row['calls'].append(rec);barrier.wait(90)
   if rank==0:save()
   return rec
  def destroy(ctx):check(f('destroy')(ctx));row['destroy_returned']=True
  def shutdown():check(f('shutdown')());row['Runtime_expired']=True;row['normal_close']=row.get('destroy_returned',False);save()
  try:return driver_core.run_owner(rank,create,trial,destroy,shutdown,barrier.abort)
  except BaseException as exc:row.update(error=repr(exc),traceback=traceback.format_exc());save();raise
 save()
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
  report['matched_queries']=[]
  for item in driver_core.plan():
   rows=[o['calls'][item['index']]for o in report['owners']];critical=max(rows,key=lambda r:r['stats']['prepare_s']+r['stats']['commit_s']);owner_index=rows.index(critical)
   report['matched_queries'].append(dict(**item,max8_CPP_prepare_commit_s=critical['stats']['prepare_s']+critical['stats']['commit_s'],CPP_critical_owner=owner_index,critical_owner_stats=critical['stats'],max8_CAPI_wall_s=max(r['CAPI_wall_s']for r in rows),same_trial_checked_batch_wall_s=max(r['measurement_end_s']for r in rows)-min(r['measurement_begin_s']for r in rows)))
  report.update(completed=True,passed=len(metrics)==5,normal=all(o['normal_close']for o in report['owners']),final_mapped_providers=mapped_providers())
 except BaseException as exc:report.update(completed=True,passed=False,error=repr(exc),traceback=traceback.format_exc());raise
 finally:report['driver_wall_s']=time.monotonic()-started;save()
 print(json.dumps(dict(completed=report['completed'],passed=report['passed'],normal=report['normal'],path=str(out))))
if __name__=='__main__':main()
