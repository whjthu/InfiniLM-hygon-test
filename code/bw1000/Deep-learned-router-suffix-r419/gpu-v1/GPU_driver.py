"""One complete learned MoE layer, same immutable BF16 input, TP8; no Model."""
from pathlib import Path
import os,sys,json,hashlib,time,threading,concurrent.futures,traceback,ctypes as C,subprocess
import numpy as np
R=Path('/refactor');D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert os.environ.get('DEEP419_GPU_GO')=='FROZEN_COMPLETE_MOE_CLAIM'
 mp=D/'GPU_MANIFEST.json';assert sha(mp)==os.environ['DEEP419_GPU_MANIFEST_SHA'];m=json.loads(mp.read_bytes());assert m['GPU_GO']is True
 assert all(sha(D/n)==h for n,h in m['source_SHA256'].items())
 S=R/'incoming/Deep-learned-router-suffix-r419/cpu-v5';assert sha(S/'SOURCE_MANIFEST.json')==m['production_source_manifest_SHA256'];sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
 assert sha(m['bridge'])==m['bridge_SHA256'] and sha(m['Native'])==m['Native_SHA256']
 pins=json.loads((S/'PARENT_PINS.json').read_bytes());assert all(sha(p)==v['SHA256']for p,v in pins['actual_DTK_bindings'].items())
 for item in m['compiled_objects']:assert sha(item['object'])==item['object_SHA256']and sha(item['MD_path'])==item['MD_SHA256']
 sys.path.insert(0,str(S));import driver_core,interface as api,semantic_reference
 assert Path(api.__file__).resolve()==S/'interface.py'and Path(driver_core.__file__).resolve()==S/'driver_core.py'
 deadline=float(os.environ['DEEP419_ABSOLUTE_DEADLINE']);started=time.monotonic()
 def remaining():assert time.time()<deadline-8,'Original GPU deadline exhausted'
 step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];folder=R/'results'/('Deep419-MoE-'+step);folder.mkdir(exist_ok=False);out=folder.parent/(folder.name+'.json')
 report=dict(completed=False,passed=False,normal=False,actual_step=step,source_manifest_SHA256=sha(mp),bridge_SHA256=m['bridge_SHA256'],Native_SHA256=m['Native_SHA256'],Model_loaded=False,Model_gain='UNKNOWN',phase_observer=False,owners=[dict(rank=i,normal_close=False,calls=[])for i in range(8)],raw_archives=[],plan=driver_core.plan(),actual_SDK_metadata_executed=False,original_CPU851_producer_passed=False,CPU_math_original_PASS=13,CPU_known_underflow_classified_FAIL=1)
 lock=threading.RLock();barrier=threading.Barrier(8);shared=[None]*8;metrics={};context={}
 def save():
  with lock:t=out.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(out)
 def journal(event):
  with lock:
   with(folder/'scalar.jsonl').open('a')as f:f.write(json.dumps(event)+'\n')
 def check(rc):
  if rc:raise RuntimeError(f('error')().decode())
 def decode(field,kind,shape):
  assert field is not None and field['dtype']==dtypes[kind]and tuple(field['shape'])==shape
  dtype={'F32':np.float32,'I64':np.int64,'BF16':np.uint16}[kind];a=field['raw'].view(dtype).reshape(shape).copy()
  if kind=='BF16':return(a.astype(np.uint32)<<16).view(np.float32)
  return a
 def nr(actual,reference):return float(np.linalg.norm(actual.astype(np.float64)-reference.astype(np.float64))/(np.linalg.norm(reference.astype(np.float64))+1e-30))
 def retain(rank,tag,fields,extra):
  p=folder/(tag+'-owner'+str(rank)+'.npz');np.savez(p,**{str(k):v['raw']for k,v in fields.items()if v is not None})
  info={str(k):None if v is None else dict(shape=v['shape'],dtype=v['dtype'],SHA256=hashlib.sha256(v['raw'].tobytes()).hexdigest())for k,v in fields.items()}
  j=p.with_suffix('.json');j.write_text(json.dumps(dict(fields=info,**extra),indent=2)+'\n')
  report['raw_archives'].append(dict(rank=rank,tag=tag,npz=str(p),npz_SHA256=sha(p),metadata=str(j),metadata_SHA256=sha(j)))
 def snapshot(ctx,rank=None,tag='unknown'):
  fields={k:api.proof(f,check,ctx,k)for k in (0,1,2)}
  try:
   tokens=fields[0]['shape'][0];assert tokens in (1,2)
   weights=decode(fields[1],'F32',(tokens,6));ids=decode(fields[2],'I64',(tokens,6))
   value=decode(fields[0],'BF16',(tokens,4096));assert np.isfinite(value).all()and np.isfinite(weights).all()
   assert all(len(set(row.tolist()))==6 and np.all((row>=0)&(row<256))for row in ids)
   return dict(fields=fields,value=value,weights=weights,ids=ids)
  except BaseException as exc:
   if rank is not None:retain(rank,'failed-'+tag,fields,dict(error=repr(exc),failed_numeric=True))
   raise
 def output_compare(a,b,limit=.01):
  err=nr(a['value'],b['value']);assert err<limit,('complete MoE NRMSE',err,limit)
  return err
 save()
 try:
  # Exactly one metadata executable in the same eight-card allocation, before
  # Runtime creation and checkpoint payload loading. Its SDK static constructors
  # really query a visible device; ownmain GPU_calls=0 does NOT cover that startup.
  remaining();exe=Path(m['metadata_executable']);assert sha(exe)==m['metadata_executable_SHA256'];tick=time.perf_counter()
  meta=subprocess.run([str(exe)],text=True,capture_output=True,timeout=min(90,max(1,int(deadline-time.time()-10))))
  (folder/'SDK-metadata-device-startup.log').write_text(meta.stdout+meta.stderr);assert meta.returncode==0,meta.stderr[-4000:]
  sdk_metadata=json.loads(meta.stdout.splitlines()[-1]);assert sdk_metadata['passed']and not sdk_metadata['Runtime_created']
  for name,v in sdk_metadata['SDK_default_metadata_only'].items():assert v['default']in v['active']and v['needs_resolution']is False,(name,v)
  report.update(actual_SDK_metadata_executed=True,SDK_metadata=sdk_metadata,SDK_device_query_startup_s=time.perf_counter()-tick,metadata_exe_static_GPU_queries_not_zero=True);save()
  import torch
  from safetensors import safe_open
  torch.set_num_threads(1);assert torch.cuda.device_count()==8 and len(os.sched_getaffinity(0))>=32
  torch.cuda.set_device(0)
  model=Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');fixture=json.loads((S/'FIXTURE_METADATA.json').read_bytes())
  assert sha(model/'config.json')==fixture['config_SHA256']and sha(model/'model.safetensors.index.json')==fixture['index_SHA256']
  cfg=json.loads((model/'config.json').read_bytes());mapping=json.loads((model/'model.safetensors.index.json').read_bytes())['weight_map'];weights={};payloads={};headers={};tick=time.perf_counter()
  for name,info in fixture['files'].items():
   remaining();p=model/name
   with p.open('rb')as stream:size=int.from_bytes(stream.read(8),'little');raw=stream.read(size)
   assert p.stat().st_size==info['size']and hashlib.sha256(raw).hexdigest()==info['header_SHA256'];h=json.loads(raw);headers[name]=info
   with safe_open(p,framework='pt',device='cpu')as reader:
    for key in fixture['keys']:
     entry=fixture['parameters'][key]
     if entry['file']!=name:continue
     assert mapping[key]==name and all(h[key][field]==entry[field]for field in ['dtype','shape','data_offsets'])
     w=reader.get_tensor(key).contiguous();assert list(w.shape)==entry['shape'];weights[key]=w;payloads[key]=hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()
  assert set(weights)==set(fixture['keys'])and len(weights)==1544
  seed=torch.Generator().manual_seed(41920261103);hidden=torch.randn((2,4096),generator=seed).to(torch.bfloat16);hidden_sha=hashlib.sha256(hidden.view(torch.uint8).numpy().tobytes()).hexdigest()
  bias=weights['layers.16.ffn.gate.bias'].numpy().copy();assert bias.dtype==np.float32 and bias.shape==(256,)
  report.update(weight_payload_read_and_hash_s=time.perf_counter()-tick,checkpoint=dict(config_SHA256=fixture['config_SHA256'],index_SHA256=fixture['index_SHA256'],headers=headers,selected_key_count=1544,selected_weight_payload_SHA256=payloads,global_selected_payload_bytes=sum(w.numel()*w.element_size()for w in weights.values())),input=dict(kind='Synthetic immutable BF16 hidden shared identically by TP8; actual layer16 weights, not Model activations',seed=41920261103,shape=[2,4096],SHA256=hidden_sha))
  lib=C.CDLL(m['bridge']);f=api.configure(lib);dtypes={name:f('dtype')(i)for i,name in enumerate(['F32','BF16','I64'])}
  def providers():
   maps=Path('/proc/self/maps').read_text();names=sorted({line.split()[-1]for line in maps.splitlines()if'/'in line and any(x in line for x in ['libDeep_','libinfiniops.so','libinfinilm_runtime.so','libinfinirt.so','libinfiniccl.so','_infinilm.cpython'])});rows=[dict(path=p,SHA256=sha(p))for p in names]
   native=[r for r in rows if'libDeep_'in r['path']];assert len(native)==2 and {v['SHA256']for v in native}=={m['bridge_SHA256'],m['Native_SHA256']},native
   assert all(any(row['SHA256']==v['SHA256']for row in rows)for v in pins['actual_DTK_bindings'].values());return rows
  report['initial_providers']=providers()
  cuda=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so');cuda.cudaDeviceGetPCIBusId.argtypes=[C.c_char_p,C.c_int,C.c_int];hardware=[]
  for rank in range(8):
   prop=torch.cuda.get_device_properties(rank);bus=C.create_string_buffer(64);assert'gfx936'in prop.gcnArchName and cuda.cudaDeviceGetPCIBusId(bus,64,rank)==0;hardware.append(dict(rank=rank,PCI=bus.value.decode(),arch=prop.gcnArchName))
  assert len({v['PCI']for v in hardware})==8;report['hardware']=hardware
  uid=C.create_string_buffer(f('id_size')());check(f('id')(uid));common_affinity=sorted(os.sched_getaffinity(0));baseline={}
  def owner(rank):
   row=report['owners'][rank];initial=sorted(os.sched_getaffinity(0));assigned=common_affinity[rank*4:rank*4+4];device_input=None;device_initial=None
   def create(device):
    remaining();torch.cuda.set_device(device);ctx=f('create')(device,json.dumps(cfg).encode(),uid);assert ctx,f('error')().decode();row.update(TID=threading.get_native_id(),initial_affinity=initial,expected_affinity=assigned);return ctx
   def initialize(ctx):
    nonlocal device_input,device_initial
    names=json.loads(f('keys')(ctx));assert set(names)==set(weights)
    for name in names:
     remaining();w=weights[name];shape=(api.Z*w.ndim)(*w.shape);dtype={torch.int8:0,torch.float32:1,torch.bfloat16:2}[w.dtype];check(f('load')(ctx,name.encode(),api.P(w.data_ptr()),w.ndim,shape,dtype))
    check(f('finish')(ctx));device_input=hidden.to(rank);device_initial=device_input.clone();torch.cuda.synchronize(rank);row['loaded_key_count']=len(names);barrier.wait(90)
   def qualify(ctx):
    remaining();x=device_input[:1];stream=api.P(torch.cuda.current_stream(rank).cuda_stream);check(f('project')(ctx,api.P(x.data_ptr()),stream));p=api.proof(f,check,ctx,3);logits=decode(p,'F32',(1,256)).reshape(256);assert np.isfinite(logits).all();logits_hash=hashlib.sha256(p['raw'].tobytes()).hexdigest();packets=[]
    for mode in [0,1]:
     check(f('same_logits')(ctx,mode));fields={k:api.proof(f,check,ctx,k)for k in [1,2,3,4,5]}
     st=api.stats(f,check,ctx);retain(rank,'same-logits-mode'+str(mode),fields,dict(stats=st,collected_before_numeric_gate=True))
     assert hashlib.sha256(fields[3]['raw'].tobytes()).hexdigest()==logits_hash
     candidate=dict(scores=decode(fields[4],'F32',(1,256)).reshape(256),selection=decode(fields[5],'F32',(1,256)).reshape(256),ids=decode(fields[2],'I64',(1,6)).reshape(6),weights=decode(fields[1],'F32',(1,6)).reshape(6));assert all(np.isfinite(v).all()for v in candidate.values());st=api.stats(f,check,ctx);assert(st['calls'],st['hits'],st['fallbacks'])==(1,mode,1-mode)
     cpu=semantic_reference.torch_reference(torch,logits,bias);cpu_check=semantic_reference.validate(torch,logits,bias,cpu,candidate)
     if mode:device_check=semantic_reference.validate(torch,logits,bias,packets[0],candidate)
     else:device_check=None
     packets.append(candidate);row.setdefault('same_logit_checks',[]).append(dict(mode=mode,stats=st,independent_CPU_reference=cpu_check,actual_SDK_baseline_reference=device_check))
    row['same_logits_qualified']=True;row['actual_affinity']=sorted(os.sched_getaffinity(0));assert row['actual_affinity']==assigned
    # Only one ineligible geometry, untimed. Hash layers are not loaded by this layer16 bridge.
    fall=[]
    for mode in [0,1]:
     barrier.wait(90);check(f('forward')(ctx,2,api.P(device_input.data_ptr()),stream,mode));packet=snapshot(ctx,rank,'T2-fallback-mode'+str(mode));st=api.stats(f,check,ctx);retain(rank,'T2-fallback-mode'+str(mode),packet['fields'],dict(stats=st,untimed=True));assert(st['calls'],st['hits'],st['fallbacks'])==(1,0,1);fall.append(packet)
    row['T2_fallback_NRMSE']=output_compare(fall[1],fall[0]);assert np.array_equal(fall[0]['ids'],fall[1]['ids']);assert torch.equal(device_input,device_initial);barrier.wait(90)
   def trial(ctx,item):
    remaining();barrier.wait(90);begin=time.perf_counter();x=device_input[:1];canary=x.clone();torch.cuda.synchronize(rank);ready=time.perf_counter();tick=time.perf_counter();check(f('forward')(ctx,1,api.P(x.data_ptr()),api.P(torch.cuda.current_stream(rank).cuda_stream),item['mode']));returned=time.perf_counter();st=api.stats(f,check,ctx);packet=snapshot(ctx,rank,'query'+str(item['index']))
    try:
     assert(st['calls'],st['hits'],st['fallbacks'])==(1,item['mode'],1-item['mode']);assert torch.equal(x,canary);proof_end=time.perf_counter()
     if rank in baseline:change=output_compare(packet,baseline[rank])
     else:assert item['mode']==0;baseline[rank]=packet;change=0.
    except BaseException as exc:retain(rank,'failed-query'+str(item['index']),packet['fields'],dict(stats=st,error=repr(exc)));raise
    shared[rank]=packet;barrier.wait(90)
    if rank==0:
     for i,pkt in enumerate(shared):retain(i,'query'+str(item['index']),pkt['fields'],dict(item=item,collected_before_cross_owner_gate=True))
     assert all(np.array_equal(shared[0]['ids'],v['ids'])for v in shared[1:]),'TP8 inconsistent route IDs'
     cross=[output_compare(v,shared[0],.002)for v in shared]
     metrics[item['index']]=cross
    barrier.wait(90);end=time.perf_counter();record=dict(**item,measurement_begin_s=begin,local_proof_end_s=proof_end,measurement_end_s=end,input_clone_and_drain_s=ready-begin,forward_CAPI_s=returned-tick,CAPI_with_proof_s=proof_end-tick,checked_local_wall_s=end-begin,stats=st,output_vs_first_OFF_NRMSE=change,cross_owner_output_NRMSE=metrics[item['index']][rank],proof_D2H_bytes=8264,proof_D2H_sync_calls=3)
    if row['calls']:assert st['owner_thread_hash']==row['calls'][0]['stats']['owner_thread_hash']
    row['calls'].append(record);journal(dict(rank=rank,**record));barrier.wait(90);return record
   def destroy(ctx):
    # Cleanup must still execute even if the final external-input canary fails.
    error=None
    try:
     if device_input is not None:assert torch.equal(device_input,device_initial)
    except BaseException as exc:error=exc
    check(f('destroy')(ctx));row['destroy_returned']=True
    if error is not None:raise error
   def shutdown():check(f('shutdown')());row['Runtime_expired']=True;row['normal_close']=bool(row.get('destroy_returned'))
   try:return driver_core.run_owner(rank,create,initialize,qualify,trial,destroy,shutdown,barrier.abort)
   except BaseException as exc:row.update(error=repr(exc),traceback=traceback.format_exc());save();raise
  save();trial_start=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:list(pool.map(owner,range(8)))
  report['all_owner_initialize_qualify_trial_close_s']=time.perf_counter()-trial_start;report['matched_queries']=[]
  for item in driver_core.plan():
   rows=[o['calls'][item['index']]for o in report['owners']];critical=max(range(8),key=lambda i:rows[i]['stats']['MoE_complete_s'])
   report['matched_queries'].append(dict(**item,max8_CPP_complete_s=rows[critical]['stats']['MoE_complete_s'],CPP_critical_owner=critical,max8_forward_CAPI_s=max(r['forward_CAPI_s']for r in rows),max8_CAPI_with_proof_s=max(r['CAPI_with_proof_s']for r in rows),same_trial_checked_batch_wall_s=max(r['measurement_end_s']for r in rows)-min(r['measurement_begin_s']for r in rows)))
  assert hashlib.sha256(hidden.view(torch.uint8).numpy().tobytes()).hexdigest()==hidden_sha
  assert all(hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()==payloads[k]for k,w in weights.items());report['CPU_source_weights_and_hidden_canary_unchanged']=True
  report.update(completed=True,passed=len(metrics)==10 and all(o['same_logits_qualified']for o in report['owners']),normal=all(o['normal_close']for o in report['owners']),final_providers=providers(),scope='One actual complete layer16 learned MoE TP8; observerOFF; same immutable input/weights. No prefix/cache/Model. CPU source weight canary and output controls do not read back every private GPU weight byte. Hash3 original branch only source/CPU fallback qualified.',timing_scope='CPP complete includes original expert/shared/TP and drain; forward_CAPI includes original external-input checks/producer sync. Proof CAPI includes 3 D2H+sync and input canary. Checked batch additionally includes barriers/CPU oracle/archive. All startup/metadata/checkpoint IO included in driver/allocation wall; no claimed Model multiplier.')
 except BaseException as exc:report.update(completed=True,passed=False,error=repr(exc),traceback=traceback.format_exc());raise
 finally:report['driver_wall_s']=time.monotonic()-started;save()
 print(json.dumps(dict(path=str(out),completed=report['completed'],passed=report['passed'],normal=report['normal'])))

if __name__=='__main__':main()
