"""One real global rank: complete78layer checkpoint/create/load/forward/reset/destroy."""
from pathlib import Path
import base64,ctypes as C,hashlib,json,os,socket,threading,time,traceback,re
from shared import D,R,OUT,NODES,LABEL,H,save,actual,verify,WORK_SECONDS
FLOOR=8*1024**3;VOCAB=154880
class DlInfo(C.Structure):_fields_=[('name',C.c_char_p),('base',C.c_void_p),('symbol',C.c_char_p),('address',C.c_void_p)]
def sockets():
 out=[]
 for f in Path('/proc/self/fd').iterdir():
  try:v=os.readlink(str(f))
  except OSError:continue
  if v.startswith('socket:['):out.append(v[8:-1])
 return sorted(set(out))
class Memory:
 def __init__(self,cuda,device):self.cuda=cuda;self.device=device;self.stop=threading.Event();self.lock=threading.Lock();self.phase='before_create';self.minimum=None;self.count=0;self.errors=[];self.byphase={};self.lowpoints=[];self.max_gap=0.;self.previous=None;self.thread=None
 def read(self):
  f=C.c_size_t();t=C.c_size_t();rc=self.cuda.cudaMemGetInfo(C.byref(f),C.byref(t));assert rc==0 and t.value==68702699520;return f.value,t.value
 def start(self):
  def work():
   try:
    assert self.cuda.cudaSetDevice(self.device)==0
    while not self.stop.is_set():
     free,total=self.read();now=time.monotonic()
     with self.lock:
      self.count+=1
      if self.previous is not None:self.max_gap=max(self.max_gap,now-self.previous)
      self.previous=now
      if self.minimum is None or free<self.minimum:
       self.minimum=free
       if len(self.lowpoints)<8192:self.lowpoints.append(dict(monotonic=now,free=free,phase=self.phase))
      rec=self.byphase.setdefault(self.phase,dict(samples=0,min_free=free));rec['samples']+=1;rec['min_free']=min(rec['min_free'],free)
     self.stop.wait(.005)
   except BaseException as e:self.errors.append(repr(e))
  self.thread=threading.Thread(target=work,name='bounded-inline-driver-sampler',daemon=True);self.thread.start()
 def summary(self):
  with self.lock:return dict(samples=self.count,min_observed=self.minimum,byphase=dict(self.byphase),lowpoints=list(self.lowpoints),max_gap_s=self.max_gap,errors=list(self.errors),target_s=.005,scope='Inline bounded sameownerprocess driver samples and API-boundary checks; no resetpeak/emptycache. Initialization/subinterval peaks UNKNOWN, not hardallocatorupperbound.')
 def close(self):
  self.stop.set()
  if self.thread:self.thread.join(2)
  if self.thread and self.thread.is_alive():self.errors.append('driver sampler join incomplete')
def main_owned(node,device,fd):
 from control import send,recv,identity,set_phase
 epoch=float(os.environ['GLM470_EPOCH']);rank=node*8+device;pipe=socket.socket(fileno=fd);ident=identity(epoch);target=OUT/('owner-%02d.json'%rank);local=Path('/local')/('r470-owner-state-%02d-%d.json'%(rank,os.getpid()));tracepath=Path('/local')/('r470-pipe-owner-%02d-%d.jsonl'%(rank,os.getpid()));trace=tracepath.open('xb',buffering=0);ctx=None;sampler=None
 q=dict(completed=False,passed=False,normal_close=False,Model_READY=False,weights=False,HTTP=0,fullModel_execution=True,Model_qualified=False,node=NODES[node],node_index=node,device=device,global_rank=rank,PID=os.getpid(),UID=os.getuid(),startticks=Path('/proc/self/stat').read_text().split()[21],owner_TID=threading.get_native_id(),actual_GPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],gpu_epoch=epoch,requests=[],memory_boundaries=[],Native_forward_attempts=0,Native_forward_completed=0,control_calls=0,resets=0,events=[])
 def deadline(label,cap=WORK_SECONDS):
  if time.time()>=epoch+cap:raise TimeoutError('Original epoch deadline '+label)
  if sampler and(sampler.errors or(sampler.minimum is not None and sampler.minimum<FLOOR)):raise RuntimeError('Observed original8GiB floor/sampler failure')
 def mark(phase,**extra):
  q['phase']=phase;event=dict(phase=phase,epoch=time.time(),monotonic=time.monotonic(),**extra);q['events'].append(event);trace.write((json.dumps(event)+'\n').encode());save(local,q)
  if phase not in ['load.progress','forward.enter','forward.return','sampled']:save(target,q)
 def memory(label):
  free,total=sampler.read();q['memory_boundaries'].append(dict(phase=label,epoch=time.time(),free=free,total=total,floor=FLOOR));assert free>=FLOOR,(label,'8GiB floor',free);deadline(label);return free
 def gate(phase):
  set_phase(phase);mark('barrier.enter',round=phase);record=dict(identity=ident,phase=phase,rank=rank,node=NODES[node],device=device,PID=q['PID'],startticks=q['startticks']);send(pipe,record,epoch);go=recv(pipe,epoch);assert set(go)=={'kind','identity','phase','digest'}and go['kind']=='GO'and go['identity']==ident and go['phase']==phase and re.fullmatch('[0-9a-f]{64}',go['digest']);mark('barrier.return',round=phase,digest=go['digest'])
 try:
  assert socket.gethostname().split('.')[0]==NODES[node]and os.environ['SLURM_JOB_ID']=='910371'and q['owner_TID']==q['PID'];common=sorted(os.sched_getaffinity(0));assert len(common)>=64;assigned=common[device*8:device*8+4];assert len(assigned)==4;os.sched_setaffinity(0,assigned);q['CPUmask']=sorted(os.sched_getaffinity(0));assert q['CPUmask']==assigned;m,cpu,inp,plan=verify(os.environ['GLM470_GPU_MF']);q.update(source_manifest_SHA256=m['source_manifest_SHA256'],GPU_manifest_SHA256=os.environ['GLM470_GPU_MF'],Native_SHA256=cpu['library_SHA256'],private_input_SHA256=inp['private_plan']['SHA256']);mark('verified_sources')
  import torch
  torch.set_num_threads(1);torch.cuda.set_device(device);assert torch.cuda.device_count()==8 and torch.cuda.current_device()==device
  rt=C.CDLL('/opt/dtk/cuda/cuda/lib64/libcudart.so.12')
  for name,args in [('cudaSetDevice',[C.c_int]),('cudaMemGetInfo',[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)]),('cudaDeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int])]:f=getattr(rt,name);f.argtypes=args;f.restype=C.c_int
  pci=C.create_string_buffer(64);assert rt.cudaDeviceGetPCIBusId(pci,64,device)==0;q['physical_PCI']=pci.value.decode().lower();sampler=Memory(rt,device);memory('before_create');sampler.start()
  lib=C.CDLL(str(actual(cpu['library'])));from typed_binding import GlobalTP16Context,attach
  api=attach(lib);dl=C.CDLL('libdl.so.2');dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];dl.dladdr.restype=C.c_int;q['CAPI_bindings']={}
  for name,f in api.items():
   info=DlInfo();assert dl.dladdr(C.cast(f,C.c_void_p),C.byref(info))and info.name;path=Path(info.name.decode()).resolve();assert H(path)==cpu['library_SHA256'];q['CAPI_bindings'][name]=dict(path=str(path),SHA256=H(path))
  maps={v.split()[-1]for v in Path('/proc/self/maps').read_text().splitlines()if'/'in v and v.split()[-1].startswith('/')};q['actual_provider_maps']={};bind=json.loads((D/'CPU_BINDING.json').read_bytes())
  for p,h in bind['actual_project_DSO_SHA256'].items():
   matches=[v for v in maps if Path(v).name==Path(p).name];assert matches and len({(os.stat(v).st_dev,os.stat(v).st_ino)for v in matches})==1 and all(H(v)==h for v in matches);q['actual_provider_maps'][p]=dict(SHA256=h,paths=matches)
  n=int(api['id_size']());assert 0<n<=4096
  if rank==0:
   ids=[C.create_string_buffer(n)for _ in range(2)]
   for a in ids:assert api['id'](a,n)==0
   data=[a.raw for a in ids];assert data[0]!=data[1];save(OUT/'UIDS.json',dict(actual_GPU=q['actual_GPU'],epoch=epoch,source=q['source_manifest_SHA256'],producer_PID=q['PID'],producer_startticks=q['startticks'],world=base64.b64encode(data[0]).decode(),tp=base64.b64encode(data[1]).decode()))
  uidfile=OUT/'UIDS.json'
  while True:
   deadline('originalUIDS',600)
   try:uid=json.loads(uidfile.read_bytes());break
   except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
  assert uid['actual_GPU']==q['actual_GPU']and uid['epoch']==epoch and uid['source']==q['source_manifest_SHA256'];data=[base64.b64decode(uid[k],validate=True)for k in ['world','tp']];assert len(data[0])==len(data[1])==n and data[0]!=data[1];q['UID_SHA256']=[hashlib.sha256(a).hexdigest()for a in data]
  cfg=json.loads((Path(inp['model_path'])/'config.json').read_bytes());cfg.update(glm_global_tp16_r470=True,glm_global_tp16_Model_r470=True,glm_vocab_parallel_embedding_r471=True);mark('create.enter');ctx=GlobalTP16Context(lib,cfg,node,device,*data);geo=ctx.geometry();assert geo['Model_constructed']and geo['cache_allocated']and not geo['weights_loaded']and not geo['Model_READY'];assert geo['replicated_embedding_BF16_bytes']==0 and geo['embedding_BF16_local_bytes']==118947840 and geo['embedding_vocab_begin']==rank*9680 and geo['embedding_vocab_end']==(rank+1)*9680;q['constructor_geometry']=geo;q['owned_socket_inodes']=sockets();memory('after_full_constructor');mark('create.return');gate('created')
  from checkpoint import load_checkpoint
  from safetensors import safe_open
  sampler.phase='full_weight_load';mark('load.enter');load_checkpoint(ctx,model_path=inp['model_path'],torch=torch,safe_open=safe_open,report=q,deadline=lambda label:deadline(label,10800),memory=memory,mark=mark);q.update(weights=True,Model_READY=True);q['owned_socket_inodes']=sorted(set(q['owned_socket_inodes']+sockets()));mark('load.return');gate('loaded');sampler.phase='full_forward'
  from scoring import grade
  decoder=None
  if rank==0:
   from tokenizers import Tokenizer
   decoder=Tokenizer.from_file(inp['tokenizer']['path'])
  eos=set(inp['eos_token_ids']);special=set(inp['tokenizer']['special_ids']);stream=torch.cuda.current_stream(device);logits=torch.empty((1,VOCAB),dtype=torch.bfloat16,device='cuda');message=torch.zeros(1,dtype=torch.float32,device='cuda')
  def forward(ids,past):
   deadline('before full78forward');tokens=torch.tensor(ids,dtype=torch.int64,device='cuda');positions=torch.arange(past,past+len(ids),dtype=torch.int64,device='cuda');q['Native_forward_attempts']+=1;mark('forward.enter',tokens=len(ids),past=past);begin=time.monotonic();ctx.forward(len(ids),past,tokens.data_ptr(),positions.data_ptr(),logits.data_ptr(),int(stream.cuda_stream));q['Native_forward_completed']+=1;assert bool(torch.isfinite(logits).all().item());memory('after_full78forward');elapsed=time.monotonic()-begin;mark('forward.return',tokens=len(ids),past=past,elapsed_s=elapsed);return elapsed
  for ordinal,row in enumerate(plan['requests']):
   assert row['max_new_tokens']==4096 and len(row['prompt_token_ids'])+4096<=32768;assert hashlib.sha256(json.dumps(row['prompt_token_ids'],separators=(',',':')).encode()).hexdigest()==row['prompt_IDs_SHA256'];assert row['body'].get('temperature')==0.0,'Original request must explicitly select greedy';deadline('request');ctx.reset();q['resets']+=1;memory('after_reset_before');request=dict(ordinal=row['ordinal'],case_id=row['case_id'],suite=row['suite'],prompt_tokens=len(row['prompt_token_ids']),prompt_IDs_SHA256=row['prompt_IDs_SHA256'],body_SHA256=row['body_SHA256'],planned_max_new_tokens=4096,reset_before=True,reset_after=False,generated_token_ids=[],sampled_ns=[],completed=False,prefill_CAPI_s=[],decode_CAPI_s=[],Native_forward_begin=q['Native_forward_completed']);q['requests'].append(request);mark('request.enter',ordinal=ordinal);begin=time.monotonic_ns();past=0
   for offset in range(0,len(row['prompt_token_ids']),4096):chunk=row['prompt_token_ids'][offset:offset+4096];request['prefill_CAPI_s'].append(forward(chunk,past));past+=len(chunk)
   for generated in range(4096):
    deadline('root sample and sameworld control');message.zero_()
    if rank==0:message.fill_(int(torch.argmax(logits[0]).item()))
    ctx.control(1,message.data_ptr(),int(stream.cuda_stream));q['control_calls']+=1;token=float(message.item());assert token.is_integer()and 0<=int(token)<VOCAB;token=int(token);request['generated_token_ids'].append(token);request['sampled_ns'].append(time.monotonic_ns())
    if token in eos:break
    if generated+1==4096:break
    request['decode_CAPI_s'].append(forward([token],past));past+=1
   ids=request['generated_token_ids'];natural=bool(ids and ids[-1]in eos);request.update(completed=True,natural_EOS=natural,cap_hit=not natural and len(ids)==4096,generated_token_ids_SHA256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),whole_backend_s=(time.monotonic_ns()-begin)/1e9,Native_forward_end=q['Native_forward_completed'],past_after=past);assert len(ids)>0;assert q['Native_forward_completed']-request['Native_forward_begin']==len(request['prefill_CAPI_s'])+len(ids)-1
   if rank==0:
    text=decoder.decode(ids,skip_special_tokens=True);request['output_text']=text;request['score']=grade(row['case'],text,natural_EOS=natural,cap_hit=request['cap_hit'],unexpected_tool=False)
   ticks=[t for token,t in zip(ids,request['sampled_ns'])if token not in eos and token not in special];request['visible_nonEOS_nonspecial']=len(ticks);request['backend_decode_tok_s']=(len(ticks)-1)*1e9/(ticks[-1]-ticks[0])if len(ticks)>1 and ticks[-1]>ticks[0]else None;request['backend_sampled_TTFT_s']=(ticks[0]-begin)/1e9 if ticks else None;request['clock_scope']='Sameownerprocess fullNative return/greedy/worldcontrol sampled timeline, not HTTP/client incrementalSLO; body/inputs frozen, no extra requests';ctx.reset();q['resets']+=1;request['reset_after']=True;memory('after_reset_after');mark('request.return',ordinal=ordinal,generated=len(ids),natural_EOS=natural,cap_hit=request['cap_hit']);gate('case'+str(ordinal))
  q['owned_socket_inodes']=sorted(set(q['owned_socket_inodes']+sockets()));gate('quiescent');sampler.phase='normal_destroy';mark('destroy.enter');ctx.close();ctx=None;q['normal_close']=True;q['Runtime_weak_expired_CABI_assertion']=True;memory('after_destroy');mark('destroy.return');gate('closed');q['passed']=True
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc(),normal_close=False if ctx is not None else q['normal_close'],failure_context_live=ctx is not None,failure_cleanup='No collective reentry/recreate/destroy on unsynchronizedfailure; ownSlurm failstop and independentpost required');raise
 finally:
  if sampler:sampler.close();q['driver_memory']=sampler.summary()
  q.update(completed=True,elapsed_s=time.time()-epoch,Model_READY=False,Model_qualified=False,whole_benchmark_qualified=False,client_SLO_qualified=False,worker_shutdown_completed=bool(q['passed']and q['normal_close']));save(local,q);save(target,q);trace.close();pipe.close()
