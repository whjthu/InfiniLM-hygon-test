import pathlib,json,hashlib,time,traceback,os,ctypes,gc,signal
signal.alarm(650)
import sys
sys.path.append("/refactor/incoming")
import torch
from batched_query_r9142 import query_batched
import socket,threading
from batch_frames_r9142 import send_query,receive_bodies
from batch_wire_r9128 import send,recv
assert os.environ.get("KIMI_TRUE_DRAFT_BATCH_R9142")=="1"
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
torch.set_num_threads(1);r=pathlib.Path('/refactor');inc=r/'incoming';label='Kimi-draft-truebatch-fixture-r9142';out=r/'results'/(label+'-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');start=time.monotonic();d={'status':'RUNNING','level':'trained draft mechanics synthetic features','hardware':'BW1000','GPU_count':1,'target_acceptance_qualified':False,'spec_speedup_claim':False,'phases':[],'cases':[]};loaded=None;head=None;mapping=None
def save():
 t=out.with_suffix('.tmp');t.write_bytes((json.dumps(d,indent=2)+'\n').encode());t.replace(out)
def timed(name,fn):
 torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();return v,time.perf_counter()-t
def filehash(path,offset,length):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb')as f:
  f.seek(offset)
  while length:
   b=f.read(min(length,8<<20));assert b;h.update(b);length-=len(b)
 return h.hexdigest()
def error(a,b):
 aa=a.float();bb=b.float();return {'finite':bool(torch.isfinite(aa).all()and torch.isfinite(bb).all()),'maxabs':float((aa-bb).abs().max()),'relativeRMS':float(((aa-bb).square().mean().sqrt()/bb.square().mean().sqrt().clamp_min(1e-20))),'argmax_equal':bool(torch.equal(a.argmax(-1),b.argmax(-1)))}
try:
 m=json.loads((inc/'DFlash2-trained-manifest-r9019.json').read_text());assert all(hashlib.sha256((inc/n).read_bytes()).hexdigest()==h for n,h in m['incoming_source_SHA256'].items());meta=json.loads((inc/'DFlash2-target-metadata-r9019.json').read_text());receipt=json.loads((r/'results/DFlash2-CDN-detached-r9018.json').read_text());assert receipt['completed']and receipt['bytes']==7385765464 and receipt['prefix_SHA256']=='1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2'
 assert torch.cuda.device_count()==1;torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();d.update(torch_version=torch.__version__,HIP_version=torch.version.hip,initial_memory={'free':free,'total':total});assert free>16<<30
 class Dl(ctypes.Structure):_fields_=[('name',ctypes.c_char_p),('base',ctypes.c_void_p),('sym',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 lib=ctypes.CDLL('libc.so.6');lib.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];sym=ctypes.CDLL(torch._C.__file__);info=Dl();assert lib.dladdr(ctypes.cast(sym.PyInit__C,ctypes.c_void_p),ctypes.byref(info));d['Torch_actual_dladdr']=info.name.decode();d['Torch_C_SHA256']=hashlib.sha256(pathlib.Path(torch._C.__file__).read_bytes()).hexdigest();save()
 checkpoint=r/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors';hmeta=meta['tensors']['language_model.lm_head.weight'];emeta=meta['tensors']['language_model.model.embed_tokens.weight'];shard=pathlib.Path(hmeta['shard'].replace(str(pathlib.Path('/public/home/aclsylqidf/shared')),'/shared'));assert hashlib.sha256((pathlib.Path('/shared/Kimi-K3/config.json')).read_bytes()).hexdigest()==meta['config_SHA256']
 with shard.open('rb')as f:n=int.from_bytes(f.read(8),'little');raw=f.read(n);assert hashlib.sha256(raw).hexdigest()==hmeta['header_SHA256']
 mapping=torch.from_file(str(shard),shared=False,size=shard.stat().st_size,dtype=torch.uint8);a,b=hmeta['data_offsets'];weight=mapping.narrow(0,hmeta['payload_offset']+a,b-a).view(torch.bfloat16).reshape(hmeta['shape']).to('cuda:0');head=ExternalTargetHead(weight,values_compatibility=None,proof={'actual_target_key':'language_model.lm_head.weight','shard_header_SHA256':hmeta['header_SHA256']})
 loaded,load_s=timed('load',lambda:load_pinned_worker(checkpoint,head,enabled=True,device='cuda:0',source_dir=inc));d['load_s']=load_s;d['loader_receipt']=loaded.receipt
 model=loaded.worker.model
 def serial_reference(contexts,anchors):
  vals=[];hiddenlist=[]
  for features,anchor in zip(contexts,anchors):
   N=features.shape[1];ctx=model.context_norm(model.context_proj(features));ids=torch.full((1,8),163592,dtype=torch.long,device='cuda');ids[0,0]=anchor;x=model.embed_tokens(ids);cp=torch.arange(N,device='cuda').view(1,-1);dp=torch.arange(N,N+8,device='cuda').view(1,-1);mask=(torch.abs(dp[:,:,None]-torch.arange(N+8,device='cuda')[None,None,:])<4096).unsqueeze(1)
   for i,layer in enumerate(model.layers):x=layer(x,ctx,dp,cp,mask if i<4 else None)
   hidden=model.final_norm(x);logits=head(hidden[:,1:]);prev=torch.tensor([anchor],device='cuda');props=[]
   for j in range(7):
    scores,candidates=model.candidate_selector.score_candidates(hidden[:,j+1],logits[:,j],prev);prev=candidates.gather(-1,scores.argmax(-1,keepdim=True))[:,0];props.append(prev.clone())
   vals.append(torch.stack(props,1));hiddenlist.append(hidden)
  return torch.cat(vals),torch.cat(hiddenlist)
 # Head owner remains mainthread; client performs only binarysocket IO.
 server=socket.socket();server.bind(('127.0.0.1',0));server.listen(1);address=server.getsockname();specs=[];torch.manual_seed(9142)
 for C in [1,2,4,8,16,32]:
  items=[{'request':i+1,'past':0,'rows':[31,64,127,255][i%4],'anchor':163584+i%2,'K':[1,3,7][i%3]}for i in range(C)];bodies=[(torch.randn((q['rows'],35840),dtype=torch.bfloat16)*.1).view(torch.uint8).numpy().tobytes()for q in items];specs.append((C,items,bodies))
 client_timings=[];errors=[]
 def caller():
  try:
   c=socket.create_connection(address);c.settimeout(600);seq=0
   for C,items,bodies in specs:
    for pair in range(3):
     for mode in (['serial','batch']if pair%2==0 else['batch','serial']):
      tick=time.perf_counter();send_query(c,{'cmd':'query_batch','seq':seq,'epoch':1,'requests':items,'mode':mode,'pair':pair},bodies);h,payload=recv(c);assert h['seq']==seq and not payload;client_timings.append({'C':C,'mode':mode,'pair':pair,'RPC_s':time.perf_counter()-tick,'worker_full_s':h['worker_full_s']});seq+=1
   send(c,{'cmd':'close','seq':seq});h,_=recv(c);assert h['closed'];c.close()
  except BaseException as e:errors.append(repr(e))
 client=threading.Thread(target=caller,daemon=True);client.start();conn,_=server.accept();conn.settimeout(600);baseline={}
 with torch.inference_mode():
  while True:
   m,payload=recv(conn)
   if m['cmd']=='close':send(conn,{'closed':True,'seq':m['seq']});break
   tick=time.perf_counter();blobs=receive_bodies(conn,m);items=m['requests'];contexts=[torch.frombuffer(bytearray(b),dtype=torch.bfloat16).reshape(1,q['rows'],35840).to('cuda')for q,b in zip(items,blobs)];anchors=[q['anchor']for q in items];Ks=[q['K']for q in items];C=len(items)
   if m['mode']=='serial':
    (tokens,hidden),wall=timed('serial',lambda:serial_reference(contexts,anchors));baseline[C]=(tokens.detach().clone(),hidden.detach().clone());diag=None
   else:
    (tokens,debug),wall=timed('batch',lambda:query_batched(model,head,contexts,anchors,Ks,enabled=True,return_debug=True));hidden=debug['hidden'];ref=baseline.get(C);diag=None
    if ref is not None:
     diff=hidden.float()-ref[1].float();diag={'hidden_maxabs':float(diff.abs().max()),'hidden_relativeRMS':float(diff.square().mean().sqrt()/ref[1].float().square().mean().sqrt().clamp_min(1e-20)),'proposal_agreement_fraction_diagnostic':float((tokens==ref[0]).float().mean()),'not_tokenexact_gate':True}
   assert bool(torch.isfinite(hidden).all());free,total=torch.cuda.mem_get_info();row={'C':C,'mode':m['mode'],'pair':m['pair'],'lengths':[q['rows']for q in items],'query_model_s':wall,'finite':True,'numeric_diagnostic':diag,'GPU_free_bytes':free,'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'actual_Torch_model_batch_C':C if m['mode']=='batch' else 1,'serial_forward_calls':C if m['mode']=='serial' else 0,'worker_full_s':time.perf_counter()-tick};d['cases'].append(row);send(conn,{'seq':m['seq'],'worker_full_s':row['worker_full_s']});save()
 conn.close();server.close();client.join(5);assert not errors and not client.is_alive();d.update(status='TRAINED_BATCH_DRAFT_COMPONENT_COMPLETED',completed=True,client_RPC_timings=client_timings,RPC_transport='actual loopbackTCP, allbinarytransport/receiveH2D/query charged; notinterhost bandwidth',synthetic_features=True,target_acceptance_qualified=False,spec_speedup_claim=False)
except BaseException as e:d.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
 if loaded is not None:loaded.close()
 if head is not None:head.close()
 d['normal_close']=True;save()
print('KIMI_TRUEBATCH_R9142_NORMAL',flush=True)
