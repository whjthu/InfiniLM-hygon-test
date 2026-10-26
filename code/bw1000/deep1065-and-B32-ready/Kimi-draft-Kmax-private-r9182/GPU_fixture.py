import pathlib,json,hashlib,time,traceback,os,ctypes,gc,signal
signal.alarm(820)
import sys
sys.path.append("/refactor/incoming")
import torch
from query_Kmax_r9182 import query_batched
import socket,threading
from batch_frames_r9142 import send_query,receive_bodies
from batch_wire_r9128 import send,recv
assert os.environ.get("KIMI_TRUE_DRAFT_BATCH_R9142")=="1"
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
torch.set_num_threads(1);r=pathlib.Path('/refactor');inc=r/'incoming';label='Kimi-draft-Kmax-fixture-r9182';out=r/'results'/(label+'-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');start=time.monotonic();d={'status':'RUNNING','level':'trained draft mechanics synthetic features','hardware':'BW1000','GPU_count':1,'target_acceptance_qualified':False,'spec_speedup_claim':False,'phases':[],'cases':[]};loaded=None;head=None;mapping=None
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

 baseline_sha=hashlib.sha256((inc/'Kimi-modern-PP4-cap32-r9145b/batched_query_r9142.py').read_bytes()).hexdigest()
 assert baseline_sha=='643a0f23182e9bf7b73a96f91db95cbbb49f9cbcd13c95b9d957ad67b6931505'
 d['baseline_source_SHA256']=baseline_sha;d['query_scope']='GPU-resident cached context query complete wall: pack/proj+norm/embed+masks/5layers/finalnorm/head/selector/D2H/pythonprefix; excludes appendH2D/network and targetverify';d['profile_extra_drains']=0;d['matrix_C']=[1,2,4,8,16,32];d['matrix_K']=[1,3,7];d['ABBA_blocks']=3;d['profile_cases']=[];torch.manual_seed(9182)
 def run_query(contexts,anchors,Ks,crop,profile=False):
  torch.cuda.synchronize();tick=time.perf_counter();tokens,debug=query_batched(model,head,contexts,anchors,Ks,enabled=True,return_debug=True,crop_head=crop,profile=profile);values=tokens.cpu().tolist();requests=[row[:K]for row,K in zip(values,Ks)];torch.cuda.synchronize();wall=time.perf_counter()-tick
  assert len(requests)==len(Ks)and all(len(row)==K for row,K in zip(requests,Ks));assert all(0<=v<163840 for row in requests for v in row)
  row={'mode':'Kmax'if crop else'old7','request_query_full_wall_s':wall,'head_positions':debug['head_positions'],'selector_steps':debug['selector_steps'],'fixed_transformer_positions':debug['fixed_transformer_positions'],'returned_shape':list(tokens.shape),'profile':profile}
  if profile:
   points=debug['event_points'];row['segments']=[{'from':a,'to':bb,'stream_elapsed_ms':float(e.elapsed_time(f))}for(a,e),(bb,f)in zip(points,points[1:])];row['total_stream_ms']=float(points[0][1].elapsed_time(points[-1][1]));row['wall_minus_stream_ms']=wall*1000-row['total_stream_ms'];row['stream_semantics']='elapsed includes queuedwait/launchgaps; notpurebusy; finalsync only, nosegmentdrain'
  return row,requests,debug['hidden']
 def one_case(C,Ks,label):
  lengths=[[512,384,256,128][i%4]for i in range(C)];contexts=[torch.randn((1,N,35840),device='cuda',dtype=torch.bfloat16)*.1 for N in lengths];anchors=[163584+i%2 for i in range(C)];versions=[x._version for x in contexts]
  for crop in [False,True]:run_query(contexts,anchors,Ks,crop)
  rows=[]
  for block in range(3):
   for order,crop in enumerate([False,True,True,False]):
    row,values,hidden=run_query(contexts,anchors,Ks,crop);row.update(block=block,ABBA_index=order);rows.append(row)
  profile=[];outputs=[];hiddens=[]
  for crop in [False,True]:
   row,values,hidden=run_query(contexts,anchors,Ks,crop,True);profile.append(row);outputs.append(values);hiddens.append(hidden.detach().clone());assert bool(torch.isfinite(hidden).all())
  diff=hiddens[1].float()-hiddens[0].float();RMS=float(diff.square().mean().sqrt()/hiddens[0].float().square().mean().sqrt().clamp_min(1e-20));agree=sum(x==y for a,b in zip(outputs[0],outputs[1])for x,y in zip(a,b))/sum(Ks);assert versions==[x._version for x in contexts]
  import statistics
  before=statistics.median(x['request_query_full_wall_s']for x in rows if x['mode']=='old7');after=statistics.median(x['request_query_full_wall_s']for x in rows if x['mode']=='Kmax');d['cases'].append({'label':label,'C':C,'Ks':Ks,'lengths':lengths,'padded_L':max(lengths),'rows':rows,'profile':profile,'old_median_s':before,'new_median_s':after,'latency_reduction_fraction':1-after/before,'query_rate_ratio':before/after,'trained_hidden_finite':True,'hidden_relativeRMS_diagnostic':RMS,'proposal_prefix_agreement_diagnostic':agree,'tokenidentity_required':False,'input_tensor_version_unchanged':True,'GPU_free_bytes':torch.cuda.mem_get_info()[0],'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()});save()
 with torch.inference_mode():
  for C in [1,2,4,8,16,32]:
   for K in [1,3,7]:one_case(C,[K]*C,'C%d-K%d'%(C,K))
  for C in [2,4,8,16,32]:one_case(C,[[1,3,7][i%3]for i in range(C)],'C%d-variableK'%C)
 d.update(status='TRAINED_KMAX_DRAFT_COMPONENT_COMPLETED',completed=True,synthetic_features=True,target_acceptance_qualified=False,spec_speedup_claim=False,fixture_query_cases=len(d['cases']),service_RPC_tested=False,Model_performance_round=False)
except BaseException as e:d.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
 if loaded is not None:loaded.close()
 if head is not None:head.close()
 d['normal_close']=True;save()
print('KIMI_DRAFT_KMAX_R9182_NORMAL',flush=True)
