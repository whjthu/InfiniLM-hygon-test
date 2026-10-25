import pathlib,json,hashlib,time,traceback,os,ctypes,gc,signal
signal.alarm(440)
import torch
from dflash2_strict_loader_r9007 import load_pinned_worker,ExternalTargetHead
torch.set_num_threads(1);r=pathlib.Path('/refactor');inc=r/'incoming';label='dflash2-trained-r9019';out=r/'results'/(label+'-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'.json');start=time.monotonic();d={'status':'RUNNING','level':'trained draft mechanics synthetic features','hardware':'BW1000','GPU_count':1,'target_acceptance_qualified':False,'spec_speedup_claim':False,'phases':[]};loaded=None;head=None;mapping=None
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
 dh=json.loads((inc/'actual-safetensors-header.json').read_text());t=dh['embed_tokens.weight'];targetSHA=filehash(shard,emeta['payload_offset']+emeta['data_offsets'][0],emeta['data_offsets'][1]-emeta['data_offsets'][0]);draftSHA=filehash(checkpoint,9816+t['data_offsets'][0],t['data_offsets'][1]-t['data_offsets'][0]);d['embedding_full_byte_SHA256']={'target':targetSHA,'draft':draftSHA,'equal':targetSHA==draftSHA,'no_second_GPU_embedding':True};mapping=None;save()
 model=loaded.worker.model;last={}
 def measured_head(x):
  value,wall=timed('head',lambda:head(x));last['logits']=value;last['head_wall_s']=wall;return value
 loaded.worker.head=measured_head
 def reference(features,anchor):
  length=features.shape[1];ctx=model.context_norm(model.context_proj(features));ids=torch.full((1,8),163592,dtype=torch.long,device='cuda');ids[0,0]=anchor;x=model.embed_tokens(ids);cp=torch.arange(length,device='cuda').view(1,-1);dp=torch.arange(length,length+8,device='cuda').view(1,-1);mask=(torch.abs(dp[:,:,None]-torch.arange(length+8,device='cuda')[None,None,:])<4096).unsqueeze(1)
  for i,layer in enumerate(model.layers):x=layer(x,ctx,dp,cp,mask if i<4 else None)
  hidden=model.final_norm(x);return hidden,measured_head(hidden[:,1:])
 def selected_ids(hidden,logits):
  prev=torch.tensor([163584],device='cuda');ids=[]
  for j in range(7):
   scores,pool=model.candidate_selector.score_candidates(hidden[:,j+1],logits[:,j],prev);prev=pool.gather(-1,scores.argmax(-1,keepdim=True))[:,0];ids.append(prev)
  return torch.stack(ids,1)
 torch.manual_seed(9019);features=torch.randn((1,263,35840),dtype=torch.bfloat16,device='cuda')*.1;phases=[]
 with torch.inference_mode():
  for iteration in range(4):
   if loaded.worker.positions is not None:loaded.worker.crop_context(0)
   _,append_s=timed('append',lambda:loaded.worker.append_context(features[:,:255],torch.arange(255,device='cuda').view(1,-1)))
   ref,ref_s=timed('full',lambda:reference(features[:,:255],163584));result,inc_s=timed('query',lambda:loaded.worker.speculate(163584));logits=last['logits'];stats=error(logits,ref[1]);assert stats['finite'];row={'phase':'warm'if iteration==0 else'measure','iteration':iteration,'T':255,'append_context_wall_s':append_s,'fullrecompute_wall_s':ref_s,'incremental_query_including_head_wall_s':inc_s,'head_wall_s':last['head_wall_s'],'whole_logits_error':stats,'hidden_error':error(result['hidden'],ref[0]),'tokens':result['tokens'].cpu().tolist(),'fullrecompute_selected_IDs_equal':bool(torch.equal(result['tokens'],selected_ids(*ref)))};phases.append(row);d['phases']=phases;save()
  repeat=loaded.worker.speculate(163584);d['ownrepeat']={'hidden_exact':bool(torch.equal(repeat['hidden'],result['hidden'])),'tokens_exact':bool(torch.equal(repeat['tokens'],result['tokens']))};assert all(d['ownrepeat'].values())
  _,append_s=timed('append8',lambda:loaded.worker.append_context(features[:,255:],torch.arange(255,263,device='cuda').view(1,-1)));d['append8_s']=append_s;loaded.worker.crop_context(258);candidate=loaded.worker.speculate(163584);candidate_logits=last['logits'];ref=reference(features[:,:258],163584);d['append_crop_logits']=error(candidate_logits,ref[1]);assert d['append_crop_logits']['finite'];loaded.worker.crop_context(255);replacement=features[:,255:258]*-.5;loaded.worker.append_context(replacement,torch.arange(255,258,device='cuda').view(1,-1));candidate=loaded.worker.speculate(163584);candidate_logits=last['logits'];ref=reference(torch.cat((features[:,:255],replacement),1),163584);d['reject_reappend_logits']=error(candidate_logits,ref[1]);assert d['reject_reappend_logits']['finite'];eos=loaded.worker.speculate(163586);assert eos['terminal']and eos['tokens'].numel()==0;d['EOS_no_query']=True
 d.update(status='TRAINED_DRAFT_MECHANICS_COMPLETED_NOT_TARGET_QUALIFIED',actual_peak_allocated=torch.cuda.max_memory_allocated(),T512='SKIPPED_no_extra_case_required',synthetic_features=True)
except BaseException as e:d.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
 d['shutdown_entered']=True;save()
 if loaded is not None:loaded.close()
 if head is not None:head.close()
 loaded=None;head=None;mapping=None
 for name in ['model','weight','features','replacement','candidate','ref','result','repeat','candidate_logits','logits']:
  globals().pop(name,None)
 last={};gc.collect()
 if torch.cuda.is_initialized():torch.cuda.synchronize();torch.cuda.empty_cache()
 d['shutdown_returncode']=0;d['elapsed_s']=time.monotonic()-start;save();print('DFLASH2_TRAINED_NORMAL',flush=True)
