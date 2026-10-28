"""One assigned GPU; registered operator components, no model or weights.

No GPU allocation/submission occurs here. Root must schedule this explicitly.
Fixed synthetic score/KV excludes Gemm/routing shape changes. A negative result
does not clear the F100 model path, projection, cache chain or rollback.
"""
import pathlib,json,hashlib,ctypes,os,time,argparse,math
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-C128-cache-component-r152';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def metadata(p,counts,swap=False):
 rows=list(enumerate(counts));rows=rows[::-1]if swap else rows
 return [(row,pos)for row,count in rows for pos in range(p,p+count)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--actual-GPU-GO',action='store_true');a=ap.parse_args();assert a.actual_GPU_GO,'Root GPU job only; not a CPU import test'
 cp=B/'CPU-build.json';c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and not c['Torch_imported'];m=json.loads((D/'manifest-r152.json').read_bytes());assert sha(D/'manifest-r152.json')==c['source_manifest_SHA256']
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
 import torch
 torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0)
 lib=ctypes.CDLL(c['library']);lib.modern_deep_compress_error.restype=ctypes.c_char_p;lib.modern_deep_public_cache_error.restype=ctypes.c_char_p
 lib.modern_deep_compress.argtypes=[ctypes.c_int]+[ctypes.c_size_t]*4+[ctypes.c_int]+[ctypes.c_void_p]*8
 lib.modern_deep_public_cache.argtypes=[ctypes.c_int]*2+[ctypes.c_size_t]*4+[ctypes.c_void_p]*4
 ptr=lambda x:ctypes.c_void_p(x.data_ptr());stream=lambda:ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
 expected=c['actual_DTK_bindings'];maps=pathlib.Path('/proc/self/maps').read_text().splitlines();loaded={pathlib.Path(s.split()[-1]).resolve()for s in maps if len(s.split())>=6 and s.split()[-1].startswith('/')}
 for p,h in expected.items():
  want=pathlib.Path(p).resolve();stem=want.name.split('.so')[0];actual={x for x in loaded if stem in x.name};assert actual=={want}and sha(want)==h['SHA256'],(p,[str(x)for x in actual])
 tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=R/'results'/('Deep-C128-cache-component-r152-'+tag+'.json');assert not out.exists()
 report={'completed':False,'actual_step':tag,'CPU_receipt_SHA256':sha(cp),'source_manifest_SHA256':sha(D/'manifest-r152.json'),'actual_SDK_SHA256':expected['/refactor/install/hygon-deep-5f91ad7-r3672/lib/libinfiniops.so']['SHA256'],'GPU_cards':1,'model_loaded':False,'weights_opened':False,'F100_replaced':False,'production_ac937_replaced':False,'model_quality':'UNQUALIFIED','TPS_gain':'UNMEASURED','MTP_head':False,'raw_activations_exported':False,'cases':[]};start=time.monotonic()
 def save():out.write_text(json.dumps(report,indent=2)+'\n')
 def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 def metrics(x,y):
  x=x.detach().cpu().float().reshape(-1);y=y.detach().cpu().float().reshape(-1);delta=x-y;den=y.square().mean().sqrt().clamp_min(1e-20)
  return {'finite':bool(torch.isfinite(x).all()and torch.isfinite(y).all()),'relative_RMS':float(delta.square().mean().sqrt()/den),'max_abs':float(delta.abs().max()),'cosine':float(torch.dot(x,y)/(x.norm()*y.norm()).clamp_min(1e-20)),'equal_value_count':int((x==y).sum()),'values':x.numel()}
 def scores(entries,delta=False):
  # Formula depends on logical row and position only, never packed T/order.
  host=[]
  for row,pos in entries:
   value=[math.sin((row*211+pos*17+j)%997/41.)*.2 for j in range(512)]
   gate=[math.cos((row*197+pos*13+j)%991/37.)*.1 for j in range(512)]
   if delta and row==0 and pos==128:value=[x+3. for x in value];gate=[x-.7 for x in gate]
   host.append(value+gate)
  return torch.tensor(host,dtype=torch.bfloat16,device='cuda')
 ape=torch.zeros((128,512),dtype=torch.float32,device='cuda')
 def compress(state,entries,delta=False,fill=123.):
  score=scores(entries,delta);pos=torch.tensor([p for row,p in entries],dtype=torch.int64,device='cuda');writes=torch.tensor([row for row,p in entries],dtype=torch.int64,device='cuda');extra=torch.zeros(len(entries),dtype=torch.int64,device='cuda');y=torch.full((len(entries),512),fill,dtype=torch.bfloat16,device='cuda');args=[0,len(entries),512,256,128,1]+[ptr(t)for t in [y,score,ape,state,pos,writes,extra]]+[stream()];code=lib.modern_deep_compress(*args)
  if code:raise RuntimeError(lib.modern_deep_compress_error().decode())
  assert torch.isfinite(y).all(),'Nonfinite component result';off=[i for i,(row,p)in enumerate(entries)if(p+1)%128];assert not off or bool((y[off]==0).all()),'Off-boundary output not zero';return y
 def c128(p,counts,bg=0.,delta=False,swap=False):
  state=torch.full((256,1024),bg,dtype=torch.float32,device='cuda');prefix=metadata(0,[p,p],swap);prefix_y=compress(state,prefix);entries=metadata(p,counts,swap);y=compress(state,entries,delta);boundary={str(row)+':'+str(pos):prefix_y[i].clone()for i,(row,pos)in enumerate(prefix)if(pos+1)%128==0};boundary.update({str(row)+':'+str(pos):y[i].clone()for i,(row,pos)in enumerate(entries)if(pos+1)%128==0});return boundary,state,torch.cat([prefix_y,y],dim=0)
 try:
  # First, exact packed shape and logical inputs repeated; then change only
  # finite unwritten background or only a future score at the SAME T4 shape.
  for p in [126,127,128]:
   for counts in [[1,2],[2,2]]:
    row={'kind':'C128_matched_shape','position':p,'counts':counts,'packed_T':sum(counts),'compared_outputs_include_prefix':True,'prefix_tokens':2*p,'step_tokens':sum(counts),'repeats':[]};report['cases'].append(row)
    ref,refstate,refout=c128(p,counts)
    for k in range(5):
     vals,state,y=c128(p,counts);row['repeats'].append({'iteration':k,'output':metrics(y,refout),'state':metrics(state,refstate)})
    poison,ps,po=c128(p,counts,bg=97.);row['unwritten_finite_background']={'backgrounds':[0.,97.],'boundary_output':{key:metrics(poison[key],ref[key])for key in ref},'all_output':metrics(po,refout)}
    perm,_,_=c128(p,counts,swap=True);row['whole_logical_row_permutation']={key:metrics(perm[key],ref[key])for key in ref}
    if counts==[2,2]and p==127:
     future,_,_=c128(p,counts,delta=True);row['same_T4_changed_future_score']={key:metrics(future[key],ref[key])for key in ref};row['future_change_scope']='Only row0 position128 value/gate changed; row0 position127 boundary must not read it.'
    row['boundary_keys']=list(ref);save()
  # Store/gather uses real registered cache op and byte masks. Storage
  # ownership is checked exactly; this is not an output-token equality gate.
  def cache_call(mode,raw,slots,values):
   ids=torch.tensor(slots,dtype=torch.int64,device='cuda');code=lib.modern_deep_public_cache(mode,0,len(slots),1,256,raw.shape[1],ptr(raw),ptr(ids),ptr(values),stream())
   if code:raise RuntimeError(lib.modern_deep_public_cache_error().decode())
  def kv(entries,delta=False):return scores(entries,delta)[:,:512].contiguous()
  stride=((256*584+575)//576)*576
  def cache_case(p,counts,bg,delta=False):
   raw=torch.full((1,stride),bg,dtype=torch.uint8,device='cuda');pre=metadata(0,[p,p]);slots=[row*128+pos%128 for row,pos in pre];cache_call(0,raw,slots,kv(pre));snapshot=torch.empty((len(slots),512),dtype=torch.bfloat16,device='cuda');cache_call(1,raw,slots,snapshot);snapshot_digest=digest(snapshot);before=raw.clone();entries=metadata(p,counts);writes=[row*128+pos%128 for row,pos in entries];cache_call(0,raw,writes,kv(entries,delta));allowed=torch.zeros(stride,dtype=torch.bool)
   for slot in writes:allowed[slot*576:(slot+1)*576]=True;allowed[256*576+slot*8:256*576+slot*8+7]=True
   changed=(raw.cpu().reshape(-1)!=before.cpu().reshape(-1));outside=int((changed&~allowed).sum());assert outside==0,'Cache store modified bytes outside requested slots';assert digest(snapshot)==snapshot_digest,'Independent gathered old snapshot overwritten';targets=[127,255];read=torch.empty((2,512),dtype=torch.bfloat16,device='cuda');cache_call(1,raw,targets,read);assert torch.isfinite(read).all();return read,{'bytes_changed_outside_write_slots':outside,'old_gathered_snapshot_retained':True,'targets':targets,'snapshot_SHA256':snapshot_digest}
  for p in [127,128]:
   for counts in [[1,2],[2,2]]:
    ref,meta=cache_case(p,counts,0);other,_=cache_case(p,counts,165);row={'kind':'PagedCache_slot_ownership','position':p,'counts':counts,'metadata':meta,'unwritten_payload_background':metrics(other,ref),'repeats':[]};report['cases'].append(row)
    for k in range(5):got,_=cache_case(p,counts,0);row['repeats'].append(metrics(got,ref))
    if p==127 and counts==[2,2]:future,_=cache_case(p,counts,0,True);row['same_T4_changed_other_slot']=metrics(future,ref)
    save()
  report.update(completed=True,normal_component_shutdown=True,scope='Synthetic matched-shape component attribution only; negative result does not clear F100 model chain or prove task quality.')
 except BaseException as exc:
  import traceback
  report.update(error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  a=lib.modern_deep_compress_shutdown();b=lib.modern_deep_public_cache_shutdown();report['normal_component_shutdown']=a==b==0;report['elapsed_s']=time.monotonic()-start;save()
 print('DEEP_COMPONENT_R152_NORMAL',json.dumps({'completed':report['completed'],'normal_close':report['normal_component_shutdown']}),flush=True)
if __name__=='__main__':main()
