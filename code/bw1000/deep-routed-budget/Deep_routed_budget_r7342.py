"""Single visible GPU, actual TP0 expertbanks and real prefill-row component."""
import pathlib,os,json,hashlib,ctypes,time,traceback,collections
import torch,numpy as np
from safetensors import safe_open
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];out=r/'results'/('Deep-routed-budget-r7342-'+tag+'.json');assert not out.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();hash_tensor=lambda t:hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
torch.set_num_threads(1);assert torch.cuda.device_count()==1;torch.cuda.set_device(0) # before any GPU DSO static constructors
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
receiptpath=r/'candidates/deep-routed-budget-r7340/CPU-build.json';receipt=json.loads(receiptpath.read_text());assert receipt['completed']and receipt['producer']=='910374.718';assert sha(receipt['library'])==receipt['library_SHA256']=='35138a938ad2541ed4f88310c530992ffca23e452e1a072bf3f51a9f1b007834'
for p,v in receipt['DTK_bindings'].items():assert sha(p)==v['SHA256']
lib=ctypes.CDLL(receipt['library']);P=ctypes.c_void_p;I=ctypes.c_int
lib.deep_routed_budget_create_r7338.argtypes=[I];lib.deep_routed_budget_create_r7338.restype=P
lib.deep_routed_budget_forward_r7338.argtypes=[P,I,ctypes.POINTER(P)];lib.deep_routed_budget_destroy_r7338.argtypes=[P,ctypes.POINTER(I)]
lib.deep_routed_budget_report_r7338.restype=ctypes.c_char_p;lib.deep_routed_budget_error_r7338.restype=ctypes.c_char_p
report={'producer':tag,'completed':False,'library_SHA256':receipt['library_SHA256'],'CPU_receipt_SHA256':sha(receiptpath),'calls':[],'scope':'Actual TP0 expertbanks and591e0 fourprefillprompts last8positions each,32realrows.Not699decode,not32users,noModel/Layer/CCL. Mode0originalroute;1samebodyobserverOFF;2observerON. ProfileOFF ordinarycalls andsampleevents recordedseparately.'}
props=torch.cuda.get_device_properties(0);report['hardware']={'device_name':torch.cuda.get_device_name(0),'visible_cards':torch.cuda.device_count(),'total_memory':props.total_memory,'gcnArchName':getattr(props,'gcnArchName',None),'node':os.environ.get('SLURMD_NODENAME')}
def save():
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(out)
def check(code):
 if code:raise RuntimeError(lib.deep_routed_budget_error_r7338().decode())
class DlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',P),('sname',ctypes.c_char_p),('saddr',P)]
dl=ctypes.CDLL(None);dl.dladdr.argtypes=[P,ctypes.POINTER(DlInfo)];dl.dladdr.restype=I
providers={}
for name in ['deep_routed_budget_forward_r7338','deep_routed_dot_r20','deep_routed_validate_r20']:
 info=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,name),P),ctypes.byref(info));p=pathlib.Path(info.fname.decode()).resolve();providers[name]={'path':str(p),'SHA256':sha(p)}
report['providers']=providers;assert providers['deep_routed_budget_forward_r7338']['SHA256']==receipt['library_SHA256']
assert all(providers[n]['SHA256']=='0e0512551483457af01944e9c9298442f499d959836a1c5b66dfa438a3c6c4ea'for n in ['deep_routed_dot_r20','deep_routed_validate_r20'])
started=time.perf_counter();ctx=None
try:
 meta=r/'results/deep-layer0-r7005-910374.591-owner0-B32.json';assert sha(meta)=='977be1e7bdaef015744cc6d3921a73d5408d059ec99b16ebd0f7f2f64e42b37e';event=json.loads(meta.read_text())['history'][0];assert event['rows']==[0,1,2,3];stages={v['name']:v for v in event['stages']};selected=[512*j+t for j in range(4)for t in range(504,512)];inputs=[];proof=[]
 for name,dtype in [('ffn_norm',np.uint16),('moe_indices',np.int64),('moe_weights',np.float32)]:
  s=stages[name];p=r/'results/deep-layer0-r7005-910374.591'/s['raw_file'];assert sha(p)==s['whole_SHA256'];a=np.fromfile(p,dtype=dtype).reshape(s['shape'])[selected].copy();t=torch.from_numpy(a).view(torch.bfloat16)if name=='ffn_norm'else torch.from_numpy(a);inputs.append(t.cuda(0));proof.append({'name':name,'source_SHA256':sha(p),'selected_SHA256':hash_tensor(t),'shape':list(t.shape)})
 report['input_proof']=proof;report['selected_prefill_tensor_rows']=selected;report['input_prepare_seconds']=time.perf_counter()-started
 model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];gold={v['key']:v['CPU_tensor_SHA256']for v in json.loads((r/'results/deep-actualprefix-r6994-910374.587-CPUweights.json').read_text())}
 banks=[torch.empty((256,512,4096),dtype=torch.int8),torch.empty((256,512,1),dtype=torch.float32),torch.empty((256,4096,256),dtype=torch.int8),torch.empty((256,4096,1),dtype=torch.float32)]
 groups=collections.defaultdict(list)
 for e in range(256):
  for w in ['w1','w3','w2']:
   for suffix in ['weight','scale']:
    key=f'layers.0.ffn.experts.{e}.{w}.{suffix}';groups[mapping[key]].append((key,e,w,suffix))
 weights_begin=time.perf_counter();weightproof=[];full_source_bytes=0
 for filename,keys in sorted(groups.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as f:
   for key,e,w,suffix in keys:
    t=f.get_tensor(key).contiguous();h=hash_tensor(t);assert h==gold[key],key;full_source_bytes+=t.numel()*t.element_size()
    if w in['w1','w3']:
     offset=0 if w=='w1' else 256;bank=banks[0 if suffix=='weight' else 1];bank[e,offset:offset+256].copy_(t[:256])
    elif suffix=='weight':banks[2][e].copy_(t[:,:256])
    else:banks[3][e].copy_(t)
    weightproof.append({'key':key,'SHA256':h})
 report['CPU_checkpoint_read_hash_slice_seconds']=time.perf_counter()-weights_begin;report['full_source_tensor_bytes_read_hashed']=full_source_bytes;report['weight_keys']=len(weightproof);report['source_weight_manifest_SHA256']=hashlib.sha256(json.dumps(weightproof,sort_keys=True).encode()).hexdigest();report['TP0_bank_SHA256']=[hash_tensor(t)for t in banks];report['CPU_bank_bytes']=sum(t.numel()*t.element_size()for t in banks)
 h2d=time.perf_counter();gpu_banks=[t.cuda(0)for t in banks];del banks;torch.cuda.synchronize(0);report['GPU_bank_H2D_seconds']=time.perf_counter()-h2d;report['GPU_memory_after_weights']=list(torch.cuda.mem_get_info(0));output=torch.empty((32,4096),device='cuda',dtype=torch.float32);tensors=[*inputs,*gpu_banks,output];ptrs=(P*8)(*[t.data_ptr()for t in tensors]);ctx=lib.deep_routed_budget_create_r7338(0);assert ctx,lib.deep_routed_budget_error_r7338().decode();reference=None
 plan=[('warm0',0),('warm1',1),('warm2',2)]+[(f'ordinary{j}',mode)for j,mode in enumerate([0,1,1,0]*4)]+[(f'observed{j}',2)for j in range(3)]+[('restore0',0),('restore1',1)]
 report['plan']=plan;save()
 for label,mode in plan:
  before=time.perf_counter();check(lib.deep_routed_budget_forward_r7338(ctx,mode,ptrs));callwall=time.perf_counter()-before;native=json.loads(lib.deep_routed_budget_report_r7338().decode());a=output.cpu().numpy();assert np.isfinite(a).all()
  if reference is None:reference=a.copy()
  assert np.array_equal(a,reference),(label,'wholeF32routeoutput_mismatch');native.update(label=label,complete_CAPI_wall_seconds=callwall,full_output_exact=True,output_SHA256=hashlib.sha256(a.tobytes()).hexdigest());report['calls'].append(native);save()
 report['completed']=True
except BaseException as e:report.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 if ctx:
  expired=I();report['destroy_code']=lib.deep_routed_budget_destroy_r7338(ctx,ctypes.byref(expired));report['Runtime_expired']=expired.value
 report['driver_wall_includes_loading_CPU_validation_seconds']=time.perf_counter()-started;save()
print(json.dumps({'completed':report['completed'],'path':str(out),'SHA256':sha(out)}));assert report['destroy_code']==0 and report['Runtime_expired']==1
