from model_affinity_check_r2628 import verify as verify_owner_affinity
import ctypes,hashlib,json,os,pathlib,re,time
import numpy as np
import resource
import torch
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
from deep_affinity_env_contract_r7166 import validate
assert validate(os.environ)['requested_width']==4
torch.set_num_threads(1)
assert torch.cuda.device_count()==8
torch.cuda.set_device(0)
r=pathlib.Path('/refactor');model=pathlib.Path(os.environ.get('DEEP_MODEL_VIEW_R3616','/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8'));tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
path=r/'results'/f'deep-activation-model-r8848b-{tag}.json';assert not path.exists()
for name,h in {'/refactor/results/deep-attention-r3509-910406.102.json': '91c3d2891266bb9cdaea933bd5eefdd2e239103c1740b6bac475421c391d10c9', '/refactor/results/native-engine-allselected-r816-910373.442.json': '100cce801a5b239a503bd67447a04ada63db34d52c4aa10e08589b71b7c43c41', '/refactor/results/deep-attention-r3511-B8-audit.json': 'bcf82e89560bfd85aea43a113a15cf568e369f5101fd555304b948830c898851', '/refactor/control/deep-chat-routed-r20-910373.162.json': '69ea973b8d20bee5f33e7efbc031fbe2857f8ad844a4a6418969d06450a69ef5', '/refactor/results/modern-deepseek-chat-routed-r20-910373.162-rank0.json': '92633ddb0f61a2a8e5c2c08d25af72f49f30c0d71d6cbaa6d7a3f8d46395e92f'}.items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==h,('historical_reference_changed',name)
assert os.environ['INFINILM_DEEP_MOE_TAIL512_R7211']=='0';assert os.environ['INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750']=='0'
manifest=json.loads((r/'incoming/Deep-activation-Model-manifest-r8848b.json').read_text());cpu_receipt_path=r/manifest['cpu_receipt_path'];assert hashlib.sha256(cpu_receipt_path.read_bytes()).hexdigest()==manifest['existing_remote_SHA256'][manifest['cpu_receipt_path']]
prebuild=json.loads(cpu_receipt_path.read_text());assert prebuild['completed'] and prebuild['libraries']['full']['SHA256']==manifest['DSO_SHA256']
assert hashlib.sha256(pathlib.Path(prebuild['libraries']['full']['path']).read_bytes()).hexdigest()==prebuild['libraries']['full']['SHA256']
for n,h in prebuild['link']['full']['objects'].items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==h
for n,v in prebuild['actual_DTK_bindings'].items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==v['SHA256']
# Sealed build receipt retains compile-input provenance; reused DSO requires no current header equality.
lib=ctypes.CDLL(prebuild['libraries']['full']['path']);lib.modern_native_engine_error.argtypes=[];lib.modern_native_engine_error.restype=ctypes.c_char_p
class DeepDlInfo(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
dla=ctypes.CDLL(None).dladdr;dla.argtypes=[ctypes.c_void_p,ctypes.POINTER(DeepDlInfo)];dla.restype=ctypes.c_int
for symbol in ['deep_model_activation_counts_r8846','modern_native_engine_forward','deep_prefill_pack_counts_r8767','deep_moe_tail_calls_r7211']:
 info=DeepDlInfo();assert dla(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info));actual=pathlib.Path(info.fname.decode()).resolve();assert actual==pathlib.Path(prebuild['libraries']['full']['path']).resolve();assert hashlib.sha256(actual.read_bytes()).hexdigest()==prebuild['libraries']['full']['SHA256']
assert not any('/libdeep_tailSUM_full_r7212.so'in line or '/libdeep_prefill_pack_r8768.so'in line for line in pathlib.Path('/proc/self/maps').read_text().splitlines())
lib.modern_deep_c4_tile_policy_r5920.argtypes=[ctypes.c_int];lib.modern_deep_c4_tile_policy_r5920.restype=ctypes.c_int
lib.modern_deep_c4_tile_get_r5920.argtypes=[];lib.modern_deep_c4_tile_get_r5920.restype=ctypes.c_int
lib.modern_deep_c4_tile_calls_r5920.argtypes=[ctypes.c_int]*2;lib.modern_deep_c4_tile_calls_r5920.restype=ctypes.c_uint64
lib.modern_deep_FP32_calls_r3597.argtypes=[ctypes.c_int];lib.modern_deep_FP32_calls_r3597.restype=ctypes.c_uint64
assert lib.modern_deep_c4_tile_get_r5920()==0 and lib.modern_deep_c4_tile_policy_r5920(1)==1
assert lib.modern_deep_c4_tile_policy_r5920(16)==0 and lib.modern_deep_c4_tile_get_r5920()==16
assert lib.modern_deep_c4_tile_policy_r5920(0)==0 and lib.modern_deep_c4_tile_get_r5920()==0
assert all(lib.modern_deep_c4_tile_calls_r5920(d,k)==0 for d in range(8)for k in range(2))
artifact=pathlib.Path(prebuild['libraries']['full']['path']);artifact_raw=artifact.read_bytes();CPU_build=prebuild
assert hashlib.sha256(artifact_raw).hexdigest()==CPU_build['libraries']['full']['SHA256'] and CPU_build['completed']
assert all(n.encode()+b'\0'in artifact_raw for n in ['INFINILM_DEEP_OUTPUT_TILE_R4808','INFINILM_DEEP_LAYER_TRACE_R4026'])
assert b'DEEP_C4_TILE_R5825\0'not in artifact_raw and b'DEEP_CSA_CAPTURE_R5800\0'not in artifact_raw
lib.modern_deepseek_sinkhorn_r89_counts.argtypes=[ctypes.c_void_p]
lib.deep_output_tile_calls_r4808.argtypes=[ctypes.c_int];lib.deep_output_tile_calls_r4808.restype=ctypes.c_uint64
for n in ['modern_deep_latent_fused_calls_r681','modern_deep_latent_fallback_calls_r681']:
 f=getattr(lib,n);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_uint64
lib.modern_deep_allselected_set_r813.argtypes=[ctypes.c_int];assert lib.modern_deep_allselected_set_r813(99)==1 and lib.modern_deep_allselected_set_r813(1)==0
lib.modern_deep_allselected_calls_r813.argtypes=[ctypes.c_int]*2;lib.modern_deep_allselected_calls_r813.restype=ctypes.c_uint64
mapped=sorted({line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfiniops.so'in line})
assert len(mapped)==1 and'/install/hygon-deep-5f91ad7-r3672/lib/'in mapped[0]
assert hashlib.sha256(pathlib.Path(mapped[0]).read_bytes()).hexdigest()==json.loads((r/'results/latest-sdk-r3672-Deep-build.json').read_text())['library_sha256']
lib.modern_native_engine_create.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_size_t];lib.modern_native_engine_create.restype=ctypes.c_void_p
lib.modern_native_engine_keys.argtypes=[ctypes.c_void_p];lib.modern_native_engine_keys.restype=ctypes.c_char_p
for name in ['finish','reset','destroy','rank_logits_equal']:
 f=getattr(lib,'modern_native_engine_'+name);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_int
lib.modern_native_engine_load.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.modern_native_engine_load.restype=ctypes.c_int
lib.modern_native_engine_forward.argtypes=[ctypes.c_void_p]+[ctypes.c_size_t]*2+[ctypes.c_void_p]*8;lib.modern_native_engine_forward.restype=ctypes.c_int
lib.deep_hc_canonical_policy_r7130.argtypes=[ctypes.c_int];lib.deep_hc_canonical_policy_r7130.restype=ctypes.c_int;lib.deep_hc_canonical_get_r7130.argtypes=[];lib.deep_hc_canonical_get_r7130.restype=ctypes.c_int;lib.deep_hc_canonical_calls_r7130.argtypes=[ctypes.c_int]*2;lib.deep_hc_canonical_calls_r7130.restype=ctypes.c_uint64
assert lib.deep_hc_canonical_get_r7130()==0
for name,args,result in [('modern_deep_layout_policy_r7052',[ctypes.c_int],ctypes.c_int),('modern_deep_layout_get_r7052',[],ctypes.c_int),('modern_deep_layout_calls_r7052',[ctypes.c_int]*2,ctypes.c_uint64)]:
 f=getattr(lib,name);f.argtypes=args;f.restype=result
assert lib.modern_deep_layout_get_r7052()==0
layout_mode=0
tail_SUM_mode=0
for name,args,result in [('deep_moe_tail_get_r7211',[],ctypes.c_int),('deep_moe_tail_set_r7211',[ctypes.c_int],ctypes.c_int),('deep_moe_tail_calls_r7211',[ctypes.c_int]*2,ctypes.c_uint64)]:
 f=getattr(lib,name);f.argtypes=args;f.restype=result
assert lib.deep_moe_tail_get_r7211()==0
policy_lib=ctypes.CDLL(None);policy_lib.modern_blas_policy_r243_json.restype=ctypes.c_char_p
def policy_snapshot():
 value=json.loads(policy_lib.modern_blas_policy_r243_json().decode())
 assert value['failures']==0 and all(v['created']==v['verified']for v in value['owners'])
 assert len({v['tid']for v in value['owners']if'libinfiniops.so'in v['caller']})==8,('MissingownedBLASpolicyhandles',value)
 return value

def check(code):
 if code:raise RuntimeError(lib.modern_native_engine_error().decode())
class MallInfo2(ctypes.Structure):
 _fields_=[(name,ctypes.c_size_t)for name in ['arena','ordblks','smblks','hblks','hblkhd','usmblks','fsmblks','uordblks','fordblks','keepcost']]
host_libc=ctypes.CDLL(None);host_libc.mallinfo2.restype=MallInfo2
def memory_snapshot(label):
 heap=host_libc.mallinfo2();values={v.split(':',1)[0]:v.split(':',1)[1].strip()for v in pathlib.Path('/proc/self/status').read_text().splitlines()if v.startswith(('VmRSS:','VmHWM:','VmSize:'))}
 counts=(ctypes.c_size_t*8)();assert lib.modern_deepseek_sinkhorn_r89_counts(counts)==0
 values.update(label=label,ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,active_heap_bytes=int(heap.uordblks+heap.hblkhd),sinkhorn_calls_by_device=list(counts))
 print('NATIVE_ENGINE_MHC_R89_MEMORY',json.dumps(values),flush=True);return values

assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='fd2ad6f5dbd77e55a49e9b2ff29c89030fbe4b538e8870c183d194648767be09'
assert hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest()=='7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a'
cfg=json.loads((model/'config.json').read_text());mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];start=time.perf_counter()
for required in['modern_Deep_B64_M32_config_r8650.py','Deep_activation_Model_protocol_r8848b.py','Deep_B64_M32_r8650_inside.sh']:assert(r/'incoming'/required).is_file(),required
for required in['results/deep-attention-r3511-B8-audit.json']:assert(r/required).is_file(),required
# Plain Python entry; no hipprof wrapper and layer trace environment unset by shell.
ctx=lib.modern_native_engine_create(json.dumps(cfg).encode(),128,1024)
if not ctx:raise RuntimeError(lib.modern_native_engine_error().decode())
report={'completed':False,'step':tag,'weight_source_path':str(model)}
try:
 keys=set(json.loads(lib.modern_native_engine_keys(ctx)));assert len(keys)==67569 and keys<=set(mapping)
 report={'fused_sinkhorn_variant':1,'fused_scope':'Only mHC normalization is fused: pairwise4 row sums, sequential4 column sums, FP32 division. Actual all86 B4 parameter sets byte-exact to original on8 workers; complete model prefill/quality/performance and host-memory behavior require this run.','scope':'Actual normal C++ InferEngine/RankWorker/ModelFactory with8 owned worker threads on8DCUs; typed native sparse model/cache adapter. True raw weights loaded once in Python and broadcast by Engine loader. Three reference-framed known-answer cases repeated2, sampled IDs/logits compared to previously qualified native model. This is not public Python/HTTP, broad quality, per-request cache recycling or peakTPS qualification.','step':tag,'main_layers':43,'keys':len(keys),'cases':[],'invalid_checks':[],'completed':False}
 report['fused_scope']='Actual814 Modelcaller + CPU197bank/kernel reused. A acceptedPACKON, B parkedleaseON, C B+fusedactivation. Oneweightload primaryA/C/C/A full128. Bundlegain only;defaultOFF,originalRuntime/quant2/epilogue/stream/drain/Modelshared/decode unchanged.'
 report['sdk_sha256']=hashlib.sha256(pathlib.Path(mapped[0]).read_bytes()).hexdigest()
 report['library_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();assert report['library_sha256']==CPU_build['libraries']['full']['SHA256']
 report['C4_CPU_build_receipt_sha256']=hashlib.sha256((r/'results/deep-c4-projection-model-r5920-build.json').read_bytes()).hexdigest();report['C4_defaultOFF_CPU_gate']='parent5920gate unchanged; newHCdefaultOFFcheckedactualDSO';report['cache_rows']=128;report['cache_capacity']=1024;report['GPU_memory_after_create']=[list(torch.cuda.mem_get_info(k))for k in range(8)];assert all(f>8*1024**3 for f,t in report['GPU_memory_after_create']),('loading_reserve_8GiB',report['GPU_memory_after_create'])
 report['owner_affinity_before_weight_load']=verify_owner_affinity(lib)
 report['weight_source_path']=str(model);report['LM_source_commit']='5f91ad738583db8f5617aeae53b76523f85d1ec6'
 grouped={}
 for key in keys:grouped.setdefault(mapping[key],[]).append(key)
 pending=[];pending_bytes=loaded=0
 def flush():
  global pending_bytes,loaded
  if not pending:return
  descriptors=[{'name':name,'pointer':value.data_ptr(),'shape':list(value.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.int8:2,torch.int64:3}[value.dtype]}for name,value in pending]
  check(lib.modern_native_engine_load(ctx,json.dumps(descriptors).encode()));loaded+=len(pending);pending.clear();pending_bytes=0
  print('NATIVE_ENGINE_LOAD',loaded,len(keys),time.perf_counter()-start,flush=True)
 for filename,names in sorted(grouped.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as f:
   for name in sorted(names):
    value=f.get_tensor(name).contiguous();pending.append((name,value));pending_bytes+=value.numel()*value.element_size()
    if len(pending)>=1024 or pending_bytes>=512*1024*1024:flush()
   flush()
 loaded_paths={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfinilm_runtime.so'in v or'_infinilm.cpython-'in v};assert any('/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/'in x and'libinfinilm_runtime.so'in x for x in loaded_paths),loaded_paths;assert all('/candidates/rebased-5f91ad7-r3668/Common/Deep/InfiniLM/'in x for x in loaded_paths),loaded_paths;report['actual_new_runtime_paths']=sorted(loaded_paths);assert [hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()for p in loaded_paths if p.endswith('libinfinilm_runtime.so')]==[next(v['SHA256']for n,v in prebuild['actual_DTK_bindings'].items()if n.endswith('/libinfinilm_runtime.so'))]
 check(lib.modern_native_engine_finish(ctx));report['owner_affinity_after_weight_load']=verify_owner_affinity(lib);report['phase_memory']=[memory_snapshot('weights_ready')];report['loading_seconds']=time.perf_counter()-start;path.write_text(json.dumps(report,indent=2));print('NATIVE_ENGINE_WEIGHTS_READY',loaded,report['loading_seconds'],flush=True)
 payload=json.loads((r/'control/deep-chat-routed-r20-910373.162.json').read_text());baseline=json.loads((r/'results/modern-deepseek-chat-routed-r20-910373.162-rank0.json').read_text())
 tokenizer=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True)
 HC_mode=0
 projection_tile_mode=0
 c4_tile_mode=0
 precision_mode=5
 csa_layers=sum(int(v)==4 for v in cfg["compress_ratios"][:cfg["num_hidden_layers"]])
 def forward(ids,positions,rows,past,total,offsets):
  assert projection_tile_mode in [0,16] and 0<len(rows)<=128
  assert len(offsets)==len(rows)+1 and offsets[0]==0 and offsets[-1]==len(ids)
  widths=[b-a for a,b in zip(offsets,offsets[1:])]
  decode=all(w==1 for w in widths) and all(p>0 for p in past)
  assert all(t==p+w for t,p,w in zip(total,past,widths))
  tile_enabled=projection_tile_mode==16 and decode and 16<len(rows)<=64
  os.environ['INFINILM_DEEP_OUTPUT_TILE_R4808']='16'if tile_enabled else'0'
  tile_before=[int(lib.deep_output_tile_calls_r4808(k))for k in range(8)]
  assert c4_tile_mode in[0,16]and precision_mode==5
  check(lib.modern_deep_c4_tile_policy_r5920(c4_tile_mode));assert lib.modern_deep_c4_tile_get_r5920()==c4_tile_mode
  c4_enabled=c4_tile_mode==16 and precision_mode==5 and decode and 16<len(rows)<=64
  c4_before=[[int(lib.modern_deep_c4_tile_calls_r5920(d,k))for k in range(2)]for d in range(8)]
  FP32_before=[int(lib.modern_deep_FP32_calls_r3597(d))for d in range(8)]
  hc_policy=lib.deep_hc_canonical_get_r7130();assert hc_policy==HC_mode
  hc_before=[[int(lib.deep_hc_canonical_calls_r7130(d,k)) for k in range(3)] for d in range(8)]
  assert lib.modern_deep_layout_get_r7052()==layout_mode
  assert lib.deep_moe_tail_get_r7211()==tail_SUM_mode
  tail_before=[[int(lib.deep_moe_tail_calls_r7211(d,k))for k in range(5)]for d in range(8)]
  layout_before=[[int(lib.modern_deep_layout_calls_r7052(d,k))for k in range(4)]for d in range(8)]
  values=[np.ascontiguousarray(x,dtype=np.int64 if i>=4 else np.int32)for i,x in enumerate([rows,past,total,offsets,ids,positions])]
  logits=np.full((len(rows),cfg['vocab_size']),0x7fc0,dtype=np.uint16);emitted=np.full(len(rows),-999,dtype=np.int64)
  code=lib.modern_native_engine_forward(ctx,len(ids),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in values],ctypes.c_void_p(logits.ctypes.data),ctypes.c_void_p(emitted.ctypes.data))
  if code==0:
   delta=[int(lib.deep_output_tile_calls_r4808(k))-v for k,v in enumerate(tile_before)]
   assert delta==[43 if tile_enabled else 0]*8,('All43projectiondispatchgate',tile_enabled,delta)
   c4_delta=[[int(lib.modern_deep_c4_tile_calls_r5920(d,k))-c4_before[d][k]for k in range(2)]for d in range(8)]
   assert c4_delta==[[2*csa_layers if c4_enabled else 0]*2 for d in range(8)],('C4_actual_same_image_path',c4_enabled,c4_delta)
   FP32_delta=[int(lib.modern_deep_FP32_calls_r3597(d))-FP32_before[d]for d in range(8)]
   assert FP32_delta==[43 if precision_mode==5 and len(rows)>1 else 0]*8,('FP32scope_actual',precision_mode,len(rows),FP32_delta)
   hc_after=[[int(lib.deep_hc_canonical_calls_r7130(d,k)) for k in range(3)] for d in range(8)];hc_delta=[[hc_after[d][k]-hc_before[d][k] for k in range(3)] for d in range(8)];hc_kind=(1 if HC_mode else 0) if len(ids)==512 else 2;assert hc_delta==[[86 if k==hc_kind else 0 for k in range(3)] for d in range(8)],('actualall86HCdispatch',HC_mode,len(ids),hc_delta)
   report.setdefault('HC_dispatch_checked_forwards',0);report['HC_dispatch_checked_forwards']+=1
   layout_after=[[int(lib.modern_deep_layout_calls_r7052(d,k))for k in range(4)]for d in range(8)];layout_delta=[[layout_after[d][k]-layout_before[d][k]for k in range(4)]for d in range(8)];eligible=all(p==0 for p in past)and len(set(widths))>1 and len(rows)*max(widths)<=4096;target=len(rows)==1 and len(ids)==512 and all(p==0 for p in past);tail_selected=bool(tail_SUM_mode and target);moe=int(bool(layout_mode&1)and eligible)or int(tail_selected);attn=int(bool(layout_mode&2)and len(rows)==1);expected_layout=[43*moe,43*(33554432 if tail_selected else len(rows)*max(widths)*4096*4 if moe else 0),43*attn,43*(1-moe)];assert layout_delta==[expected_layout]*8,('all43layoutdispatch',layout_mode,widths,past,layout_delta,expected_layout)
   report.setdefault('layout_dispatch_checked_forwards',0);report['layout_dispatch_checked_forwards']+=1
   tail_after=[[int(lib.deep_moe_tail_calls_r7211(d,k))for k in range(5)]for d in range(8)];tail_delta=[[tail_after[d][k]-tail_before[d][k]for k in range(5)]for d in range(8)];expected_tail=[43*int(tail_selected),43*33554432*int(tail_selected),43*8388608*int(tail_selected),43*int(target and not tail_selected),43*int(not target)];assert tail_delta==[expected_tail]*8,('all43strict_tailSUM_dispatch',tail_SUM_mode,rows,widths,past,tail_delta,expected_tail)
   key_tail=f'flag{tail_SUM_mode}-target{int(target)}-selected{int(tail_selected)}-rows{len(rows)}-T{len(ids)}';report.setdefault('tail_SUM_dispatch_checks',{});report['tail_SUM_dispatch_checks'][key_tail]=report['tail_SUM_dispatch_checks'].get(key_tail,0)+1

   key=f"{'decode' if decode else 'prefill'}-rows{len(rows)}-tile{16 if tile_enabled else 0}-c4{c4_tile_mode}-active{int(c4_enabled)}"
   stats=report.setdefault('dispatch_checks_r4842',{})
   stats[key]=stats.get(key,0)+1
  return code,logits,emitted
 chat_payload=payload
 new_scalar_references={}
 report['reference_numeric_policy']='Explicit human instruction: final full token IDs exact, logits bitwise differences diagnostic; finite, CPUargmax, EOS, ownership and cleanup retained.'
 def run_chat(epoch):
  saved_cases=report['cases'];report['cases']=[]
  for case in ([v for v in chat_payload['cases'] if v['label']=='retrieval'] if epoch=='diagnostic_B' else chat_payload['cases']):
   for repeat in ([1] if epoch=='diagnostic_B' else [1,2]):
    report.setdefault('phase_memory',[]).append(memory_snapshot('chat_before_reset'));check(lib.modern_native_engine_reset(ctx));report['phase_memory'].append(memory_snapshot('chat_after_reset'));phase=case['label']+'_repeat'+str(repeat);prompt=case['input_ids'];generated=[];hashes=[];times=[];begin=time.perf_counter()
    for step in range(256):
     ids=prompt if not step else[generated[-1]];past=0 if not step else len(prompt)+step-1;positions=list(range(past,past+len(ids)))
     tick=time.perf_counter();code,logits,emitted=forward(ids,positions,[0],[past],[past+len(ids)],[0,len(ids)]);check(code);times.append(time.perf_counter()-tick)
     values=torch.from_numpy(logits).view(torch.bfloat16);assert torch.isfinite(values).all()and int(values.argmax(-1).item())==int(emitted[0]),'Normal Engine greedy IDs differ from first-argmax semantics'
     if step<4:
      hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
      (r/'results'/f'deep-finite-r578-{tag}-{epoch}-{phase}-step{step}-B1-bf16.bin').write_bytes(logits.tobytes())
      old_case=next(c for c in baseline['cases']if c['phase']==phase)
      if step < len(old_case['first4_logits_sha256']) and hashes[-1]!=old_case['first4_logits_sha256'][step]:(r/'results'/f'native-engine-allselected-r815-{tag}-{phase}-step{step}-changed-bf16.bin').write_bytes(logits.tobytes())
      if step >= len(old_case['first4_logits_sha256']):report.setdefault('missing_historical_logit_hash_diagnostics',[]).append({'epoch':epoch,'phase':phase,'step':step,'hard_gate':False})
     generated.append(int(emitted[0]))
     if step==0:
      badcode,badlogits,badids=forward([cfg['vocab_size']],[len(prompt)],[0],[len(prompt)],[len(prompt)+1],[0,1]);assert badcode and(badlogits==0x7fc0).all()and(badids==-999).all();report['invalid_checks'].append({'phase':phase,'case':'out_of_vocabulary_after_prefill','rejected_before_output_and_cache_mutation':True})
     if generated[-1]==cfg['eos_token_id']:break
    old=next(c for c in baseline['cases']if c['phase']==phase)
    row={'epoch':epoch,'phase':phase,'generated_ids':generated,'first4_logits_sha256':hashes,'all8_worker_full_logits_exact_first4':None,'baseline_ids_exact':generated==old['generated_ids'],'baseline_first4_logits_exact':hashes==old['first4_logits_sha256'],'stopped_at_eos':generated[-1]==cfg['eos_token_id'],'answer':tokenizer.decode(generated,skip_special_tokens=True).strip(),'elapsed_seconds':time.perf_counter()-begin,'step_seconds':times}
    row['blas_policy']=policy_snapshot();report['cases'].append(row);path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print('NATIVE_ENGINE_CHAT_CASE',json.dumps(row,ensure_ascii=False),flush=True)
    value=generated;key=case['label']
    if key not in new_scalar_references:new_scalar_references[key]=value
    row['QA_quality_passed']=bool(row['baseline_ids_exact'] and row['stopped_at_eos'] and value==new_scalar_references[key])


  chats=report['cases'];assert len(chats)==(1 if epoch=='diagnostic_B' else 6);report['cases']=saved_cases;report.setdefault('chat_cases_by_epoch',{})[epoch]=chats
  if epoch=='after':
   before=report['chat_cases_by_epoch']['before'];report['QA_ON_vs_OFF_fullseq_exact']=all(x['generated_ids']==y['generated_ids']for x,y in zip(before,chats))
  frozen=json.loads((r/'results/native-engine-allselected-r816-910373.442.json').read_text());assert frozen['completed']and frozen['engine_worker_shutdown_completed']
  for actual in chats:
   gold=next(v for v in frozen['chat_cases_by_epoch']['before' if epoch=='before' else 'after']if v['phase']==actual['phase']);actual['frozen816_full_IDs_exact']=actual['generated_ids']==gold['generated_ids']
  report['QA816_reference_SHA256']=hashlib.sha256((r/'results/native-engine-allselected-r816-910373.442.json').read_bytes()).hexdigest()
  report['finite_'+epoch+'_passed']=all(c['QA_quality_passed'] and c['frozen816_full_IDs_exact'] for c in chats);path.write_text(json.dumps(report,indent=2))


 natural_path=r/'results'/f'deep-serving-cases-r2130-{tag}.json';natural=json.loads(natural_path.read_text());report['natural_input_sha256']=hashlib.sha256(natural_path.read_bytes()).hexdigest();report['natural_cases']=[]

 from Deep_activation_Model_protocol_r8848b import run
 run(globals());report['owner_affinity']=verify_owner_affinity(lib);report['blas_policy_before_destroy']=policy_snapshot();report['collection_finished']=True
except BaseException as failure:
 report['error']=repr(failure);raise
finally:
 try:
  check(lib.modern_native_engine_destroy(ctx));report['engine_worker_shutdown_completed']=True;report['completed']=bool(report.get('collection_finished'))
 except BaseException as close:report['completed']=False;report['shutdown_error']=repr(close);raise
 finally:path.write_text(json.dumps(report,indent=2))
print('DEEP_PACK_MODEL_R8774_COMPLETE',flush=True)
