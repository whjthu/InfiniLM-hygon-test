"""One actual entry.main(--execute) integration, only Native/device/I/O leaves recorded."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import ctypes as C,hashlib,importlib,json,os,sys,tempfile,time
D=Path(__file__).resolve().parent;G=D.parents[3];P=Path(os.environ.get('DEEP467_PARENT_ROOT',str(G/'code/bw1000/Deep-HC-fullpre-public-pair-r449/source-v1')));BASE=Path(os.environ.get('DEEP467_BASE_PROVIDER_ROOT',str(G/'code/bw1000/public-Deep-native-provider-r427/source-v3')));PUBLIC=Path(os.environ.get('DEEP467_PUBLIC_ROOT',str(G/'code/bw1000/public-GLM-native-provider-r425')));NAT=Path(os.environ.get('DEEP467_NATIVE_SOURCE_ROOT',str(G/'code/bw1000/Deep-HC-fullpre-Native-r377/cpu-v1')));sys.path[:0]=[str(D),str(P),str(BASE)]
import entry,precision
sys.path.insert(0,str(D.parent/'profile-source-v1'))
from check_local_CPU import Context,Fn
from public_deep_model import worker,checkpoint_loader
import native_context
from hc_consume import LABELS
CTX=0x1234ab567890

def posix(p):
 s=Path(p).resolve().as_posix();return '/'+s[3:]if len(s)>2 and s[1:3]==':/'else s

def main(private_input):
 fake=Context();lib=fake.binding.lib;calls=[];refs=[]
 # The real parent declaration is consumed, and real libffi callbacks carry
 # the high64bit context across all new entry -> CapiTimer -> forward pointers.
 def create(raw,rows,capacity):
  assert(rows,capacity)==(1,32768);assert json.loads(raw)['num_hidden_layers']==43;calls.append('create');return CTX
 def reset(ctx):assert ctx==CTX;fake.reset();calls.append('reset');return 0
 def destroy(ctx):assert ctx==CTX;calls.append('destroy');return 0
 def forward(ctx,tokens,count,rows,past,total,offsets,ids,positions,logits,sampled):
  assert ctx==CTX and count==1
  r=list((C.c_int32*count).from_address(rows));p=list((C.c_int32*count).from_address(past));t=list((C.c_int32*count).from_address(total));o=list((C.c_int32*(count+1)).from_address(offsets));i=list((C.c_int64*tokens).from_address(ids));z=list((C.c_int64*tokens).from_address(positions))
  rc,_,s=fake.call(i,z,r,p,t,o);(C.c_uint16*(count*129280)).from_address(logits)[:]=[0]*(count*129280);(C.c_int64*count).from_address(sampled)[:]=s;calls.append('forward');return rc
 for n,restype,args,fn in [('create',C.c_void_p,[C.c_char_p,C.c_size_t,C.c_size_t],create),('reset',C.c_int,[C.c_void_p],reset),('destroy',C.c_int,[C.c_void_p],destroy),('forward',C.c_int,[C.c_void_p,C.c_size_t,C.c_size_t]+[C.c_void_p]*8,forward)]:
  cb=C.CFUNCTYPE(restype,*args)(fn);refs.append(cb);setattr(lib,'modern_native_engine_'+n,cb)
 for n in ['finish','rank_logits_equal']:setattr(lib,'modern_native_engine_'+n,Fn(lambda ctx:0))
 lib.modern_native_engine_error=Fn(lambda:b'recording Native error');lib.modern_native_engine_keys=Fn(lambda ctx:b'[]');lib.modern_native_engine_load=Fn(lambda ctx,key:0)
 hc=[[0]*3 for _ in range(8)];layout=[[0]*4 for _ in range(8)];tail=[[0]*5 for _ in range(8)]
 # Forward counters use the actual CAPI call packet, not optimistic constants.
 original_forward=lib.modern_native_engine_forward
 def forward_with_counts(ctx,tokens,count,*pointers):
  rc=original_forward(ctx,tokens,count,*pointers);past=list((C.c_int32*count).from_address(pointers[1]));target=count==1 and tokens==512 and all(x==0 for x in past)
  for d in range(8):hc[d][0 if tokens==512 else 2]+=86;layout[d][3]+=43;tail[d][3 if target else 4]+=43
  return rc
 cb=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_size_t,C.c_size_t,*([C.c_void_p]*8))(forward_with_counts);refs.append(cb);lib.modern_native_engine_forward=cb
 for n,f in dict(deep_hc_canonical_policy_r7130=lambda x:0,deep_hc_canonical_get_r7130=lambda:0,modern_deep_layout_get_r7052=lambda:0,deep_moe_tail_get_r7211=lambda:0,modern_deep_c4_tile_get_r5920=lambda:0,modern_deep_c4_tile_policy_r5920=lambda x:0,modern_deep_allselected_set_r813=lambda x:0,deep_output_tile_calls_r4808=lambda d:0,modern_deep_c4_tile_calls_r5920=lambda d,k:0,modern_deep_FP32_calls_r3597=lambda d:0,deep_hc_canonical_calls_r7130=lambda d,k:hc[d][k],deep_moe_tail_calls_r7211=lambda d,k:tail[d][k],modern_deep_layout_calls_r7052=lambda d,k:layout[d][k],modern_deep_precision_policy_r3443=lambda x:0).items():setattr(lib,n,Fn(f))
 def hcread():
  rows=[dict(device=d,layer=l,HC=h,owner=d+1,stream=d+11,thread=d+21,created=6,queried=6,destroyed=6,whole_stream_ms=5.,original_drain_host_ms=.1,spans=[dict(stage=i,stream_ms=1.,host_wall_ms=1.,thread_cpu_ms=.1)for i in range(5)])for d in range(8)for l in range(43)for h in range(2)]
  return json.dumps(dict(schema=1,profile='HCpre-r467',labels=LABELS,rows=rows)).encode()
 lib.deep_hc_pre_split_read_r467=Fn(hcread);lib.deep_c4_compact_bank_set_v1=Fn(lambda x:0);lib.deep_c4_compact_bank_stats_v1=Fn(lambda:json.dumps(dict(enabled=False,owners=[dict(device=d,calls=0,historical_rows_avoided=0)for d in range(8)])).encode())
 binding=checkpoint_loader.declare(lib)
 owner_report=dict(actual_owner_report=dict(owners=[dict(device=d,changes=1,verified_mask=','.join(str(d*4+i)for i in range(4)))for d in range(8)]))
 receipt=dict(library='/refactor/candidates/Deep-HCpre-phase-r467-CPU-v2/libDeep_HCpre_phase_r467.so',library_SHA256=entry.NATIVE,actual_DTK_bindings={})
 tree_calls=[];original_tree=entry.tree;parent_aliases={
 '/refactor/incoming/public-Deep-native-provider-r427/source-v3':BASE,
 '/refactor/incoming/public-GLM-native-provider-r425':PUBLIC,
 '/refactor/incoming/Deep-HC-fullpre-Native-r377/cpu-v1':NAT}
 def trees(root,pin):
  actual=parent_aliases.get(str(root),Path(root));tree_calls.append(dict(requested=str(root),actual=str(actual),expected=pin));return original_tree(actual,pin)
 def loader(binding,ctx,**kw):
  assert ctx==CTX;kw['report'].update(loaded_key_count=67569,loading_seconds=0.,raw_loader_dtype_codes={'scope':'RECORDING weight leaf; actual unchanged67569 loader source pinned'});calls.append('load')
 provider_calls=[]
 def provider(fn,path):provider_calls.append(path);assert path==receipt['library'];return dict(path=path,SHA256=entry.NATIVE,recording_ELF_leaf=True)
 cfgpath=Path(os.environ.get('DEEP467_MODEL_CONFIG',str(G/'code/bw1000/Deep-a314-C1-C4-cost-budget-r463/PINNED_METADATA.json')));cfgraw=json.loads(cfgpath.read_bytes());cfg=cfgraw.get('model',cfgraw);cfg['num_nextn_predict_layers']=1
 with tempfile.TemporaryDirectory()as temp,ExitStack()as stack:
  root=Path(temp);c=json.loads((D/'config.template.json').read_bytes());c.update(execute_authorized=True,parent_root=posix(P),base_provider_root=posix(BASE),public_package_root=posix(PUBLIC),input_path=posix(private_input),output_root=posix(root/'run'),actual_step='910374.recording',gpu_start_epoch=time.time())
  cf=root/'config.json';cf.write_text(json.dumps(c));original_hash=entry.H
  stack.enter_context(patch.object(entry,'tree',trees));stack.enter_context(patch.object(entry,'H',lambda p:c['Common_extension_SHA256']if str(p)==c['Common_extension']else original_hash(p)))
  stack.enter_context(patch.object(entry.socket,'gethostname',lambda:c['node']));stack.enter_context(patch.object(os,'sched_getaffinity',lambda _:set(range(64)),create=True));stack.enter_context(patch.dict(os.environ,{'SLURM_JOB_ID':'910374','SLURM_STEP_ID':'recording','INFINILM_OWNER_CPU_WIDTH_R2392':'4','INFINILM_DEEP_MOE_TAIL512_R7211':'0','INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750':'0'}))
  stack.enter_context(patch.object(worker,'verify_native_chain',lambda:(None,receipt,None)));stack.enter_context(patch.object(worker,'torch_setup',lambda:SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda i:(20*1024**3,64*1024**3)))));stack.enter_context(patch.object(worker,'safe_open_dependency',lambda:lambda *a,**k:(_ for _ in()).throw(AssertionError('weights must not open'))));stack.enter_context(patch.object(worker,'read_model_metadata',lambda p:(cfg,{})));stack.enter_context(patch.object(worker,'bind_library',lambda path,sha:binding))
  for n,f in [('verify_affinity',lambda lib:owner_report),('observe_threads',lambda lib:dict(status='OBSERVED',owners=[dict(device=d,TID=90000000+d)for d in range(8)])),('providers',lambda b,r:{k:r['library']for k in ['create','load','finish','forward','reset','destroy']}),('load_checkpoint',loader),('blas_snapshot',lambda:dict(failures=0,owners=[dict(tid=d+1,caller='libinfiniops.so',created=1,verified=1)for d in range(8)]))]:stack.enter_context(patch.object(native_context,n,f))
  stack.enter_context(patch.object(entry,'receipt_from_worker',lambda worker:receipt));stack.enter_context(patch.object(entry,'bind_library',lambda path,sha:binding));
  stack.enter_context(patch.object(entry,'verify_provider',provider));stack.enter_context(patch.object(precision,'verify_provider',provider))
  result=entry.main(['--config',str(cf),'--config-sha',original_hash(cf),'--source-sha',original_hash(D/'SOURCE_MANIFEST.json'),'--execute'])
  saved=json.loads((root/'run/entry-report.json').read_bytes());assert result==saved and result['passed']and result['normal_Native_shutdown']and result['observed_all8_threads_exited'];assert len(result['requests'])==5 and len(result['actual_single_CAPI_calls'])==530
  assert calls.count('create')==calls.count('load')==calls.count('destroy')==1 and calls.count('reset')==10 and calls.count('forward')==530;assert len(provider_calls)==8 and len(tree_calls)==5
  assert result['HC_dispatch_checked_forwards']==result['layout_dispatch_checked_forwards']==530;assert result['requests'][3]['profile_coverage_complete'];assert len(result['requests'][3]['profile_windows'])==2
  assert all(w['HC_pre_split']['rows']==688 for w in result['requests'][3]['profile_windows'])
  return dict(passed=True,new_HCpacket_handoff=True,HCrows_per_window=688,actual_entry_main_execute=True,actual_parent_tree_import_configure=True,actual_parent_NativeContext_enter_close=True,actual_CapiTimer_and_policy0_forward=True,actual_parent_generate_QuiescentMode=True,actual_finite_BF16_consumer=True,actual_source_config_descriptor_handoff=True,high64_ctx_libffi_roundtrip=hex(CTX),recording_create=1,recording_load=1,recording_destroy=1,recording_reset=10,recording_CAPI_forward=530,profile_samples=[8,87],tree_verifications=tree_calls,explicit_recording_leaves=['Native ffi/ELF','Torch device/memory','weights67569load','process/proc owner identity'],Native_dlopen=False,SDK_Runtime=False,weights_opened=False,GPU=False,Model_READY=False,new_jobs=0)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=main(a.input);Path(a.output).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
