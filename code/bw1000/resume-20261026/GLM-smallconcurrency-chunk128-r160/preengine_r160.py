"""Visible-GPU job preengine audit; initializes GPU0 before DSO static FA. No Enginecreate or weights."""
import ast,ctypes,datetime,hashlib,json,os,pathlib,re,socket
r=pathlib.Path('/refactor')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate():
 import torch
 assert torch.cuda.device_count()==8,'requires visible8 only inside assigned GPU job'
 torch.cuda.set_device(0)
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_fp16_accumulation=False
 modes={'float32_precision':torch.get_float32_matmul_precision(),**{n:getattr(torch.backends.cuda.matmul,n)for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation']}};assert modes['float32_precision']=='highest'and not any(v for k,v in modes.items()if k!='float32_precision')
 manifest=json.loads((r/'results/glm-preengine-inputs-r5394.json').read_text())
 receipts={}
 for rel,h in manifest['inputs_sha256'].items():
  p=r/rel;assert sha(p)==h,('preengine hash',rel)
 for rel,paths in manifest['receipt_required_paths'].items():
  q=json.loads((r/rel).read_text());receipts[rel]=q
  for keys in paths:
   v=q
   for k in keys:v=v[k]
   assert v is not None,('missing receipt',rel,keys)
 proof=receipts['results/glm-hybrid-decode-model-r5390-build.json']
 assert proof['completed'] and proof['library_sha256']==manifest['model_library_sha256']
 assert sha(pathlib.Path(proof['library']))==proof['library_sha256']
 for p,h in proof['linked_objects_sha256'].items():assert sha(pathlib.Path(p))==h,p
 for p,h in proof['linked_sources_sha256'].items():assert sha(pathlib.Path(p))==h,p
 for p,h in proof['SDK_libraries_sha256'].items():assert sha(r/'install/hygon-glm-5f91ad7-r3672/lib'/p)==h,p
 for p,h in proof['fresh_helpers'].items():assert sha(r/'candidates/glm-grouped-native-model-r5117'/p)==h,p
 assert sha(r/'candidates/glm-hybrid-decode-r5386/libnative.so')==proof['native_library_sha256']
 native=json.loads((r/'candidates/glm-hybrid-decode-r5386/build.json').read_text());assert native['library_sha256']==proof['native_library_sha256']
 for n,h in native['linked_objects_sha256'].items():assert sha(r/'candidates/glm-hybrid-decode-r5386'/n)==h,n
 unique=receipts['results/glm-unique16-inputs-r5388.json'];assert unique['completed']and len(unique['cases'])==16 and len(set(tuple(c['input_ids'])for c in unique['cases']))==16
 assert unique['chat_template_provenance']==receipts['results/glm-chat-template-r20.json']['provenance']
 assert all(0<len(c['input_ids'])<=512 and len(c['input_ids'])+127<=640 and c['input_tokens']==len(c['input_ids'])and all(0<=v<154880 for v in c['input_ids'])for c in unique['cases'])
 assert sha(r/'candidates/glm-hybrid-decode-model-r5390/src/glm_moe_dsa_routed_ffn_r19.cpp')==proof['capture_source_sha256']
 assert sha(r/'incoming/glm_actual_capture_r5176.inc')==proof['capture_fragment_sha256']
 assert sha(r/'incoming/glm_actual_capture_patch_r5172.py')==proof['capture_patcher_sha256']
 assert sha(r/'candidates/glm-grouped-native-model-r5117/build.json')==proof['parent_build_proof_sha256']
 assert sha(r/'candidates/glm-deferred-attention-model-5f91ad7-r5096/objects-manifest.json')==proof['parent_objects_manifest_sha256']
 assert receipts['results/latest-sdk-r3672-GLM-build.json']['library_sha256']==proof['SDK_libraries_sha256']['libinfiniops.so']
 assert receipts['results/model-latest-r3675-GLM-build.json']['fresh_helpers']['helpers/libglm_fp8_batched_r776.so']==proof['fresh_helpers']['helpers/libglm_fp8_batched_r776.so']
 assert receipts['results/glm-balanced-micro8-r153-lock.json']['mode']in[0,2]
 assert receipts['results/glm-long-prompts-r21.json']['passed'] and receipts['results/glm-chat-template-r20.json']['passed']
 model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');cfg=json.loads((model/'config.json').read_text());index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
 assert cfg['num_hidden_layers']==78 and len(cfg['mlp_layer_types'])==78 and len(cfg['indexer_types'])==78 and index
 assert all((model/p).is_file()for p in set(index.values()))
 for stage in[0,3]:
  for rank in range(8):
   q=json.loads((r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{stage}-rank{rank}.json').read_text());assert q['completed']
   for epoch in['before_select','after_select']:assert len(q[epoch+'_chat_cases'])==6
 # Resolve every C symbol referenced anywhere in the immutable harness and its
 # original baseline ABI, including symbols whose argtypes are set in callbacks.
 from preflight_r160 import read_private_metadata
 model_meta=read_private_metadata()
 private=pathlib.Path(model_meta['library_actual_path']);lib=ctypes.CDLL(str(private));symbols=set()
 for rel in manifest['ABI_source_files']:
  text=(r/rel).read_text();ast.parse(text)
  symbols.update(re.findall(r'\blib\.([A-Za-z_]\w*)',text))
 for rel in ['modern_glm_prefill_micro_r9059.py','glm_prefill_micro_probe_r9059.py','glm_crossrequest_control_r9059.py','glm_layer77_policy_r9059.py','glm_window_control_r9059.py']:
  symbols.update(re.findall(r'\blib\.([A-Za-z_]\w*)',(r/'incoming'/rel).read_text()))
 symbols.update(['glm_ragged_policy_r9136','glm_ragged_enabled_r9136','glm_ragged_counts_r9136','glm_ragged_Bcounts_r9136','modern_glm_cross_control_r5775','modern_glm_cross_journal_r5775','modern_glm_cross_error_r5775','modern_glm_vector_policy_r8540','modern_glm_vector_get_r8540','modern_glm_vector_counts_r8540'])
 symbols.update(['modern_glm_sparse_prepare_r131','modern_glm_sparse_policy_r131','modern_glm_sparse_read_r131','modern_glm_model_m15_oproj_policy_r125','modern_glm_m15_oproj_stats_r125'])
 for name in sorted(symbols):assert getattr(lib,name),('missing CABI',name)
 # dladdr uses actual symbol owners, not just maps filenames.
 class DlInfo(ctypes.Structure):
  _fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.dladdr.restype=ctypes.c_int
 owners={}
 for sym,expect in [('modern_glm_sparse_prepare_r131',private),('modern_glm_sparse_policy_r131',private),('modern_glm_sparse_read_r131',private),('modern_glm_model_m15_oproj_policy_r125',private),('modern_glm_m15_oproj_stats_r125',private),('glm_ragged_policy_r9136',private),('glm_ragged_enabled_r9136',private),('glm_ragged_counts_r9136',private),('glm_ragged_Bcounts_r9136',private),('modern_glm_shared_validation_counts_r5814',private),('modern_glm_cross_control_r5775',private),('modern_glm_cross_journal_r5775',private),('modern_glm_crossrequest_prefill_calls_r7328',private),('modern_glm_model_create',private),('modern_glm_model_resident_mask_r5640',private),('modern_glm_model_resident_report_r5640',private),('modern_glm_vector_policy_r8540',private),('modern_glm_vector_counts_r8540',private),('glm_grouped_gate_r5123',r/'candidates/glm-hybrid-decode-r5386/libnative.so')]:
  info=DlInfo();assert dl.dladdr(ctypes.cast(getattr(lib,sym),ctypes.c_void_p),ctypes.byref(info))
  actual=pathlib.Path(info.fname.decode()).resolve();assert actual==expect.resolve(),(sym,str(actual),str(expect));owners[sym]={'file':str(actual),'SHA256':sha(actual)}
 mapped=sorted({str(pathlib.Path(v.split()[-1]).resolve())for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if v.split()and v.split()[-1].startswith('/')and '.so'in pathlib.Path(v.split()[-1]).name})
 assert str(private.resolve())in mapped and not any('glm-hybrid-decode-model-r5390'in p for p in mapped),mapped
 assert sha(private.resolve())==model_meta['library_SHA256']

 policy=ctypes.CDLL(None);policy.modern_blas_policy_r243_json.restype=ctypes.c_char_p
 policyq=json.loads(policy.modern_blas_policy_r243_json());assert policyq['failures']==0
 lib.glm_dedup_scope_r5386.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.glm_dedup_scope_get_r5386.argtypes=[ctypes.c_int]
 for device in range(8):assert lib.glm_dedup_scope_get_r5386(device)==0 and lib.glm_dedup_scope_r5386(device,0,None)==0
 # Test exported bounded capture ABI in disarmed/defaultOFF state. No GPU call.
 lib.glm_actual_capture_arm_r5164.argtypes=[ctypes.c_int,ctypes.c_char_p]
 previous=os.environ.pop('GLM_ACTUAL_CAPTURE_R5164',None)
 assert lib.glm_actual_capture_arm_r5164(0,b'')==0
 assert lib.glm_actual_capture_arm_r5164(0,b'/refactor/results/glm-actual-capture-r5164-preflight')!=0
 if previous is not None:os.environ['GLM_ACTUAL_CAPTURE_R5164']=previous
 assert os.environ['LD_PRELOAD']=='/refactor/candidates/blas-policy-r271/libblas_policy_r271.so'
 assert receipts['results/glm-balanced-micro8-r153-lock.json']['mode']==2
 assert os.environ['GLM_PP_SIZE_R3603']=='2' and os.environ['GLM_CACHE_ROWS_R3627']=='32' and os.environ['GLM_CACHE_CAP_R3627']=='640'
 assert os.environ['GLM_SHARED_PREFIX_VALIDATE_R5814']=='0'
 assert os.environ['GLM_ACTUAL_CAPTURE_R5164']=='0' and os.environ['USE_WEIGHT_CACHE_R3604']=='0'
 assert '/install/hygon-glm-5f91ad7-r3672/lib:'in os.environ['LD_LIBRARY_PATH']
 for d in['results','control','logs']:
  p=r/d/('glm-transient-preengine-r9183-'+socket.gethostname()+'-'+str(os.getpid())+'.tmp');p.write_bytes(b'preengine');assert p.read_bytes()==b'preengine';p.unlink()
 result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),'owner_rank':int(os.environ.get('GLM_PREFLIGHT_RANK_R5397','-1')),'matmul_modes_pin_readback':modes,'completed':True,'actual_cuda_device':torch.cuda.current_device(),'visible_GPU0_no_Enginecreate_no_weight_tensor_IO':True,'library_sha256':model_meta['library_SHA256'],'actual_symbol_owners':owners,'Model_mappings':mapped,'receipt_count':len(receipts),'all_receipt_required_paths_walked':True,'ABI_symbols_resolved':sorted(symbols),'source_manifest_sha256':sha(r/'results/glm-preengine-inputs-r5394.json'),'config_and_index_metadata_only':True,'output_directories_writable':True}
 out=r/'results'/('glm-transient-preengine-r9183-'+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']+'-'+socket.gethostname()+'-'+str(os.getpid())+'.json');out.write_text(json.dumps(result,indent=2));print('GLM_PREENGINE_R5194_OK',socket.gethostname(),len(symbols),flush=True)
 return {'receipts':receipts,'audit':result}
if __name__=='__main__':validate()
