import ctypes,pathlib,json,hashlib,os
def load(r):
 sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 receipt=r/'candidates/Kimi-speculative-batch-Model-r9122c/CPU-build.json';b=json.loads(receipt.read_text());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen'];library=pathlib.Path(b['native']);assert sha(library)==b['native_SHA256']
 for p,h in b['unchanged_object_SHA256'].items():assert sha(p)==h,p
 for name,h in b['source_SHA256'].items():assert sha(pathlib.Path(b['source_root'])/name)==h,name
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());assert any(v['task']=='multiply'and v['grouped']and len(v['generated_ids'][0])==15 and v['generated_ids'][0][-1]in chat['eos_token_ids']for v in known['cases'])
 lib=ctypes.CDLL(str(library));providers={}
 class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int
 names=['modern_kimi_engine_create_batch_r9122c','modern_kimi_I192_T_counts_r9129','modern_kimi_batch_reset_r9122','modern_kimi_batch_mode_r9122','modern_kimi_batch_prefill_r9122','modern_kimi_batch_forward_r9122','modern_kimi_batch_commit_r9122','modern_kimi_batch_replay_r9122','modern_kimi_batch_rows_r9122','modern_kimi_batch_stats_r9122','modern_kimi_batch_capabilities_r9122','modern_kimi_batch_taps_begin_r9122','modern_kimi_batch_taps_info_r9122','modern_kimi_batch_taps_count_r9122','modern_kimi_batch_taps_copy_r9122','modern_kimi_state_mode_r9114','modern_kimi_state_replay_r9114','modern_kimi_state_stats_r9114','modern_kimi_state_capabilities_r9114','modern_kimi_engine_create_pair_r9006','modern_kimi_pair_geometry_r9006','modern_kimi_pair_rows_r9006','modern_kimi_pair_prefill_r9006','modern_kimi_pair_forward_r9006','modern_kimi_pair_commit_r9006','modern_kimi_pair_reset_r9006','modern_kimi_pair_mode_r9007','modern_kimi_pair_ordinary_forward_r9007','modern_kimi_engine_forward_r2475','modern_kimi_engine_destroy_r2475','modern_kimi_I192_counts_r8669y','modern_kimi_I192_workspace_r8669zc','modern_kimi_vocab_head_set_r8341','modern_kimi_diag_capabilities_r9008','modern_kimi_diag_arm_r9008','modern_kimi_diag_read_r9008']
 for name in names:
  z=Dl();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(z));p=pathlib.Path(z.fname.decode()).resolve();assert p==library.resolve();providers[name]={'path':str(p),'SHA256':sha(p)}
 P=ctypes.c_void_p;S=ctypes.c_size_t;I=ctypes.c_int
 lib.modern_kimi_diag_capabilities_r9008.argtypes=[];lib.modern_kimi_diag_capabilities_r9008.restype=ctypes.c_char_p
 lib.modern_kimi_diag_arm_r9008.argtypes=[P,ctypes.c_char_p,I];lib.modern_kimi_diag_arm_r9008.restype=I
 lib.modern_kimi_diag_read_r9008.argtypes=[P];lib.modern_kimi_diag_read_r9008.restype=ctypes.c_char_p
 lib.modern_kimi_engine_create_batch_r9122c.argtypes=[ctypes.c_char_p,I,I,I,ctypes.c_char_p,I,S];lib.modern_kimi_engine_create_batch_r9122c.restype=P
 lib.modern_kimi_engine_create_pair_r9006.argtypes=[ctypes.c_char_p,I,I,I,ctypes.c_char_p,I,S];lib.modern_kimi_engine_create_pair_r9006.restype=P
 lib.modern_kimi_pair_geometry_r9006.argtypes=[S];lib.modern_kimi_pair_geometry_r9006.restype=ctypes.c_char_p
 lib.modern_kimi_pair_rows_r9006.argtypes=[P,I,P];lib.modern_kimi_pair_prefill_r9006.argtypes=[P,S]+[P]*6
 lib.modern_kimi_pair_forward_r9006.argtypes=[P,S,S,I]+[P]*11;lib.modern_kimi_pair_commit_r9006.argtypes=[P,I,S,I];lib.modern_kimi_pair_reset_r9006.argtypes=[P]
 lib.modern_kimi_pair_mode_r9007.argtypes=[P,I,P];lib.modern_kimi_pair_ordinary_forward_r9007.argtypes=[P,ctypes.c_uint64,S,S,I]+[P]*9
 assert json.loads(lib.modern_kimi_pair_geometry_r9006(8))['state_pool_rows']==17
 assert os.environ['INFINILM_KIMI_SPEC_R3632']=='1'and os.environ['INFINILM_KIMI_PAIR_TP16_R9006']=='1'and os.environ['INFINILM_KIMI_SPEC_MATH_R3676']=='0'
 c=lib.modern_kimi_I192_counts_r8669y;c.argtypes=[I,P,S]
 for device in range(8):
  v=(ctypes.c_uint64*2)();assert c(device,v,2)==0 and list(v)==[0,0]
 lib.modern_kimi_vocab_head_set_r8341.argtypes=[I];assert lib.modern_kimi_vocab_head_set_r8341(0)==0
 maps=pathlib.Path('/proc/self/maps').read_text().splitlines();expected={'libinfiniops.so':'6d5a6204f7eef70e84deb9cc737a36c6ae2c1fbc75a0f70645557c8441ed85ca','libinfiniccl.so':'76eb65adacc8a6832970991ca276b32562543b7fb7e8dd09cda9f4c269f24809','libinfinirt.so':'e78b7b4aed105e5968fc598f60a2cacafcaf7c07c13dd4a971abfbdd92d47560','libinfinilm_runtime.so':'04a6a1a86c97d176e4304b9bb4c1d6fcb8550784beaa47e0f5841fb7603ffabf'}
 for name,h in expected.items():
  paths={x.split()[-1]for x in maps if name in x};assert len(paths)==1 and sha(next(iter(paths)))==h,(name,paths)
 return library,lib,lib.modern_kimi_n4_tile_counts_r8167,{'CPU_step':b['producer'],'actualproviders':providers,'SDK_actual_SHA256':expected,'CPU_only_source_qualification':True}

def startup(lib):
 names=['modern_kimi_engine_create_batch_r9122c','modern_kimi_I192_T_counts_r9129','modern_kimi_batch_reset_r9122','modern_kimi_batch_mode_r9122','modern_kimi_batch_prefill_r9122','modern_kimi_batch_forward_r9122','modern_kimi_batch_commit_r9122','modern_kimi_batch_replay_r9122','modern_kimi_batch_rows_r9122','modern_kimi_batch_stats_r9122','modern_kimi_batch_capabilities_r9122','modern_kimi_batch_taps_begin_r9122','modern_kimi_batch_taps_count_r9122','modern_kimi_batch_taps_info_r9122','modern_kimi_batch_taps_copy_r9122','modern_kimi_batch_taps_capabilities_r9122']
 for name in names:assert hasattr(lib,name)
 lib.modern_kimi_batch_taps_capabilities_r9122.argtypes=[];lib.modern_kimi_batch_taps_capabilities_r9122.restype=ctypes.c_uint64;assert lib.modern_kimi_batch_taps_capabilities_r9122()==0x01200500
 return {'native_batch':True,'batch_feature_export':True,'GPU_qualification_pending':True,'old_B1_ABI_preserved':True,'max_forward_tokens':64,'max_requests':32,'max_width':8}
