"""Actual compiled 9001bf dualhead Model + dependency/provider gate, visible GPU context required."""
import pathlib,json,hashlib,ctypes,os
def load(r):
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 mp=r/'incoming/Kimi-head-Model-manifest-r8235.json';m=json.loads(mp.read_text());p=r/'results/Kimi-prefill512-CPU-r8564.json';assert sha(p)=='09ef77520046c2df72aea1c32392d31538ce5038b86ed5f67f1193247f774ebc';b=json.loads(p.read_text());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen'];assert b['parent_DSO_SHA256']=='dcd75bdbbf370de1d40fd602bba010d4ae10f5d91a9764c365c99d3eb7f1049d';library=pathlib.Path(b['library']);assert sha(library)==b['library_SHA256']=='e3e335857563db06e12e8a06cda8f8e149b8d46aebf45b7ee51a4dba1a554753'
 for unit in b['compiled_units']:
  for name,h in unit['deps'].items():assert sha(pathlib.Path(name))==h,name
 for name,h in b['all_linked_object_SHA256'].items():assert sha(pathlib.Path(name))==h,name
 for name,h in b['actual_dynamic_dependency_SHA256'].items():assert sha(pathlib.Path(name))==h,name
 assert os.environ.get('INFINILM_KIMI_PREFILL512_R8562')=='0'and os.environ.get('INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562')=='0'
 flags=m['ordinary_initial_flags']
 for name,value in flags.items():assert os.environ.get(name)==value,(name,os.environ.get(name),value)
 cfg=json.loads((pathlib.Path('/shared/Kimi-K3')/'config.json').read_text());assert cfg.get('dtype')=='bfloat16'and cfg['text_config'].get('dtype')=='bfloat16'and 'lm_head_output_dtype'not in cfg['text_config']
 index=json.loads(pathlib.Path('/shared/Kimi-K3/model.safetensors.index.json').read_text());key='language_model.lm_head.weight';import struct
 with(pathlib.Path('/shared/Kimi-K3')/index['weight_map'][key]).open('rb')as stream:n=struct.unpack('<Q',stream.read(8))[0];meta=json.loads(stream.read(n))[key]
 assert meta['dtype']=='BF16'and meta['shape']==[163840,7168]and meta['data_offsets'][1]-meta['data_offsets'][0]==2348810240
 lib=ctypes.CDLL(str(library))
 class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int
 names=['modern_kimi_engine_create_r2475','modern_kimi_engine_forward_r2475','modern_kimi_engine_destroy_r2475','modern_kimi_n4_tile_counts_r8167','modern_kimi_n4_tile_raw_r8167','modern_kimi_vocab_head_set_r8341','modern_kimi_vocab_head_get_r8341','modern_kimi_vocab_head_count_r8341']+[v.split()[-1]for v in b['nm_defined'].splitlines()if'KimiK3ForConditionalGenerationHeadR8341C1'in v]
 assert len(names)>=9;providers={}
 for name in names:
  d=Dl();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(d));actual=pathlib.Path(d.fname.decode()).resolve();assert actual==library.resolve(),(name,actual);providers[name]={'path':str(actual),'SHA256':sha(actual)}
 lines=pathlib.Path('/proc/self/maps').read_text().splitlines();maps={p for p in[v.split()[-1]for v in lines]if p.startswith('/')and('libmodern_kimi'in p.lower()or'libkimi_native_boundary'in p.lower()or'libkimi_prefill512'in p.lower())};assert maps=={str(library)},maps
 mapped={}
 for name,h in b['actual_dynamic_dependency_SHA256'].items():
  matches={v.split()[-1]for v in lines if v.split()[-1].startswith('/')and pathlib.Path(v.split()[-1]).resolve()==pathlib.Path(name).resolve()};assert len(matches)==1,(name,matches);actual=next(iter(matches));assert sha(pathlib.Path(actual))==h;mapped[actual]=h
 counter=lib.modern_kimi_n4_tile_counts_r8167;counter.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];counter.restype=ctypes.c_int
 for device in range(8):
  v=(ctypes.c_uint64*2)();assert counter(device,v,2)==0 and list(v)==[0,0]
 for device,n in[(-1,2),(8,2),(0,0),(0,1),(0,3)]:
  v=(ctypes.c_uint64*3)(*([0xfeed]*3));assert counter(device,v,n)!=0 and list(v)==[0xfeed]*3
 assert counter(0,None,2)!=0
 head_set=lib.modern_kimi_vocab_head_set_r8341;head_set.argtypes=[ctypes.c_int];head_set.restype=ctypes.c_int
 head_get=lib.modern_kimi_vocab_head_get_r8341;head_get.argtypes=[];head_get.restype=ctypes.c_int
 head_count=lib.modern_kimi_vocab_head_count_r8341;head_count.argtypes=[ctypes.c_int,ctypes.c_int];head_count.restype=ctypes.c_uint64
 assert head_get()==0 and head_set(0)==0
 assert all(head_count(d,f)==0 for d in range(8)for f in range(3))
 for invalid in [-1,2]:assert head_set(invalid)!=0 and head_get()==0
 for d,f in [(-1,0),(8,0),(0,-1),(0,3)]:assert head_count(d,f)==2**64-1
 for name,h in m['owner_fence_source_SHA256'].items():assert sha(pathlib.Path(name))==h,name
 return library,lib,counter,{'Model_SHA256':sha(library),'CPU_receipt_SHA256':sha(p),'actualproviders':providers,'one_Model_maps':sorted(maps),'actual_mapped_dependencies':mapped,'N4_counter_zero_and_invalid_no_write':True,'Spec11':'KNOWN_ODR_CONSTRUCTOR_FAILURE_NOT_TESTED_NO_SCOPE_CONSTRUCTION','head_default0_counterzero_invalidrejected':True,'owner_fence_source_SHA256':m['owner_fence_source_SHA256'],'metadata_head':meta,'not_thin_unit':True,'actual_initial_flags':flags}
