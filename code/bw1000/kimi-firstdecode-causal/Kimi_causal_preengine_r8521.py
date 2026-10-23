"""Four-node visible-device ABI/source/load gate, no Engine or weight IO."""
import ast,ctypes,hashlib,json,os,pathlib
assert os.environ.get('INFINILM_OWNER_CPU_WIDTH_R2392')=='4',('owner_CPU_width_early',os.environ.get('INFINILM_OWNER_CPU_WIDTH_R2392'))
import torch,infinilm
assert torch.cuda.device_count()==8
torch.cuda.set_device(0)  # GPU-visible preengine; establish device context before actual DSO dlopen.
from Kimi_capacity_contract_r8152 import validate_source,retained_rows,GEOMETRY
r=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((r/'incoming/Kimi-head-Model-manifest-r8235.json').read_text())
for name,h in manifest['source_sha256'].items():assert sha(r/'incoming'/name)==h,name
prep=json.loads((r/'incoming/r5034-Kimi-packed-preparation.json').read_text())
for name,h in prep['source_sha256'].items():assert sha(r/'incoming'/name)==h,name
for filefield,hashfield in [('component_gate','component_gate_sha256'),('packed_component_gate','packed_component_gate_sha256'),('diverse_cases','diverse_cases_sha256')]:
 assert sha(r/'incoming'/prep[filefield])==prep[hashfield],filefield
assert sha(r/'results/kimi-packed-W2-Model-r5049-build.json')==prep['build_record_remote_SHA256']
build=json.loads((r/'results/kimi-packed-W2-Model-r5049-build.json').read_text())
model=pathlib.Path(build['library']);helper=r/'candidates/kimi-helper-bounded-bf16-r6054/libkimi_grouped_prefill.so'
assert build['completed'] and sha(model)==prep['model_sha256']==build['library_SHA256']
from Kimi_native_binding_r8521 import load as load_N4_Model
model,lib,n4_counter_r8167,N4_binary_gate=load_N4_Model(r)
assert sha(helper)==prep['helper_sha256']=='85a54291b8aedd1e59556013ab38d15cc077940e70dd8dfd70acbe50b78c25e4'
h=ctypes.CDLL(str(helper))
for name in ['modern_grouped_mxfp4_owned','modern_grouped_metadata_counts_r5858','modern_bounded_counts_r6054']:
 assert ctypes.cast(getattr(lib,name),ctypes.c_void_p).value==ctypes.cast(getattr(h,name),ctypes.c_void_p).value,name
for handle,name,width in [(lib,'modern_kimi_packed_counts_r6136',8),(lib,'modern_kimi_packed_counts_r6122',6),(h,'modern_grouped_metadata_counts_r5858',7),(h,'modern_bounded_counts_r6054',4)]:
 f=getattr(handle,name);f.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];f.restype=ctypes.c_int
 for device in range(8):
  a=(ctypes.c_uint64*width)();assert f(device,a,width)==0 and list(a)==[0]*width
 for dev,n in[(-1,width),(8,width),(0,0),(0,width-1),(0,width+1)]:
  a=(ctypes.c_uint64*(width+1))(*([0xfeed]*(width+1)));assert f(dev,a,n)!=0 and list(a)==[0xfeed]*(width+1)
 assert f(0,None,width)!=0
class DlInfo(ctypes.Structure):
 _fields_=[('fname',ctypes.c_char_p),('fbase',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(DlInfo)];dl.restype=ctypes.c_int
providers={}
for name in ['modern_kimi_grouped_r4620','modern_kimi_packed_counts_r6136','modern_kimi_packed_counts_r6122']:
 info=DlInfo();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(info))
 assert pathlib.Path(info.fname.decode()).resolve()==model.resolve(),(name,info.fname)
 providers[name]=info.fname.decode()
assert os.environ.get('INFINILM_KIMI_COMPACT_CACHE_R4974')is None
for name in ['INFINILM_KIMI_PACKED_AUTO_R6136','INFINILM_KIMI_PACKED_W2_R6122','INFINILM_KIMI_METADATA_PACK_R5858','INFINILM_KIMI_BOUNDED_BF16_R6054','INFINILM_KIMI_STATIC_DIV_R6020','INFINILM_KIMI_N4_TILE_R8167']:
 assert os.environ[name]=='0',name
test=lib.modern_kimi_compact_contract_r4974;test.restype=ctypes.c_int;assert test()==0
lib.modern_kimi_engine_error_r2475.restype=ctypes.c_char_p
setter=lib.modern_kimi_engine_set_cache_pages_r4974;setter.argtypes=[ctypes.c_void_p,ctypes.c_size_t];setter.restype=ctypes.c_int
assert setter(None,3)!=0 and b'defaultOFF'in lib.modern_kimi_engine_error_r2475()
mapped={line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libkimi_grouped_prefill.so'in line}
assert mapped=={str(helper)},mapped
source=(r/'incoming/modern_Kimi_head_r8235.py').read_text();tree=ast.parse(source)
create=[n.lineno for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='modern_kimi_engine_create_r2475']
assert len(create)==1
assert validate_source(source);metadata_contract=retained_rows()
assert next(i+1 for i,line in enumerate(source.splitlines())if"report['packed_preengine']="in line)<create[0]
out={'step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'run':os.environ['GLOBAL_RUN_R2354'],'completed':True,'rank':int(os.environ['SLURM_PROCID']),'owner_CPU_width_early':4,'GPU_visible_dlopen_device':0,'Engine_created':False,'weights_loaded':False,'GPU_compute_requested':False,
 'N4_binary_gate':N4_binary_gate,'Model_SHA256':sha(model),'helper_SHA256':sha(helper),'packed_object_SHA256':build['GPU_object_SHA256'],
 'all_counter_initial_zero_and_invalid_no_write':True,'compact505_CPU_contract':True,'source_and_component_gate_SHA256_checked':True,
 'capacity32_CPU_contract':metadata_contract,'expected_actual_cache_geometry':GEOMETRY,'CLI_contract':'launcher TPnodes1 PP4; create max_rows32 literal; no user cache CLI override','CAPI_boundary':'create size_t32; setpages4 and actualgeometry132 mandatory in Model; no native32 numerical claim',
 'providers':providers,'mapped_helper':sorted(mapped),'actual_GPU_branch_observer_enabled':False,'Spec_scope11_compatibility_claimed':False}
p=r/'results'/f'kimi-causal-preengine-r8521-{os.environ["GLOBAL_RUN_R2354"]}-node{out["rank"]}.json'
assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps(out))
