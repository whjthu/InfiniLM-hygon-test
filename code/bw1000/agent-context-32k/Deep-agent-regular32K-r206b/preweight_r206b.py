"""New Native provider + actual HF/Rust parity before opening weight payload."""
import pathlib,json,hashlib,ctypes,sys,os
from native_binding_r206b import binding
from profiles_r206b import fingerprint
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-agent-regular32K-r206b';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run():
 b,c,n=binding();cpu=json.loads((B/'CPU-preflight.json').read_bytes());assert cpu['completed']and cpu['CPU_only']and not cpu['Torch_imported']
 assert sha(D/'Native-binding.json')==cpu['Native_binding_SHA256']
 for p,h in cpu['baseline_files_SHA256'].items():assert sha(p)==h
 for p,h in cpu['overlay_files_SHA256'].items():assert sha(p)==h
 import torch
 assert torch.cuda.device_count()==8;torch.cuda.set_device(0)
 lib=ctypes.CDLL(c['library'])
 class Info(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dl.restype=ctypes.c_int;providers={}
 for symbol in ['modern_native_engine_create','modern_native_engine_load','modern_native_engine_finish','modern_native_engine_forward','modern_native_engine_reset','modern_native_engine_destroy']:
  info=Info();assert dl(ctypes.cast(getattr(lib,symbol),ctypes.c_void_p),ctypes.byref(info));p=pathlib.Path(info.fname.decode()).resolve();assert p==pathlib.Path(c['library']).resolve()and sha(p)==b['library_SHA256'];providers[symbol]={'path':str(p),'SHA256':sha(p)}
 from transformers import PreTrainedTokenizerFast
 assert sha(pathlib.Path(cpu['model_directory'])/'tokenizer.json')==cpu['profiles']['tokenizer_json_SHA256']
 tokenizer=PreTrainedTokenizerFast.from_pretrained(cpu['model_directory'],local_files_only=True)
 actual=fingerprint(B/'profiles',tokenizer);assert actual==cpu['profiles']['parity'],'Actual HF tokenizer fixture IDs differ from CPU Rust; do not load weights'
 maps={pathlib.Path(x.split()[-1]).resolve()for x in pathlib.Path('/proc/self/maps').read_text().splitlines()if len(x.split())>=6 and x.split()[-1].startswith('/')};actual_bindings=[]
 for name,proof in c['actual_DTK_bindings'].items():
  expected=pathlib.Path(name).resolve();stem=pathlib.Path(name).name.split('.so')[0];found={p for p in maps if stem in p.name};assert found=={expected}and sha(expected)==proof['SHA256'];actual_bindings.append({'path':str(expected),'SHA256':sha(expected)})
 return dict(passed=True,weights_payload_opened=False,visible_GPUs=8,actual_new_Native_providers=providers,actual_required_DSO_bindings=actual_bindings,actual_HF_Rust_parity=True,parity=actual,new_Native_GPU_model_qualification=False)
