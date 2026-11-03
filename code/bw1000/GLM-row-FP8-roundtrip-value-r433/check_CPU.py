"""Only new declared FFI and F32 source contract. No old suites or device."""
from pathlib import Path
import ast,ctypes,importlib.util,json
def run():
 d=Path(__file__).resolve().parent
 for p in d.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
 sp=importlib.util.spec_from_file_location('r433_actual_binding',d/'binding.py');module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
 seen=[];buffer=ctypes.create_string_buffer(64);ctx=ctypes.addressof(buffer);assert ctx>2**32
 @ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_int)
 def callback(pointer,mode):seen.append((pointer,mode));return 0 if pointer==ctx and mode in [0,1]else -7
 class Lib:pass
 lib=Lib();lib.glm_attention_value_roundtrip_r433=ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.cast(callback,ctypes.c_void_p).value)
 fn=module.bind_policy(lib);assert fn(ctx,0)==fn(ctx,1)==0 and fn(ctx,2)==-7 and seen==[(ctx,0),(ctx,1),(ctx,2)]
 c=(d/'attention_component_r433.cpp').read_text();h=(d/'value_roundtrip_r433.hpp').read_text()
 assert 'glm_attention_value_roundtrip_r433'in c and 'bool value_roundtrip=false'in c and 'prepared433.reset();if(modern_glm_fp8_runtime_clear_r809'in c
 assert c.index('auto start=Clock::now();')<c.index('if(!c->prepared433)c->prepared433=std::make_unique')<c.index('attn=m::value_r433::attention')
 assert 'm::value_r433::counts={};if(c->prepared433)c->prepared433->check();c->value_roundtrip=bool(mode)'in c
 assert '2,m,n,k,sr,sc,ro,0,r.stream()'in h and 'glm_fp8_run_r776(plan,1'in h
 assert 'Fp32Precision::FullFp32'in h and 'aggregate=w.alloc({8,512},rt::DataType::kFloat32)'in h and 'row_roundtrip_F32_identity_calls'in h
 assert 'memcpyH2D' in h and 'identity_host;'in h and 'catch(...){try{r.syncStream();}'in h
 return dict(passed=True,actual_bind_policy_libffi_high_context=True,Native_setter_not_executed=True,source_F32_until_finalcast=True,source_original_BlockQuant_and_F32identity_helper=True,source_owned_public_weights_and_exception_staging=True,old_CPU_suites_rerun=False,Slurm=0,GPU=0,Native_dlopen=False,numerics_qualified=False)
if __name__=='__main__':print(json.dumps(run(),indent=2))
