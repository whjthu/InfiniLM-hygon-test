from pathlib import Path
import json,hashlib,ast
D=Path(__file__).resolve().parent;S=D.parent/'Kimi-KDA-fused-Delta-r343'
header=(S/'fused_r343.hpp').read_text().replace('#include <array>','#include <array>\n#include <nlohmann/json.hpp>\n#include "layout_r345.hpp"')
header=header.replace('uint64_t chosen=0,fallback=0;','uint64_t chosen=0,fallback=0;std::string diag;')
old=' for(size_t j=0;j<12;++j)yes &= x[j]&&x[j]->device()==r.device()&&x[j]->is_contiguous();'
new=''' static const std::vector<rt::Shape>expected{{1,1,6,128},{3,6,128,128},{1,1,6,128},{1,1,6,128},{1,1,6,128},{1,1,6,128},{1,1,6},{6},{6,128},{2},{1},{1}};
 nlohmann::json metadata;
 for(size_t j=0;j<12;++j){if(!x[j])throw std::invalid_argument("r345 null operand");bool linear=layout_r345::accept(x[j]->shape(),x[j]->strides(),expected[j]);yes &= x[j]->device()==r.device()&&linear;
  if(s.diag.empty())metadata.push_back({{"operand",j},{"shape",x[j]->shape()},{"strides",x[j]->strides()},{"dtype",int(x[j]->dtype())},{"strict_contiguous",x[j]->is_contiguous()},{"fixedC1_linear",linear}});
 }'''
assert old in header;header=header.replace(old,new)
header=header.replace(' if(!yes){++s.fallback;return false;}', ' if(s.diag.empty())s.diag=nlohmann::json{{"eligible",yes},{"T",s.T},{"read",s.read},{"write",s.write},{"scale",scale},{"lower",lower},{"norm",norm},{"metadata",metadata}}.dump();\n if(!yes){++s.fallback;return false;}')
(D/'fused_r343.hpp').write_bytes(header.encode())
build=(S/'build_CPU.py').read_text().replace('Kimi-KDA-fused-Delta-r343','Kimi-KDA-C1-layout-Delta-r345').replace('KIMI_R343_MANIFEST_SHA','KIMI_R345_MANIFEST_SHA')
build=build.replace("(N/'fused_r343.hpp').write_bytes((D/'fused_r343.hpp').read_bytes())","(N/'fused_r343.hpp').write_bytes((D/'fused_r343.hpp').read_bytes());(N/'layout_r345.hpp').write_bytes((D/'layout_r345.hpp').read_bytes())")
build=build.replace("old179+'/fused_r343.hpp'}","old179+'/fused_r343.hpp',old179+'/layout_r345.hpp'}")
old="run(['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-c',str(D/'search.cu'),'-o',str(gpu)],'fused-GPU-TU',150)"
new="""parent=R/'candidates/Kimi-KDA-fused-Delta-r343/CPU-build.json';assert sha(parent)=='0641411e83761507d6aff853a2ab80b483666b03ff5a9c0ed23d3222ca2c62f8'
        oldgpu=R/'candidates/Kimi-KDA-fused-Delta-r343/fused.o';gpu.write_bytes(oldgpu.read_bytes());q['reused_GPU_object_SHA256']=sha(gpu)
        run(['g++','-std=c++17','-O2',str(D/'CPU_layout.cpp'),'-I'+str(D),'-o',str(B/'CPU_layout')],'layout-compile',30)
        q['CPU_layout']=json.loads(run([str(B/'CPU_layout')],'layout-contract',10))"""
assert old in build;build=build.replace(old,new).replace('new_GPU_TUs=1','new_GPU_TUs=0,reused_GPU_TUs=1')
build=build.replace('libKimi_fused_Main_r343.so','libKimi_fused_Main_r345.so').replace('libKimi_fused_shim_r343.so','libKimi_fused_shim_r345.so')
(D/'build_CPU.py').write_bytes(build.encode())
shim=(S/'layer0_native.cpp').read_text()+'''\nextern "C" const char*layer0_fused_diag_r345(void*raw){auto*c=(C*)raw;try{if(!c)throw std::invalid_argument("r345 ctx");c->r->assertOwnerThread();return fused_r343::entries[c->r.get()].diag.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}\n'''
(D/'layer0_native.cpp').write_bytes(shim.encode())
inside=(S/'CPU_inside.sh').read_text().replace('Kimi-KDA-fused-Delta-r343','Kimi-KDA-C1-layout-Delta-r345');(D/'CPU_inside.sh').write_bytes(inside.encode())
driver=(S/'driver.py').read_text().replace('Kimi-KDA-fused-Delta-r343','Kimi-KDA-C1-layout-Delta-r345').replace('KIMI_R343_GPU_MANIFEST_SHA','KIMI_R345_GPU_MANIFEST_SHA').replace('Kimi343-fused-Delta','Kimi345-C1-layout-Delta').replace('KIMI343_NORMAL','KIMI345_NORMAL')
driver=driver.replace('    lib.layer0_fused_counts_r343.argtypes=[ctypes.c_void_p,ctypes.c_void_p]','    lib.layer0_fused_counts_r343.argtypes=[ctypes.c_void_p,ctypes.c_void_p]\n    lib.layer0_fused_diag_r345.argtypes=[ctypes.c_void_p];lib.layer0_fused_diag_r345.restype=ctypes.c_char_p')
needle='                    updated=states()'
inject='''                    if on and 'first_ON_dispatch'not in own:
                        counts=(ctypes.c_uint64*2)();check(lib.layer0_fused_counts_r343(ctx,counts));diag=lib.layer0_fused_diag_r345(ctx);own['first_ON_dispatch']={'counts':list(counts),'metadata':json.loads(diag.decode())if diag else None};save()
                        assert list(counts)==[1,0],('firstON eligibility failed',own['first_ON_dispatch'])
'''
assert needle in driver;driver=driver.replace(needle,inject+needle);(D/'driver.py').write_bytes(driver.encode())
(D/'GPU_inside.sh').write_bytes((S/'GPU_inside.sh').read_bytes().replace(b'Kimi-KDA-fused-Delta-r343',b'Kimi-KDA-C1-layout-Delta-r345'))
launch=(S/'launch.py').read_text().replace('Kimi-KDA-fused-Delta-r343','Kimi-KDA-C1-layout-Delta-r345').replace('Kimi343','Kimi345').replace('KIMI_R343_GPU_MANIFEST_SHA','KIMI_R345_GPU_MANIFEST_SHA');(D/'launch.py').write_bytes(launch.encode())
for n in ['prepare_local.py','prepare_GPU_local.py','original-source.json','patched-preview.cpp']:
 if(D/n).exists():(D/n).unlink()
for p in D.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
old=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());m={'original_shim_receipt_SHA256':old['original_shim_receipt_SHA256'],'files_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in D.iterdir()if p.is_file()and p.name not in ['SOURCE_MANIFEST.json','GPU_MANIFEST.json']},'scope':'r345 fixedC1 linear layout predicate; early firstON gate; source unchanged math; CPU only until ordinary publish'}
(D/'SOURCE_MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());print(hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
