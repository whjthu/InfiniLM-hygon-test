from pathlib import Path
import json,hashlib,ast,sys
D=Path(__file__).resolve().parent;S=D.parent/'Kimi-C1-Delta-profile-r309'
code=(S/'build_CPU.py').read_text().replace('Kimi-C1-Delta-profile-r309','Kimi-KDA-fused-Delta-r343').replace('KIMI_R309_MANIFEST_SHA','KIMI_R343_MANIFEST_SHA').replace('from patch_current_delta import patch','from patch_fused import patch')
code=code.replace("(N/'delta_profile_r309.hpp').write_bytes((D/'delta_profile_r309.hpp').read_bytes())","(N/'delta_profile_r309.hpp').write_bytes((D/'delta_profile_r309.hpp').read_bytes()); (N/'fused_r343.hpp').write_bytes((D/'fused_r343.hpp').read_bytes())")
code=code.replace("extra={old179+'/delta_profile_r309.hpp'}","extra={old179+'/delta_profile_r309.hpp',old179+'/fused_r343.hpp'}")
code=code.replace("        run(link,'Main-link',50)","""        gpu=B/'fused.o'
        run(['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936','-Xcompiler=-fPIC','-c',str(D/'search.cu'),'-o',str(gpu)],'fused-GPU-TU',150)
        link.insert(link.index('-o'),str(gpu));run(link,'Main-link',50)""")
code=code.replace("libKimi_C1_Delta_Main_r309.so","libKimi_fused_Main_r343.so").replace("libKimi_C1_Delta_shim_r309.so","libKimi_fused_shim_r343.so")
code=code.replace('330-','560-').replace('new_GPU_TUs=0','new_GPU_TUs=1').replace('new_profile_header_only=True','new_profile_header_and_fused_adapter=True')
(D/'build_CPU.py').write_bytes(code.encode())
shim=(S/'layer0_native.cpp').read_text().replace('#include "delta_profile_r309.hpp"','#include "delta_profile_r309.hpp"\n#include "fused_r343.hpp"')
shim=shim.replace('kimi_delta_profile_r309::clear(*c->r);','kimi_delta_profile_r309::clear(*c->r);fused_r343::clear(*c->r);')
shim=shim.replace('result=c->d->forward(input,fc);','fused_r343::bind(*c->r,total,read[0],write[0],{offsets,reads,writes});result=c->d->forward(input,fc);')
shim+='''
extern "C" int layer0_fused_mode_r343(void*raw,int on){auto*c=(C*)raw;try{if(!c||(on!=0&&on!=1))throw std::invalid_argument("r343 mode");fused_r343::mode(*c->r,on);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int layer0_fused_counts_r343(void*raw,uint64_t*out){auto*c=(C*)raw;try{if(!c||!out)throw std::invalid_argument("r343 counts");c->r->assertOwnerThread();auto&s=fused_r343::entries[c->r.get()];out[0]=s.chosen;out[1]=s.fallback;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
'''
(D/'layer0_native.cpp').write_bytes(shim.encode())
inside=(S/'CPU_inside.sh').read_text().replace('340 python','580 python').replace('Kimi-C1-Delta-profile-r309','Kimi-KDA-fused-Delta-r343')
(D/'CPU_inside.sh').write_bytes(inside.encode())
sys.path.insert(0,str(D));from patch_fused import patch
patched=patch((D/'original_delta.cpp').read_text());(D/'patched-preview.cpp').write_bytes(patched.encode())
for p in D.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
m0=json.loads((S/'SOURCE_MANIFEST.json').read_text());m={'original_shim_receipt_SHA256':m0['original_shim_receipt_SHA256'],'files_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in D.iterdir()if p.is_file()and p.name!='SOURCE_MANIFEST.json'},'scope':'CPU-only fusedW2 Delta integration; same fixed read1/write2 C1 geometry; defaultOFF; originalfallback unmodified; noGPU validation yet'}
(D/'SOURCE_MANIFEST.json').write_bytes((json.dumps(m,indent=2)+'\n').encode());print(hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
