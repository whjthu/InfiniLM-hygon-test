import pathlib,json,hashlib
r=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=r/'GLM_PP2_B32_r8843_modern_glm_cached_model_bridge_r809.cpp';assert sha(src)=='7a61948bda209d6acf7a82d73c34673c5350d43f6bc33f442460be76b8a8a53f'
s=src.read_text();s='#include "glm_resident_exact_budget_r8854.hpp"\n'+s
old='glm_rows32_r5731::admission_bytes(c->rows_r5688,glm_model_resident_r5640::selected(next))';assert s.count(old)==1;s=s.replace(old,'glm_resident_exact_r8854::admission_bytes(c->rows_r5688,glm_model_resident_r5640::selected(next))')
old='q["generation"]=c->resident_generation_r5640;';assert s.count(old)==1;s=s.replace(old,'q["resident_physical_bytes_per_layer"]=glm_resident_exact_r8854::admission_bytes(c->rows_r5688,1);'+old)
(r/'GLM_exact_budget_bridge_r8854.cpp').write_bytes(s.encode())
(r/'glm_resident_exact_budget_r8854.hpp').write_bytes(b'''#pragma once
#include "glm_rows32_policy_r5731.hpp"
// The exact resident caller uses owned BF16 K/V [32,640,8,256].
// This private definition is consumed by the actual bridge and its CPU check.
namespace glm_resident_exact_r8854 {
constexpr uint64_t rows32_bank_bytes=2ull*32*640*8*256*2;
inline uint64_t admission_bytes(size_t rows,size_t selected){
 if(rows==32){if(selected>40)throw std::invalid_argument("8854 exact rows32 layer count");return selected*rows32_bank_bytes;}
 return glm_rows32_r5731::admission_bytes(rows,selected);
}
}
''')
(r/'check_GLM_exact_budget_r8854.cpp').write_bytes(b'''#include "glm_resident_exact_budget_r8854.hpp"
#include <iostream>
int main(){
 using namespace glm_resident_exact_r8854;namespace g=glm_model_resident_r5640;
 static_assert(rows32_bank_bytes==167772160);
 const auto b=admission_bytes(32,40);if(b!=6710886400ull)return 1;
 g::admit(b+g::reserve_bytes+g::new_workspace_allowance,b);
 bool rejected=false;try{g::admit(b+g::reserve_bytes+g::new_workspace_allowance-1,b);}catch(const std::invalid_argument&){rejected=true;}
 if(!rejected||admission_bytes(16,1)!=128ull*1024*1024||admission_bytes(64,1)!=g::bank_bytes)return 2;
 std::cout<<"{\\"resident_bank_bytes\\":"<<rows32_bank_bytes<<",\\"layers40_bytes\\":"<<b<<",\\"reserve_bytes\\":"<<g::reserve_bytes<<",\\"allowance_bytes\\":"<<g::new_workspace_allowance<<",\\"boundary_success\\":true,\\"onebyte_short_rejected\\":true,\\"GPU\\":false}";
}
''')
inside=b'#!/bin/bash\nset -euo pipefail\nsource /opt/dtk/env.sh\nexport CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\npython -u /refactor/incoming/build_GLM_exact_budget_CPU_r8854.py\n';(r/'GLM_exact_budget_CPU_r8854_inside.sh').write_bytes(inside)
s=(r/'launch_GLM_resident_exact_CPU_r8852.py').read_text().replace('r8852','r8854').replace('resident_exact','exact_budget').replace('resident-exact','exact-budget');(r/'launch_GLM_exact_budget_CPU_r8854.py').write_bytes(s.encode())
names=['GLM_exact_budget_bridge_r8854.cpp','glm_resident_exact_budget_r8854.hpp','check_GLM_exact_budget_r8854.cpp','build_GLM_exact_budget_CPU_r8854.py','launch_GLM_exact_budget_CPU_r8854.py','GLM_exact_budget_CPU_r8854_inside.sh']
q={'sources_SHA256':{n:sha(r/n)for n in names},'parent_DSO_SHA256':'c14fcb3d31ff22564a95419403e438c4888f84abe0dccfe315bc04271332d3bd','CPU_only':True,'GPU':False,'dlopen':False,'scope':'Only actual Model bridge exact-ROWS32 native admission/report budget. Same201 exactbank resident.o +26 otherobjects; no allocator/kernel/math/SDK change.'}
(r/'GLM-exact-budget-CPU-manifest-r8854.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps({'manifest_SHA256':sha(r/'GLM-exact-budget-CPU-manifest-r8854.json'),'files':names}))
