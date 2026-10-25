from pathlib import Path
import json,hashlib,ast
r=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=(r/'glm_scalar_vectorload_r8528.cu').read_text();inc=(r/'glm_gate_register_mmac_candidate_r9048.inc').read_text();assert '__global__ void gate_up_register_mmac_r9048' in inc
s=s.replace('__global__ void validate(',inc+'\n__global__ void validate(',1)
old='''  gate_up<<<dim3(64,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>(
   static_cast<__nv_bfloat16*>(gate),static_cast<__nv_bfloat16*>(up),static_cast<const uint8_t*>(x),
   static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(wg),
   static_cast<const float*>(sg),static_cast<const uint8_t*>(wu),static_cast<const float*>(su),routes,offset);'''
assert old in s
args='''static_cast<__nv_bfloat16*>(gate),static_cast<__nv_bfloat16*>(up),static_cast<const uint8_t*>(x),
   static_cast<const float*>(xs),static_cast<const int64_t*>(ids),static_cast<const uint8_t*>(wg),
   static_cast<const float*>(sg),static_cast<const uint8_t*>(wu),static_cast<const float*>(su),routes,offset'''
s=s.replace('extern "C" int glm_vec_gate_r8528(','''namespace { thread_local int fragment_policy9048=0; thread_local uint64_t fragment_counts9048[2]={0,0}; }
extern "C" int glm_gate_fragment_policy_r9048(int mode) { if(mode<0||mode>1)return -1;fragment_policy9048=mode;return 0; }
extern "C" int glm_gate_fragment_counts_r9048(uint64_t*out,size_t n) { if(!out||n!=2)return -1;out[0]=fragment_counts9048[0];out[1]=fragment_counts9048[1];return 0; }
extern "C" int glm_vec_gate_r8528(''',1)
s=s.replace(old,'''  ++fragment_counts9048[fragment_policy9048];
  if(fragment_policy9048) gate_up_register_mmac_r9048<<<dim3(4,unsigned(routes)),256,0,static_cast<cudaStream_t>(stream)>>>('''+args+''');
  else {'''+old+'''\n  }''')
(r/'glm_gate_fragment_r9048.cu').write_bytes(s.encode())
build=(r/'build_GLM_gate_lookahead_CPU_r9044.py').read_text().replace('glm-gate-lookahead-r9044','glm-gate-fragment-r9048').replace('glm_gate_lookahead_r9042.cu','glm_gate_fragment_r9048.cu').replace('lookahead.s','fragment.s').replace('lookahead.o','fragment.o').replace('lookahead.d','fragment.d').replace('libglm_gate_lookahead_r9044.so','libglm_gate_fragment_r9048.so').replace('glm_gate_lookahead_policy_r9042','glm_gate_fragment_policy_r9048').replace('glm_gate_lookahead_counts_r9042','glm_gate_fragment_counts_r9048').replace('lookaheaddefaultOFF','fragmentMMACdefaultOFF')
build=build.replace("assert len(metadata)>=4,('ISAmetadata',metadata)","assert len(metadata)>=4,('ISAmetadata',metadata);assert 'v_mmac_f32_16x16x16_bf16' in text, 'missing_actual_BF16MMAC_ISA';q['actual_BF16MMAC_instruction_lines']=[v.strip()for v in text.splitlines()if 'v_mmac_f32_16x16x16_bf16'in v][:12]")
(r/'build_GLM_gate_fragment_CPU_r9048.py').write_bytes(build.encode())
for old in ['GLM_lookahead_CPU_r9044_inside.sh','launch_GLM_lookahead_CPU_r9044.py','actual_GLM_lookahead_CPU_r9044.py','observe_GLM_lookahead_CPU_r9044.py']:
 new=old.replace('lookahead','fragment').replace('r9044','r9048');t=(r/old).read_text().replace('lookahead','fragment').replace('r9044','r9048');(r/new).write_bytes(t.encode())
files=['glm_gate_fragment_r9048.cu','build_GLM_gate_fragment_CPU_r9048.py','GLM_fragment_CPU_r9048_inside.sh','launch_GLM_fragment_CPU_r9048.py']
(r/'GLM-fragment-CPU-manifest-r9048.json').write_bytes((json.dumps({'sources_SHA256':{n:sha(r/n)for n in files}},indent=2)+'\n').encode())
for n in files:
 if n.endswith('.py'):ast.parse((r/n).read_bytes())
print(json.dumps({'files':files,'source_SHA256':sha(r/'glm_gate_fragment_r9048.cu'),'baseline_gate_down_preserved':True,'license':'Own fragment kernel andmapping algebra; inline instruction idiom reference author attention.cu Apache-2.0/vLLM copyright preserved by URL attribution in candidate source; no external runtime/code dependency'}))
