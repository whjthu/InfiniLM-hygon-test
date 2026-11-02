"""One M1 gate-only dispatch extension, preserving current9284 fallback and ABI."""
from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;G=D.parents[2]
q=json.loads((G/'results/bw1000/2026-11-02/GLM367-current-dispatch-source-read.json').read_bytes())
base=next(x for x in q['dispatch_sources']if x['path'].endswith('/dispatch-row2-Model-r9284.cpp'))
s=base['text'];assert hashlib.sha256(s.encode()).hexdigest()==base['SHA256']=='2bedb52f033e73cdc68addaa6c3ec893802c8e3e36980a840b397083bddd3ebf'
(D/'baseline_dispatch_r9284.cpp').write_bytes(s.encode())
assert s.count('if(expected&~3u)')==2 and s.count('fetch_add(4,')==s.count('fetch_sub(4,')==1
s=s.replace('std::array<std::atomic<std::uint64_t>,4>rowcounts{};','std::array<std::atomic<std::uint64_t>,4>rowcounts{},m1counts{};')
s=s.replace('if(expected&~3u)','if(expected&~7u)').replace('(expected&2u)|unsigned(mode)','(expected&6u)|unsigned(mode)').replace('(expected&1u)|(unsigned(mode)<<1)','(expected&5u)|(unsigned(mode)<<1)')
s=s.replace('fetch_add(4,','fetch_add(8,').replace('fetch_sub(4,','fetch_sub(8,')
needle='namespace glm_vector_r8540 {\nFrame::Frame'
extra='''extern "C" int modern_glm_M1_gate_policy_r367(int d,int mode){
 if(!valid(d)||(mode!=0&&mode!=1))return -1;
 auto&s=states[d];unsigned expected=s.word.load(std::memory_order_acquire);
 for(;;){if(expected&~7u)return -2;unsigned desired=(expected&3u)|(unsigned(mode)<<2);if(s.word.compare_exchange_weak(expected,desired,std::memory_order_acq_rel)){if(((expected>>2)&1u)!=unsigned(mode))s.m1counts[3].fetch_add(1,std::memory_order_relaxed);return 0;}}
}
extern "C" int modern_glm_M1_gate_get_r367(int d){if(!valid(d))return -1;return int((states[d].word.load(std::memory_order_acquire)>>2)&1u);}
extern "C" int modern_glm_M1_gate_counts_r367(int d,std::uint64_t*out,std::size_t n){if(!valid(d)||!out||n!=4)return-1;for(std::size_t j=0;j<4;++j)out[j]=states[d].m1counts[j].load(std::memory_order_relaxed);return 0;}
'''
assert s.count(needle)==1;s=s.replace(needle,extra+needle)
old='bool hit=selected(f,routes);int rc=hit?glm_vec_gate_r8528'
new='bool m1=routes==8&&(states[f.device].word.load(std::memory_order_acquire)&4u);bool hit=selected(f,routes)||m1;int rc=hit?glm_vec_gate_r8528'
assert s.count(old)==1;s=s.replace(old,new)
old='record(f,true,hit,routes,rc);return rc;'
new='if(routes==8){states[f.device].m1counts[m1?0:1].fetch_add(1,std::memory_order_relaxed);if(rc)states[f.device].m1counts[2].fetch_add(1,std::memory_order_relaxed);}record(f,true,hit,routes,rc);return rc;'
assert s.count(old)==1;s=s.replace(old,new)
(D/'dispatch_M1_r367.cpp').write_bytes(s.encode())
h=G/'code/bw1000/glm-model-candidates/glm_vector_dispatch_r8540.hpp';data=h.read_bytes();assert hashlib.sha256(data).hexdigest()=='2e5b8d6bdd117fd21ab1920f594570d804ce3338a83605ae63d293dc8db2d6c7';(D/h.name).write_bytes(data)
print('Prepared exact9284-derived M1 gate-only selector; original GPU objects unchanged')
