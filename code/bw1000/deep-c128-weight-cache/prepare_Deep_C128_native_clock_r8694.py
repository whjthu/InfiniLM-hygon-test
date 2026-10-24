import pathlib,hashlib
r=pathlib.Path(__file__).parent
s=(r/'deep_C128_component_bridge_r8686.cpp').read_text()
s=s.replace('#include <set>','#include <set>\n#include <chrono>\n#include <cstdint>')
s=s.replace('thread_local std::string error,serialized;','thread_local std::string error,serialized;\nthread_local double timing_seconds_r8694=0;\nthread_local uint64_t timing_sequence_r8694=0;\nthread_local size_t timing_tokens_r8694=0;\nthread_local bool timing_valid_r8694=false;')
s=s.replace('producer(stream);auto&r=*c->owner;size_t dim=', 'producer(stream);timing_valid_r8694=false;const auto timing_begin_r8694=std::chrono::steady_clock::now();auto&r=*c->owner;size_t dim=')
a='rt::Tensor::from_blob(rotated,{tokens,dim},rt::DataType::kBFloat16,r.device())->copy_from(r,scattered);r.syncStream();return 0;'
b='rt::Tensor::from_blob(rotated,{tokens,dim},rt::DataType::kBFloat16,r.device())->copy_from(r,scattered);r.syncStream();const auto timing_end_r8694=std::chrono::steady_clock::now();timing_seconds_r8694=std::chrono::duration<double>(timing_end_r8694-timing_begin_r8694).count();timing_tokens_r8694=tokens;++timing_sequence_r8694;timing_valid_r8694=true;return 0;'
assert s.count(a)==1;s=s.replace(a,b)
s+='\nextern "C" int modern_real_compress_timing_r8694(double*out,size_t n){if(!out||n!=3||!timing_valid_r8694)return 1;out[0]=timing_seconds_r8694;out[1]=double(timing_sequence_r8694);out[2]=double(timing_tokens_r8694);return 0;}\n'
for n in ['cudaStreamSynchronize','syncStream()','ex::submit']:
 assert s.count(n)==(r/'deep_C128_component_bridge_r8686.cpp').read_text().count(n)
(r/'deep_C128_component_bridge_r8694.cpp').write_text(s,newline='\n')
for old,new in [('build_Deep_C128_component_r8686.py','build_Deep_C128_component_r8694.py'),('build_Deep_C128_component_r8686_inside.sh','build_Deep_C128_component_r8694_inside.sh'),('launch_Deep_C128_component_CPU_r8686.py','launch_Deep_C128_component_CPU_r8694.py'),('stage_Deep_C128_component_CPU_r8686.py','stage_Deep_C128_component_CPU_r8694.py')]:
 t=(r/old).read_text().replace('8686','8694');(r/new).write_text(t,newline='\n')
p=r/'build_Deep_C128_component_r8694.py';t=p.read_text().replace("'modern_real_compress_release_workers']","'modern_real_compress_release_workers','modern_real_compress_timing_r8694']").replace('Unchanged original TemporalCompressor probe bridge','Original TemporalCompressor bridge with native steady_clock only afterproducer and before return');p.write_text(t,newline='\n')
print('READY bridge native timer and single TU CPU scripts')
