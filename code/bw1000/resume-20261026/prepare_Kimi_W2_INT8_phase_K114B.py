import pathlib,json,hashlib,ast,re
r=pathlib.Path(__file__).resolve().parent;old=r/'Kimi-W2-INT8-register-v2-private-r113c';d=r/'Kimi-W2-INT8-phase-private-K114B';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(n,s):
 b=s if isinstance(s,bytes)else s.encode();b=b.replace(b'\r\n',b'\n');assert b'\r'not in b
 if n.endswith('.py'):ast.parse(b)
 (d/n).write_bytes(b)
s=(old/'kimi_W2_INT8_register_provider_r113v2.cu').read_text();prefix,tail=s.split('namespace{std::atomic<uint64_t>pair_counts_r9280',1)
events=r'''// Host-only owner-thread events. Default disabled; no CUDA synchronization here.
namespace k114b {
struct State{int device=-1;cudaStream_t stream=nullptr;cudaEvent_t e[5]{};bool ready=false,on=false;unsigned marks=0;};
thread_local State p;
int mark(unsigned n,cudaStream_t stream){if(!p.on)return 0;if(!p.ready||p.stream!=stream||n>4)return int(cudaErrorInvalidValue);if(n==0)p.marks=0;auto c=cudaEventRecord(p.e[n],stream);if(c==cudaSuccess)p.marks|=1u<<n;return int(c);}
}
extern "C" int modern_kimi_k114b_init(int device,void*stream){auto&p=k114b::p;int current=-1;if(p.ready||cudaGetDevice(&current)!=cudaSuccess||device!=current)return int(cudaErrorInvalidValue);p.device=device;p.stream=static_cast<cudaStream_t>(stream);for(unsigned i=0;i<5;++i){auto c=cudaEventCreate(&p.e[i]);if(c!=cudaSuccess){for(unsigned j=0;j<i;++j)cudaEventDestroy(p.e[j]);p={};return int(c);}}p.ready=true;return 0;}
extern "C" int modern_kimi_k114b_enable(int on){if(on&&!k114b::p.ready)return int(cudaErrorInvalidValue);k114b::p.on=on!=0;return 0;}
extern "C" int modern_kimi_k114b_read(float*out,size_t n){auto&p=k114b::p;if(!out||n!=5||!p.on||p.marks!=31)return int(cudaErrorInvalidValue);for(unsigned i=0;i<4;++i){auto c=cudaEventElapsedTime(out+i,p.e[i],p.e[i+1]);if(c!=cudaSuccess)return int(c);}return int(cudaEventElapsedTime(out+4,p.e[0],p.e[4]));}
extern "C" int modern_kimi_k114b_close(){auto&p=k114b::p;p.on=false;int ret=0;if(p.ready)for(auto e:p.e){auto c=cudaEventDestroy(e);if(c!=cudaSuccess)ret=int(c);}p={};return ret;}
'''
s=prefix+events+'namespace{std::atomic<uint64_t>pair_counts_r9280'+tail
s=s.replace('auto st=static_cast<cudaStream_t>(stream);if(low)', 'auto st=static_cast<cudaStream_t>(stream);if(auto c=k114b::mark(0,st))return c;if(low)')
s=s.replace(' if(i8){\n  code=', ' if(auto c=k114b::mark(1,st))return c;\n if(i8){\n  code=')
s=s.replace('  i8r113::matrix<<<640,64,0,st>>>','  if(auto c=k114b::mark(2,st))return c;\n  i8r113::matrix<<<640,64,0,st>>>')
s=s.replace('  i8r113::reduce<<<','  if(auto c=k114b::mark(3,st))return c;\n  i8r113::reduce<<<')
s=s.replace('T);return int(cudaGetLastError());\n }\n const char*toggle=', 'T);if(auto c=k114b::mark(4,st))return c;return int(cudaGetLastError());\n }\n if(auto c=k114b::mark(2,st))return c;\n const char*toggle=')
s=s.replace('size_t{192},size_t{0});return int(cudaGetLastError());','size_t{192},size_t{0});if(auto c=k114b::mark(3,st))return c;if(auto c=k114b::mark(4,st))return c;return int(cudaGetLastError());')
assert s.count('k114b::mark(')==8
# All device code is byte-identical after newline normalization, before host insert.
assert s[:s.index(events)]==(old/'kimi_W2_INT8_register_provider_r113v2.cu').read_text().split('namespace{std::atomic<uint64_t>pair_counts_r9280',1)[0]
put('kimi_W2_INT8_provider_K114B.cu',s)
w=(old/'W2_INT8_runner_r113.cpp').read_text();decl='extern "C" int modern_kimi_k114b_init(int,void*);\nextern "C" int modern_kimi_k114b_enable(int);\nextern "C" int modern_kimi_k114b_read(float*,size_t);\nextern "C" int modern_kimi_k114b_close();\n';w=w.replace('namespace {using J=',decl+'namespace {using J=')
w=w.replace('auto cfg=infinilm::config::ConfigFactory', 'const bool diagnostic=q.value("diagnostic_events_K114B",false);const int blocks=diagnostic?3:1,total_calls=4+64*blocks,arm_calls=2+32*blocks;auto cfg=infinilm::config::ConfigFactory')
w=w.replace(' for(int scenario=0;', ' if(diagnostic&&modern_kimi_k114b_init(rank,r.stream()))throw std::runtime_error("K114B event init");\n for(int scenario=0;')
w=w.replace(' for(int arm=0;arm<8;++arm){',' for(int block=0;block<blocks;++block){const bool events=diagnostic&&block==1;const char*blockname=diagnostic?(block==0?"pre_control":block==1?"events":"post_control"):"original_control";if(modern_kimi_k114b_enable(events))throw std::runtime_error("K114B enable");\n for(int arm=0;arm<8;++arm){')
needle='   samples.push_back({{"rep",rep}'
pos=w.index(needle);end=w.index('\n  }',pos);w=w[:pos]+w[pos:end]+'''\n   if(events){float ms[5]{};if(modern_kimi_k114b_read(ms,5))throw std::runtime_error("K114B event read after original sync");for(float x:ms)if(!std::isfinite(x)||x<0)throw std::runtime_error("K114B invalid timing");auto&z=samples.back();z["owner_stream_route_W13_ms"]=ms[0];z["owner_stream_quant_jobs_ms"]=ms[1];z["owner_stream_W2matrix_or_pair_ms"]=ms[2];z["owner_stream_reduce_ms"]=ms[3];z["owner_stream_provider_span_ms"]=ms[4];z["signed_wall_minus_provider_ms"]=z["full_MoE_complete_wall_ms"].get<double>()-ms[4];}
'''+w[end:]
w=w.replace('{{"T",T},{"constructed_low_reuse",constructed},{"extra_scratch_bytes"','{{"diagnostic_block",blockname},{"events_enabled",events},{"T",T},{"constructed_low_reuse",constructed},{"extra_scratch_bytes"')
w=w.replace('\n if(modern_kimi_W2_pair_counts_r9280(rank,pair_after', '\n }if(modern_kimi_k114b_enable(0))throw std::runtime_error("K114B disable");\n if(modern_kimi_W2_pair_counts_r9280(rank,pair_after')
w=w.replace('!=68*T','!=total_calls*T').replace('!=68','!=total_calls').replace('!=34','!=arm_calls').replace('(constructed?68:0)','(constructed?total_calls:0)')
w=w.replace('}phase.wait();if(rank==0){setenv(', '}if(diagnostic){r.syncStream();if(modern_kimi_k114b_close())throw std::runtime_error("K114B event close");row["K114B_events_closed_before_Runtime"]=true;}phase.wait();if(rank==0){setenv(')
assert w.count('modern_kimi_k114b_read(ms,5)')==1 and w.count('cudaEventSynchronize')==0
put('W2_INT8_runner_K114B.cpp',w)
p=(old/'run_GPU_r113.py').read_text().replace("['modern_kimi_i8_counts_r113',", "['modern_kimi_k114b_init','modern_kimi_k114b_read', 'modern_kimi_i8_counts_r113',")
p=p.replace("'config':json.loads(","'diagnostic_events_K114B':m.get('diagnostic_events_K114B',False),'config':json.loads(")
p=p.replace("len(owner['arms'])==24", "len(owner['arms'])==(72 if m.get('diagnostic_events_K114B',False)else24)")
p=p.replace("q['I192_calls']==68", "q['I192_calls']==(196 if m.get('diagnostic_events_K114B',False)else68)").replace("68*q['T']", "(196 if m.get('diagnostic_events_K114B',False)else68)*q['T']")
for k in ['W2_pair_ON_calls','I8_OFF_calls','I8_ON_calls']:p=p.replace("q['"+k+"']==34", "q['"+k+"']==(98 if m.get('diagnostic_events_K114B',False)else34)")
p=p.replace('else24','else 24').replace('else68','else 68').replace('else34','else 34');put('run_GPU_K114B.py',p)
put('expected-headers-r9125.json',(old/'expected-headers-r9125.json').read_bytes())
proof={'schema':1,'GPU':False,'flag_default_OFF':True,'base_v2_provider_SHA256':sha(old/'kimi_W2_INT8_register_provider_r113v2.cu'),'device_prefix_normalized_identical':True,'new_event_count_per_owner':5,'no_between_phase_sync':True,'read_after_existing_sync':True,'read_outside_wholewall_end':True,'blocks':['pre_control','events','post_control'],'pairs_each_block':4,'rep_each_arm':8,'calls_per_scenario_when_diagnostic':196,'arm_calls':98,'same_3_scenarios':True,'same_matrix_launch':[640,64,0],'remainder':'signed sameowner wholewall minus provider event0to4 span; CPU and other GPU/shared/router/TP/drain and launch gaps; NOT GPUbusy; no sum crossrank maxima','perturbation':'compare each block same-run per-rep max16owner wholewall and corresponding paired gains; if perturbation/drift masks 0.079ms T8 or8.23ms lowreuse attribution UNKNOWN','control_is_no_instrumentation':'event record calls fully skipped on TLS false; same instrumentable binary; original v2 crossbinary reference only','source_SHA256':{p.name:sha(p)for p in d.iterdir()}}
put('sourceproof-K114B.json',json.dumps(proof,indent=2)+'\n')
print(json.dumps({'private':str(d),'files':{p.name:sha(p)for p in d.iterdir()}}))
