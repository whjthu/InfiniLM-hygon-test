import pathlib,hashlib,json
here=pathlib.Path(__file__).resolve().parent
q=json.loads((here.parent/'Kimi-taps-source-read-r9022f.json').read_text(encoding='utf8'))
assert q['SHA256']=='7829d96b99dc88e7f0ce914bbfa495a8110efe0b3fec1de83295ba8567957dee'
trace=q['text']
needle='namespace infinilm::models::kimi_k3 {'
assert trace.count(needle)==1
trace=trace.replace(needle,'#include "kimi_target_taps_r9022.hpp"\n'+needle)
needle='bool trace_detail_layer(size_t layer){'
trace=trace.replace(needle,needle+'\n if(kimi_target_taps_r9022::enabled())return true;')
needle='void trace_hidden(runtime::Runtime&r,const runtime::Tensor&x,const global_state::ForwardContext&fc,size_t layer,const char*point){'
assert trace.count(needle)==1
trace=trace.replace(needle,needle+'\n kimi_target_taps_r9022::collect(r,x,fc,layer,point);')
(here/'kimi_k3_trace_target_taps_r9022.cpp').write_text(trace,encoding='utf8')
print(json.dumps({'generated':True,'source_SHA256':hashlib.sha256(trace.encode()).hexdigest(),'GPU':False,'compiled':False}))
