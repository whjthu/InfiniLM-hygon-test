"""Only private namespace, one TopK dispatch and symmetric value-proof retention."""
SOURCE_SHA='7f2ffe0395475a98be8723bf434aa9f4944cb5336656b6d57de0f93fc6d1187e'
HEADER_SHA='b97c99c824fdb46a55fc90295c03cb7e2e588bd6d0175f55846facba10dbc2b6'
def rebase(source,header):
 ns='namespace infinilm::models::deepseek_v4 {'
 assert source.count(ns)==1 and header.count(ns)==1
 source=source.split('extern "C" int modern_deep_allselected_set_r813',1)[0]
 source=source.replace(ns,'namespace infinilm::models::deepseek_v4::fixed_topk_r415 {')
 source=source.replace('#include "deepseek_v4_c4_indexer.hpp"','#include "private_indexer.hpp"\n#include "topk_contract.hpp"\n#include <infini/rt/device_guard.h>\nextern "C" int deep_fixed_topk_launch_r415(const void*,void*,void*,void*);')
 old='ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(r.device().type()),scores,int64_t(selected),int64_t{1},true,true,values,indices);'
 new='deep415::Geometry geometry{request.tokens,width,selected,scores->dtype()==rt::DataType::kFloat32&&values->dtype()==rt::DataType::kFloat32,indices->dtype()==rt::DataType::kInt64,scores->is_contiguous()&&values->is_contiguous()&&indices->is_contiguous(),true,true,1};deep415::dispatch(geometry,[&]{infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};int code=deep_fixed_topk_launch_r415(scores->data(),values->data(),indices->data(),r.stream());if(code)throw std::runtime_error("r415 fixedTopK launch failed="+std::to_string(code));},[&]{'+old+'});pending.topk_values.push_back(values);'
 assert source.count(old)==1;source=source.replace(old,new)
 header=header.replace(ns,'namespace infinilm::models::deepseek_v4::fixed_topk_r415 {')
 old='std::vector<runtime::Tensor>scores,indices;';assert header.count(old)==1
 header=header.replace(old,'std::vector<runtime::Tensor>scores,indices,topk_values; // private proof retention, both arms; Native class unchanged.')
 return source,header
