"""Exact current C4 source, namespace isolation and status clocks only."""
SOURCE_SHA='7f2ffe0395475a98be8723bf434aa9f4944cb5336656b6d57de0f93fc6d1187e'
HEADER_SHA='b97c99c824fdb46a55fc90295c03cb7e2e588bd6d0175f55846facba10dbc2b6'
def rebase(source,header):
 ns='namespace infinilm::models::deepseek_v4 {'
 assert source.count(ns)==1 and header.count(ns)==1
 # Only the original class implementation; old public policy exports stay in Native.
 source=source.split('extern "C" int modern_deep_allselected_set_r813',1)[0]
 source=source.replace(ns,'namespace infinilm::models::deepseek_v4::status_enclosure_r406 {')
 source=source.replace('#include "deepseek_v4_c4_indexer.hpp"','#include "private_indexer.hpp"\n#include "status_timing.hpp"')
 old='void status(rt::Runtime&r,const rt::Tensor&flag){int value=0;r.memcpyD2H(&value,flag->data(),4);r.syncStream();if(value)throw std::invalid_argument("DeepSeek C4 indexer GPU validation status="+std::to_string(value));}'
 new='void status(rt::Runtime&r,const rt::Tensor&flag,deep406::Site site){deep406::status_boundary(site,[&]{int value=0;r.memcpyD2H(&value,flag->data(),4);r.syncStream();if(value)throw std::invalid_argument("DeepSeek C4 indexer GPU validation status="+std::to_string(value));});}'
 assert source.count(old)==1;source=source.replace(old,new)
 # The five call sites are store, gather, query feature, boundary key feature, scores.
 labels=['Store','Gather','Query','Key','Scores'];assert source.count('status(r,flag);')==5
 for label in labels:source=source.replace('status(r,flag);','status(r,flag,deep406::Site::'+label+');',1)
 header=header.replace(ns,'namespace infinilm::models::deepseek_v4::status_enclosure_r406 {')
 return source,header
