"""Exact source derivation; same class/header ABI. Never edits the parent."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
P=D.parent/'agent-context-32k/GLM-PP4-registry-repair-private-r184b/resident_caller_registry_r184b.cpp'
SHA='8aa1de7218120cc8d6687fbc3110c419d8651028f6cac92982b03fd5e7578dcd'
def derive(raw):
 assert hashlib.sha256(raw).hexdigest()==SHA
 s=raw.decode();edits=[]
 def replace(old,new):
  nonlocal s
  assert s.count(old)==1,(old,s.count(old))
  s=s.replace(old,new);edits.append(dict(old=old,new=new))
 replace('#include "glm_registry_geometry_r184b.hpp"','#include "glm_registry_geometry_r184b.hpp"\n#include "selected_values_r398.hpp"')
 replace('  const bool host4504=projected_host_r4504::enabled();','  const bool v398=v_on_demand_r398::requested();\n  const bool host4504=projected_host_r4504::enabled();\n  if(v398&&(rows!=1||capacity!=32768||topk!=2048||rank.tp_size!=8||host4504))throw std::invalid_argument("r398 C1/full32K/TP8/device-K-only scope");')
 replace('  projected_value_=exact_banks_r4188::allocate(r,{host4504?size_t{1}:rows,capacity,prepare_->local_heads(),256});','  if(!v398)projected_value_=exact_banks_r4188::allocate(r,{host4504?size_t{1}:rows,capacity,prepare_->local_heads(),256});')
 replace('  r.setDeviceMemory(projected_value_->data(),0xff,projected_value_->nbytes());','  if(projected_value_)r.setDeviceMemory(projected_value_->data(),0xff,projected_value_->nbytes());')
 replace(' const bool request_batch5055=batch_flag5055&&std::string(batch_flag5055)=="1";',' const bool request_batch5055=batch_flag5055&&std::string(batch_flag5055)=="1";\n // Constructor-frozen policy: a later environment change cannot toggle storage.\n const bool v398=!projected_value_;\n if(v398&&(resident5640||request_batch5055||host_bank_r4504_||rows_!=1||capacity_!=32768))throw std::invalid_argument("r398 incompatible storage policy");')
 old=':projected_value_->narrow({{0,host_bank_r4504_?size_t{0}:request.row,1}})->view({capacity_,prepare_->local_heads(),256});'
 replace(old,':projected_value_?projected_value_->narrow({{0,host_bank_r4504_?size_t{0}:request.row,1}})->view({capacity_,prepare_->local_heads(),256}):rt::Tensor{};')
 # The normal device-bank append is the only reachable append for candidate.
 old='   }else{\n    key_row->narrow({{0,request.past,count}})->copy_from(r,fresh.first);\n    value_row->narrow({{0,request.past,count}})->copy_from(r,fresh.second);\n   }'
 replace(old,'   }else{\n    key_row->narrow({{0,request.past,count}})->copy_from(r,fresh.first);\n    if(value_row)value_row->narrow({{0,request.past,count}})->copy_from(r,fresh.second);\n   }')
 replace('expanded={key_row->narrow({{0,0,prefix}}),value_row->narrow({{0,0,prefix}})};','expanded={key_row->narrow({{0,0,prefix}}),v398?v_on_demand_r398::selected_values(*prepare_,r,latent_row,rotary_row,ids,work.held):value_row->narrow({{0,0,prefix}})};')
 replace('expanded={work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16),work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16)};','expanded={work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16),v398?v_on_demand_r398::selected_values(*prepare_,r,latent_row,rotary_row,ids,work.held):work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16)};')
 replace('ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),value_row,ids,int64_t{0},expanded.second);','if(!v398)ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),value_row,ids,int64_t{0},expanded.second);')
 return s.encode(),edits
def main():
 raw,edits=derive(P.read_bytes());(D/'caller_v_on_demand_r398.cpp').write_bytes(raw)
 q=dict(parent=str(P),parent_SHA256=SHA,derived_SHA256=hashlib.sha256(raw).hexdigest(),edits=edits,class_header_changed=False,layout_changed=False,default_OFF=True,policy_frozen_at_constructor=True,Model_GO=False)
 (D/'DERIVATION.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf-8');return q
if __name__=='__main__':print(json.dumps(main()))
