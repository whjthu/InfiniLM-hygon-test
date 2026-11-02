"""Bounded r403 caller edits; shared pair/geometry/state-copy header, no class change."""
def derive(raw):
 text=raw.decode();edits=[]
 def one(old,new):
  nonlocal text
  assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new);edits.append(dict(old=old,new=new))
 one('#include "selected_values_r400.hpp"','#include "selected_kv_r408.hpp"')
 one('  const bool host4504=projected_host_r4504::enabled();','  const bool kv408=kv_on_demand_r408::requested(v398);\n  const bool host4504=projected_host_r4504::enabled();\n  kv_on_demand_r408::scope(kv408,v398,rows,capacity,topk,rank.tp_size,host4504);')
 one('  projected_key_=exact_banks_r4188::allocate(r,{host4504?size_t{1}:rows,capacity,prepare_->local_heads(),256});','  if(!kv408)projected_key_=exact_banks_r4188::allocate(r,{host4504?size_t{1}:rows,capacity,prepare_->local_heads(),256});')
 one('  r.setDeviceMemory(projected_key_->data(),0xff,projected_key_->nbytes());','  if(projected_key_)r.setDeviceMemory(projected_key_->data(),0xff,projected_key_->nbytes());')
 one(' const bool v398=!projected_value_;',' const bool v398=!projected_value_;\n const bool kv408=!projected_key_;\n if(kv408&&!v398)throw std::logic_error("r408 frozen NoK requires NoV");')
 one(':projected_key_->narrow({{0,host_bank_r4504_?size_t{0}:request.row,1}})->view({capacity_,prepare_->local_heads(),256});',':projected_key_?projected_key_->narrow({{0,host_bank_r4504_?size_t{0}:request.row,1}})->view({capacity_,prepare_->local_heads(),256}):rt::Tensor{};')
 one('latent_row->narrow({{0,request.past,count}})->copy_from(r,prepared.kv_latent->narrow({{0,begin,count}}));rotary_row->narrow({{0,request.past,count}})->copy_from(r,prepared.k_rot->narrow({{0,begin,count}})->view({count,64}));','kv_on_demand_r408::append_latent_rotary(r,latent_row,rotary_row,prepared.kv_latent->narrow({{0,begin,count}}),prepared.k_rot->narrow({{0,begin,count}}),request.past,count);')
 one('    key_row->narrow({{0,request.past,count}})->copy_from(r,fresh.first);\n    if(value_row)value_row->narrow({{0,request.past,count}})->copy_from(r,fresh.second);','    kv_on_demand_r408::store_projected(r,key_row,value_row,fresh.first,fresh.second,request.past,count);')
 one('   if(complete_prefix){\n    expanded=', '   if(kv408){\n    expanded=kv_on_demand_r408::selected(*prepare_,r,latent_row,rotary_row,ids,work.held,true);\n    work.held.insert(work.held.end(),{expanded.first,expanded.second});\n   }else if(complete_prefix){\n    expanded=')
 assert 'auto fresh_all=prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot);'in text
 return text.encode(),dict(parent_SHA256='d54b80d68497c26c83a0c63fe87fbec376f4b106daa40d0da61238dfb5914be2',edits=edits,necessary_Model_HOST_TUs=1,headers_class_layout_changed=False,added_sync=0,fresh_projection_and_fence_retained=True,default_OFF=True)
