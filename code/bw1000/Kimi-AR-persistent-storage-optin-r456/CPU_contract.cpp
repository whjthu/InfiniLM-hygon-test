#include "ar_storage_r456.hpp"
#include "long_geometry_r453.hpp"
#include "long_agent_geometry_r179.hpp"
#include "actual_freeze_input_recording.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
static size_t checks=0;
static void require(bool v){if(!v)throw std::logic_error("r456 CPU contract");++checks;}
template<class F>static void reject(F f){try{f();}catch(const std::invalid_argument&){++checks;return;}throw std::logic_error("missing rejection");}
static size_t actual_metadata_consumer(){
 using namespace infinilm::models::kimi_native;namespace rt=infinilm::runtime;
 size_t cases=0;std::vector<size_t>lengths(2,0);int owner=0;
 auto run=[&](int32_t past,int32_t width,bool mutate){
  std::vector<int64_t>ids(width,17),positions(width),slots(width);std::vector<int32_t>rows{1},writes{1},p{past},totals{past+width},off{0,width},cu{0,past+width},table(128);
  for(int t=0;t<width;++t){positions[t]=past+t;slots[t]=(1+(past+t)/256)*256+(past+t)%256;}for(int j=0;j<128;++j)table[j]=j+1;
  auto tensor=[](auto&v,rt::Shape shape,rt::DataType d){return rt::Tensor::from_blob(v.data(),shape,d,{rt::Device::Type::kCpu,0});};
  RecordingInput x;x.input_ids=tensor(ids,{1,size_t(width)},rt::DataType::kInt64);x.position_ids=tensor(positions,{1,size_t(width)},rt::DataType::kInt64);x.slot_mapping=tensor(slots,{size_t(width)},rt::DataType::kInt64);
  x.past_sequence_lengths=tensor(p,{1},rt::DataType::kInt32);x.total_sequence_lengths=tensor(totals,{1},rt::DataType::kInt32);x.input_offsets=tensor(off,{2},rt::DataType::kInt32);x.cu_seqlens=tensor(cu,{2},rt::DataType::kInt32);x.mamba_init_state_indices=tensor(rows,{1},rt::DataType::kInt32);x.mamba_final_state_indices=tensor(writes,{1},rt::DataType::kInt32);x.block_tables=tensor(table,{1,128},rt::DataType::kInt32);
  if(mutate){writes[0]=2;reject([&]{freeze_input(x,lengths,163840,&owner,128,true,true);});writes[0]=1;x.mamba_final_state_indices=tensor(writes,{1},rt::DataType::kInt32);}
  auto requests=freeze_input(x,lengths,163840,&owner,128,true,true);require(requests.size()==1&&requests[0].row==1&&requests[0].past==size_t(past)&&requests[0].tokens==size_t(width));
  ids[0]=99;require(values<int64_t>(x.input_ids.value())[0]==17); // actual freeze_cpu copy owns backing
  commit(lengths,requests);++cases;
 };
 for(int32_t past=0;past<32768;past+=512)run(past,512,past==0);
 require(lengths[1]==32768&&lengths[0]==0);lengths.assign(2,0);run(0,512,false);lengths[1]=32767;run(32767,1,false);
 return cases;
}
int main(){try{
 namespace a=kimi_ar_storage_r456;namespace g=kimi_long_geometry_r453;
 a::cache_geometry(1,132,256);a::topology(true,true,32,1,96);a::state_geometry(2,false);
 // Original speculative long geometry remains2/260; new explicit mode is1/132.
 kimi_long_agent_r179::validate_cache(2,260,256);reject([]{kimi_long_agent_r179::validate_cache(1,132,256);});
 reject([]{a::cache_geometry(2,260,256);});reject([]{a::cache_geometry(1,131,256);});reject([]{a::cache_geometry(1,132,128);});
 reject([]{a::topology(true,false,32,1,96);});reject([]{a::topology(true,true,16,2,192);});reject([]{a::topology(true,true,8,4,384);});
 reject([]{a::state_geometry(3,false);});reject([]{a::state_geometry(2,true);});
 reject([]{a::request(0,1,2,128,false);});reject([]{a::request(1,2,2,128,false);});reject([]{a::request(2,2,2,128,false);});
 std::array<int32_t,128>table;for(size_t p=0;p<128;++p)table[p]=p+1;
 std::vector<size_t>lengths(2,0);
 // Entire unchanged32768 context, active block1..128; state0 reserved, state1
 // active. This is a pure address/index reference, not a forward/SDK simulation.
 for(size_t chunk=0;chunk<64;++chunk){
  size_t past=chunk*512;std::vector<int64_t>positions(512),slots(512);
  for(size_t t=0;t<512;++t){positions[t]=past+t;slots[t]=(1+(past+t)/256)*256+(past+t)%256;require(slots[t]>=256&&size_t(slots[t])<132*256);}
  a::request(1,1,lengths.size(),128,false);g::call_scope(0,1,1,int32_t(past),512,3);
  require(lengths[1]==past);kimi_long_agent_r179::validate_row_table(1,int32_t(past),int32_t(past+512),0,512,128,table.data(),slots.data(),positions.data());lengths.at(1)=past+512;
 }
 require(lengths.at(1)==32768&&lengths.at(0)==0);lengths.assign(2,0);require(lengths[0]==0&&lengths[1]==0);
 // Decode/append last legal slot; page0 and three end guards remain untouched.
 int64_t pos=32767,slot=33023;g::call_scope(1,1,1,32767,1,3);kimi_long_agent_r179::validate_row_table(1,32767,32768,0,1,128,table.data(),&slot,&pos);
 require(size_t(slot)==128*256+255&&slot<132*256&&132*256-1==33791);
 slot=33279;reject([&]{kimi_long_agent_r179::validate_row_table(1,32767,32768,0,1,128,table.data(),&slot,&pos);});
 reject([]{g::call_scope(1,1,1,32768,1,3);});reject([]{g::call_scope(1,1,2,1,1,3);});
 int runtime=0,object=0,foreign=0;
 require(!a::selected(&runtime,&object));{a::Scope scope(true,true,32,1,96,&runtime,&object);require(a::selected(&runtime,&object));reject([&]{a::selected(&runtime,&foreign);});}
 require(a::active==nullptr);try{a::Scope scope(true,true,32,1,96,&runtime,&object);throw std::runtime_error("recorded constructor error");}catch(const std::runtime_error&){require(a::active==nullptr);}
 {a::Scope off(false,false,16,2,192,&runtime,&object);require(!a::selected(&runtime,&object));}
 const size_t metadata=actual_metadata_consumer();
 std::cout<<"{\"passed\":true,\"checks\":"<<checks<<",\"actual_freeze_input_template_cases\":"<<metadata<<",\"Tensor_leaf_recording_not_SDK\":true,\"full_context\":32768,\"active_pages\":128,\"blocks\":132,\"state_rows\":2,\"C1_read_write_row\":1,\"last_active_slot\":33023,\"last_storage_slot\":33791,\"Native_dlopen\":false,\"Runtime_created\":false,\"GPU\":0}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
