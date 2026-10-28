#include "kimi_pair_state_r9006.hpp"
#include "batch_payload_r9122c.hpp"
#include "long_agent_geometry_r179.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <numeric>
template<class F>void reject(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}assert(bad);}
int main(){
 int owner=1,wrong=2;size_t checks=0,full_chunks=0;
 for(size_t C:{1ul,2ul,8ul,16ul,32ul}){kimi_pair_r9006::Geometry old(C);assert(old.context==1024&&old.physical==std::max(size_t(16),2*C)&&old.pages()==4*(old.physical+1));++checks;}
 reject([&]{kimi_pair_r9006::Geometry bad(2,true);});
 for(size_t prompt:{512ul,513ul,8192ul,17408ul,24576ul,32767ul}){
  kimi_pair_r9006::State s(&owner,1,true);assert(s.geometry.physical==2&&s.geometry.state_rows()==3&&s.geometry.context==32768&&s.geometry.pages()==260);s.prefilled(&owner,1,512);reject([&]{s.preflight_prefill(&owner,1,1);});size_t past=512;std::vector<int64_t>observed(512);std::iota(observed.begin(),observed.end(),int64_t(0));
  while(past<prompt){size_t width=std::min(size_t(8),prompt-past);auto row=s.row(&owner,1);std::vector<int64_t>ids(width);std::iota(ids.begin(),ids.end(),int64_t(past));auto before=row;
   reject([&]{s.check(&wrong,1,int(past),row.active,row.shadow,ids);});assert(s.row(&owner,1).past==before.past);
   s.check(&owner,1,int(past),row.active,row.shadow,ids);s.completed(&owner,1,ids);auto invalid=s;reject([&]{invalid.commit(&owner,1,width,true);});assert(invalid.row(&owner,1).phase==2);
   s.commit(&owner,1,width-1,true);observed.insert(observed.end(),ids.begin(),ids.end());past+=width;auto done=s.row(&owner,1);assert(done.past==int(past)&&done.phase==1&&done.active==row.shadow&&done.shadow==row.active);++full_chunks;
  }
  assert(observed.size()==prompt);for(size_t i=0;i<prompt;++i)assert(observed[i]==int64_t(i));++checks;
 }
 kimi_pair_r9006::State edge(&owner,1,true);edge.prefilled(&owner,1,512);while(edge.row(&owner,1).past<32768){auto r=edge.row(&owner,1);size_t width=std::min(size_t(8),size_t(32768-r.past));edge.completed(&owner,1,std::vector<int64_t>(width,17));edge.commit(&owner,1,width-1,true);}reject([&]{auto r=edge.row(&owner,1);edge.check(&owner,1,32768,r.active,r.shadow,{17});});
 kimi_pair_r9006::State old(&owner,8);old.prefilled(&owner,1,1024);reject([&]{auto r=old.row(&owner,1);old.check(&owner,1,1024,r.active,r.shadow,{17});});
 for(int64_t op:{1,2,3,4}){std::vector<int64_t>v{91220001,1,op,op<=2?1:0,op<=2?1:0,op<=2?8:0,3,0};if(op<=2)v.insert(v.end(),{1,32760,1,2,0,8,op==1?0:8});kimi_batch_payload_r9122c::validate(v.data(),v.size(),3,op,32768);reject([&]{kimi_batch_payload_r9122c::validate(v.data(),v.size(),3,op);});if(op<=2){auto bad=v;bad[9]=32761;reject([&]{kimi_batch_payload_r9122c::validate(bad.data(),bad.size(),3,op,32768);});bad=v;bad[10]=2;bad[11]=2;reject([&]{kimi_batch_payload_r9122c::validate(bad.data(),bad.size(),3,op,32768);});}++checks;}
 using namespace kimi_long_agent_r179;validate_cache(2,260,256);reject([&]{validate_cache(2,259,256);});reject([&]{validate_cache(16,260,256);});std::vector<int32_t>table(128);std::iota(table.begin(),table.end(),1);int64_t position=32767,slot=(1+position/256)*256+position%256;validate_row_table(1,32767,32768,0,1,128,table.data(),&slot,&position);
 auto bad=table;bad[127]=129;reject([&]{validate_row_table(1,32767,32768,0,1,128,bad.data(),&slot,&position);});reject([&]{validate_row_table(1,32767,32769,0,2,128,table.data(),&slot,&position);});reject([&]{validate_row_table(2,32767,32768,0,1,128,table.data(),&slot,&position);});++slot;reject([&]{validate_row_table(1,32767,32768,0,1,128,table.data(),&slot,&position);});--slot;++position;reject([&]{validate_row_table(1,32767,32768,0,1,128,table.data(),&slot,&position);});
 std::cout<<"{\"passed\":true,\"CPU_only\":true,\"GPU\":false,\"full_true_input_chunks\":"<<full_chunks<<",\"context\":32768,\"old_default_context\":1024,\"bad_table_past_slot_rejected\":true,\"model_math_qualified\":false}"<<std::endl;
}
