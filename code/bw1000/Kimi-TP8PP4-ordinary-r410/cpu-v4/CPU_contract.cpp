#include "kimi_shard_head_r9183.hpp"
#include <iostream>
int main(){
 using namespace kimi_shard_r9183;size_t checks=0;
 for(size_t TP:{size_t(8),size_t(16)}){
  auto g=head_geometry_r410(TP);
  if(g.rows!=163840/TP||g.bytes!=g.rows*7168*2||g.samples!=std::array<size_t,3>{0,g.rows/2,g.rows-1})return 1;
  json ranks=json::array();
  for(size_t rank=0;rank<TP;++rank){ranks.push_back({{"rank",rank},{"row_begin",rank*g.rows},{"digests",{rank,rank+1,rank+2}}});++checks;}
  expected_ready=false;set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});
  if(!expected_ready||expected_tp!=TP)return 2;
  try{set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});return 3;}catch(const std::logic_error&){++checks;}
  expected_ready=false;ranks[0]["row_begin"]=1;
  try{set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});return 4;}catch(const std::invalid_argument&){++checks;}
  ranks[0]["row_begin"]=0;ranks[TP-1]["rank"]=0;
  try{set_expected({{"key","language_model.lm_head.weight"},{"ranks",ranks}});return 5;}catch(const std::invalid_argument&){++checks;}
 }
 for(size_t TP:{size_t(0),size_t(4),size_t(32)})try{head_geometry_r410(TP);return 6;}catch(const std::invalid_argument&){++checks;}
 std::cout<<json({{"passed",true},{"actual_header",true},{"TP",{8,16}},{"checks",checks},{"device_calls",0},{"Native_dlopen",false}}).dump()<<"\n";
}
