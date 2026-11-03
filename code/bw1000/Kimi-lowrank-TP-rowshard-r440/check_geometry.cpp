#include "geometry.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
int main(){
 int cases=0;
 for(int which:{0,1}){
  std::size_t end=0;
  for(std::size_t rank=0;rank<16;++rank){
   auto g=kimi_r440::geometry(which,rank,16);
   if(g.row_begin!=end||g.local_n*16!=g.n)throw std::runtime_error("row cover/order");
   end+=g.local_n;++cases;
   if(!kimi_r440::eligible(1,g.k,16,which)||kimi_r440::eligible(2,g.k,16,which)||kimi_r440::eligible(1,g.k,8,which))throw std::runtime_error("C1 fallback");
  }
  auto g=kimi_r440::geometry(which,0,16);if(end!=g.n)throw std::runtime_error("full row cover");
 }
 for(auto triple:{std::array<int,3>{0,16,16},std::array<int,3>{1,0,8},std::array<int,3>{2,0,16}}){
  bool rejected=false;try{kimi_r440::geometry(triple[0],triple[1],triple[2]);}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected){throw std::runtime_error("invalid TP role accepted");}
  ++cases;
 }
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"device_or_collective_execution\":false}"<<std::endl;
}
