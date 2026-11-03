// Original header control/CPU-byte CABI only. No Runtime is constructed or
// method called; the seven records below are explicitly recording fixtures.
#include "boundary_capture.hpp"
#include <cassert>
#include <iostream>
int main(){
 using namespace deep447;
 assert(!armed.load()&&!sealed.load()&&!active());
 assert(deep_boundary_arm_r447(2,82,91)==1);
 assert(deep_boundary_arm_r447(0,82,92)==1);
 assert(deep_boundary_arm_r447(0,82,91)==0);
 assert(deep_boundary_arm_r447(0,82,91)==2);
 assert(deep_boundary_seal_r447()==2);
 for(size_t owner=0;owner<8;++owner){
  auto&s=stores[owner];s.records.resize(7);s.completed=true;
  for(size_t p=0;p<7;++p){s.records[p].info[3]=owner;s.records[p].info[5]=p;s.records[p].bytes.assign(2,static_cast<unsigned char>(p));}
 }
 assert(deep_boundary_seal_r447()==0);
 for(int owner=0;owner<8;++owner){
  assert(deep_boundary_count_r447(owner)==7);
  for(size_t p=0;p<7;++p){uint64_t v[15]{};unsigned char b[2]{};
   assert(deep_boundary_info_r447(owner,p,v,15)==0&&v[3]==uint64_t(owner)&&v[5]==p);
   assert(deep_boundary_copy_r447(owner,p,b,2)==0&&b[0]==p);
   assert(deep_boundary_copy_r447(owner,p,b,1)==2);
  }
 }
 assert(deep_boundary_clear_r447()==0);
 assert(deep_boundary_count_r447(0)==UINT64_MAX);
 assert(deep_boundary_arm_r447(1,83,92)==0);
 stores[0].active.store(true);assert(deep_boundary_abandon_r447()==1);stores[0].active.store(false);
 assert(deep_boundary_abandon_r447()==0&&deep_boundary_clear_r447()==0);
 std::cout<<"{\"passed\":true,\"recording_CPU_bytes_only\":true,\"Runtime_created\":false,\"GPU\":0}"<<std::endl;
}
