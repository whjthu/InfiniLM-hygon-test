#include "glm_vector_dispatch_r8540.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
extern "C" int modern_glm_vector_policy_r8540(int,int);
extern "C" int modern_glm_vector_get_r8540(int);
extern "C" int modern_glm_down_row2_policy_r9284(int,int);
extern "C" int modern_glm_down_row2_get_r9284(int);
extern "C" int modern_glm_M1_gate_policy_r367(int,int);
extern "C" int modern_glm_M1_gate_get_r367(int);
extern "C" int modern_glm_M1_gate_counts_r367(int,std::uint64_t*,std::size_t);
static int path=-1,ret=0;
extern "C" int glm_vec_gate_r8528(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){path=1;return ret;}
extern "C" int glm_grouped_gate_r5123(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){path=0;return ret;}
extern "C" int glm_vec_down_reference_r9281(void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){path=1;return ret;}
extern "C" int glm_vec_down_row2_r9281(void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){path=2;return ret;}
extern "C" int glm_grouped_down_r5123(void*,const void*,const void*,const void*,const void*,const void*,std::size_t,std::size_t,void*){path=0;return ret;}
int main(){
 assert(sizeof(glm_vector_r8540::Frame)==8);unsigned checks=0;
 for(int d=0;d<8;++d){
  assert(modern_glm_M1_gate_get_r367(d)==0&&modern_glm_vector_get_r8540(d)==0&&modern_glm_down_row2_get_r9284(d)==0);
  for(int vector:{0,1})for(int row2:{0,1})for(int m1:{0,1}){
   assert(modern_glm_vector_policy_r8540(d,vector)==0&&modern_glm_down_row2_policy_r9284(d,row2)==0&&modern_glm_M1_gate_policy_r367(d,m1)==0);
   assert(modern_glm_vector_get_r8540(d)==vector&&modern_glm_down_row2_get_r9284(d)==row2&&modern_glm_M1_gate_get_r367(d)==m1);
   for(std::size_t T:{1,2,3,4,5,7,8,13,15,16,17,64,512}){
    std::uint64_t a[4],b[4];assert(modern_glm_M1_gate_counts_r367(d,a,4)==0);
    {glm_vector_r8540::Frame f(d);assert(f.enabled==bool(vector));
     assert(modern_glm_vector_policy_r8540(d,0)==-2&&modern_glm_down_row2_policy_r9284(d,0)==-2&&modern_glm_M1_gate_policy_r367(d,0)==-2);
     {glm_vector_r8540::Frame nested(d);assert(modern_glm_M1_gate_policy_r367(d,0)==-2);}
     assert(modern_glm_M1_gate_policy_r367(d,0)==-2);
     bool old=vector&&(T==4||T==8||T==16);path=-1;
     assert(glm_vector_r8540::gate(f,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,T*8,0,nullptr)==0&&path==int(old||(T==1&&m1)));
     path=-1;assert(glm_vector_r8540::down(f,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,T*8,0,nullptr)==0&&path==(old?(row2&&(T==8||T==16)?2:1):0));
    }
    assert(modern_glm_M1_gate_counts_r367(d,b,4)==0&&b[0]-a[0]==unsigned(T==1&&m1)&&b[1]-a[1]==unsigned(T==1&&!m1)&&b[2]==a[2]);++checks;
   }
  }
  assert(modern_glm_vector_policy_r8540(d,0)==0&&modern_glm_down_row2_policy_r9284(d,0)==0&&modern_glm_M1_gate_policy_r367(d,0)==0);
 }
 for(int d:{-1,8})assert(modern_glm_M1_gate_policy_r367(d,0)==-1&&modern_glm_M1_gate_get_r367(d)==-1);
 assert(modern_glm_M1_gate_policy_r367(0,2)==-1&&modern_glm_M1_gate_counts_r367(0,nullptr,4)==-1);
 std::uint64_t counts[4]={91,92,93,94};assert(modern_glm_M1_gate_counts_r367(0,counts,3)==-1&&counts[0]==91);
 assert(modern_glm_M1_gate_policy_r367(0,1)==0&&modern_glm_M1_gate_get_r367(1)==0);
 ret=7;{glm_vector_r8540::Frame f(0);assert(glm_vector_r8540::gate(f,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,8,0,nullptr)==7);}
 assert(modern_glm_M1_gate_counts_r367(0,counts,4)==0&&counts[2]==1);assert(modern_glm_M1_gate_policy_r367(0,0)==0);
 bool rejected=false;try{glm_vector_r8540::Frame invalid(8);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
 std::cout<<"{\"passed\":true,\"actual_dispatch_checks\":"<<checks<<",\"CPU_only\":true,\"GPU_math_test\":false,\"Frame_ABI\":8,\"M1_down_unchanged\":true,\"default_OFF\":true}";
}
