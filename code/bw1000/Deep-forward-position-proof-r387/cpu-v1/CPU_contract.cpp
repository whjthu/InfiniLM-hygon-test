#include "deep_position_proof_r387.hpp"
#include <iostream>
using namespace deep_position_proof_r387;
extern bool cross_tu_copy(rt::Runtime&,const rt::Tensor&,std::vector<int64_t>&);
extern "C" int deep_position_proof_set_r387(int);
extern "C" int deep_position_proof_get_r387();
extern "C" void deep_position_proof_clear_r387();
extern "C" int deep_position_proof_stats_r387(int,uint64_t*,size_t);
int main(){
 int checks=0;auto ok=[&](bool x){if(!x)throw std::runtime_error("CPU387 contract");++checks;};
 rt::Runtime r,other;auto p=std::make_shared<rt::Value>(),q=std::make_shared<rt::Value>();
 auto alias=p;std::vector<int64_t>v{8496,8497,8498},out(3,-99);
 ok(deep_position_proof_get_r387()==0&&!cross_tu_copy(r,p,out));{Scope s(r,p,v);ok(!cross_tu_copy(r,p,out));}ok(active==nullptr);
 ok(deep_position_proof_set_r387(2)==-1&&deep_position_proof_get_r387()==0);
 ok(deep_position_proof_set_r387(1)==0);{Scope s(r,p,v);ok(cross_tu_copy(r,p,out)&&out==v);ok(!cross_tu_copy(other,p,out));
  ok(!cross_tu_copy(r,alias,out));ok(!cross_tu_copy(r,q,out));
  r.s=(void*)0x20;ok(!cross_tu_copy(r,p,out));r.s=(void*)0x10;
  r.graph=true;ok(!cross_tu_copy(r,p,out));r.graph=false;
  p->contiguous=false;ok(!cross_tu_copy(r,p,out));p->contiguous=true;
  p->count=2;ok(!cross_tu_copy(r,p,out));p->count=3;
  p->rank=2;ok(!cross_tu_copy(r,p,out));p->rank=1;
  p->type=rt::DataType::kFloat32;ok(!cross_tu_copy(r,p,out));p->type=rt::DataType::kInt64;
  p->dev.n=1;ok(!cross_tu_copy(r,p,out));p->dev.n=0;
  p->offset=1;ok(!cross_tu_copy(r,p,out));p->offset=0;
  out.resize(2);ok(!cross_tu_copy(r,p,out));out.resize(3);
  v.push_back(8499);ok(!cross_tu_copy(r,p,out));v.pop_back();
  std::vector<int64_t>w{8575,8576,8577};{Scope nested(r,q,w);ok(cross_tu_copy(r,q,out)&&out==w);ok(!cross_tu_copy(r,p,out));}
  ok(cross_tu_copy(r,p,out)&&out==v);
  try{Scope nested(r,q,w);throw std::runtime_error("unwind");}catch(const std::runtime_error&){}
  ok(cross_tu_copy(r,p,out)&&out==v);
  r.owner=false;bool rejected=false;try{cross_tu_copy(r,p,out);}catch(...){rejected=true;}r.owner=true;ok(rejected);
 }ok(active==nullptr&&!cross_tu_copy(r,p,out));ok(counts[0][1].load()==4&&counts[0][2].load()==96);
 ok(deep_position_proof_set_r387(0)==0);{Scope s(r,p,v);ok(!cross_tu_copy(r,p,out));}ok(active==nullptr);
 uint64_t stats[5]{};ok(deep_position_proof_stats_r387(0,stats,5)==0&&stats[1]==4&&stats[2]==96);
 ok(deep_position_proof_stats_r387(-1,stats,5)==-1&&deep_position_proof_stats_r387(8,stats,5)==-1);
 ok(deep_position_proof_stats_r387(0,nullptr,5)==-1&&deep_position_proof_stats_r387(0,stats,4)==-1);
 deep_position_proof_clear_r387();ok(deep_position_proof_get_r387()==0&&deep_position_proof_stats_r387(0,stats,5)==0);
 ok(stats[0]==0&&stats[1]==0&&stats[2]==0&&stats[3]==0&&stats[4]==0);
 std::cout<<"{\"passed\":true,\"checks\":"<<checks<<",\"cross_TU\":true,\"Runtime_mock_only\":true}\n";
}
