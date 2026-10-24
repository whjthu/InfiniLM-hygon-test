#include "deep_position_proof_r8726.hpp"
#include <cassert>
#include <iostream>
using namespace deep_position_proof_r8726;
int main(){int checks=0;auto ok=[&](bool x){if(!x)throw std::runtime_error("CPU contract");++checks;};
 rt::Runtime r,other;auto p=std::make_shared<rt::Value>(),alias=std::make_shared<rt::Value>();std::vector<int64_t>v{63,64,65},out(3,-9);
 unsetenv("INFINILM_DEEP_POSITION_PROOF_R8726");ok(!copy(r,p,out));{Scope s(r,p,v);ok(!copy(r,p,out));}ok(active==nullptr);
 setenv("INFINILM_DEEP_POSITION_PROOF_R8726","1",1);{Scope s(r,p,v);ok(copy(r,p,out)&&out==v);ok(!copy(other,p,out));ok(!copy(r,alias,out));
 r.s=(void*)0x20;ok(!copy(r,p,out));r.s=(void*)0x10;r.graph=true;ok(!copy(r,p,out));r.graph=false;
 p->contiguous=false;ok(!copy(r,p,out));p->contiguous=true;p->count=2;ok(!copy(r,p,out));p->count=3;p->type=rt::DataType::kFloat32;ok(!copy(r,p,out));p->type=rt::DataType::kInt64;p->dev.n=1;ok(!copy(r,p,out));p->dev.n=0;
 out.resize(2);ok(!copy(r,p,out));out.resize(3);std::vector<int64_t>w{1,2,3};{Scope nested(r,alias,w);ok(copy(r,alias,out)&&out==w);ok(!copy(r,p,out));}ok(copy(r,p,out)&&out==v);
 try{Scope nested(r,alias,w);throw std::runtime_error("unwind");}catch(const std::runtime_error&){}ok(copy(r,p,out)&&out==v);
 r.owner=false;bool rejected=false;try{copy(r,p,out);}catch(...){rejected=true;}r.owner=true;ok(rejected);
 }ok(active==nullptr);ok(!copy(r,p,out));ok(counts[0][0].load()==4);ok(counts[0][1].load()==4*3*8);
 setenv("INFINILM_DEEP_POSITION_PROOF_R8726","invalid",1);bool rejected=false;try{Scope s(r,p,v);}catch(const std::invalid_argument&){rejected=true;}ok(rejected&&active==nullptr);
 setenv("INFINILM_DEEP_POSITION_PROOF_R8726","0",1);{Scope s(r,p,v);ok(!copy(r,p,out));}ok(active==nullptr);
 std::cout<<"{\"passed\":true,\"checks\":"<<checks<<"}\n";
}
