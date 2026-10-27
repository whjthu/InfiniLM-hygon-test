// CPU-only C++ test of the ACTUAL transaction header; intentionally UNCOMPILED
// until root authorizes compilation. No fake native tensor/backend pass.
#include "deep_prefix_transaction_r124.hpp"
#include "prefix_decision_r124.hpp"
#include <cassert>
using namespace infinilm::models::deepseek_v4::prefix_r124;
template<class F>void rejects(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}assert(threw);}
int main(){
 int owner=0;auto e=std::make_shared<Epoch>();std::vector<size_t>length(2,127);
 auto t=std::make_shared<Transaction>(&owner,e,std::vector<Row>{{0,127,2},{1,127,2}});
 {Scope s(t.get());lengths(length,t->rows());for(size_t i=0;i<86;++i)t->stage([](const Counts&)->Publish{return []{};});for(size_t i=0;i<43;++i)t->decoder_done(i);}
 assert(length[0]==127);t->forward_done();t->seal({1,2});assert(length[1]==127);t->publish();assert(length[0]==128&&length[1]==129);rejects([&]{t->publish();});
 auto cancel=std::make_shared<Transaction>(&owner,e,std::vector<Row>{{0,128,2}});cancel->cancel();assert(length[0]==128);rejects([&]{cancel->check();});
 auto stale=std::make_shared<Transaction>(&owner,e,std::vector<Row>{{0,128,2}});++e->value;rejects([&]{stale->check();});
 auto foreign=std::make_shared<Transaction>(&owner,e,std::vector<Row>{{0,128,2}});int other=0;rejects([&]{foreign->check_owner(&other);});
 auto failure=std::make_shared<Transaction>(&owner,e,std::vector<Row>{{0,128,2}});int published=0;failure->stage([&](const Counts&){return [&]{++published;};});failure->stage([](const Counts&)->Publish{throw std::runtime_error("injected prepare failure");});for(size_t i=0;i<85;++i)failure->stage([](const Counts&)->Publish{return []{};});for(size_t i=0;i<43;++i)failure->decoder_done(i);failure->forward_done();rejects([&]{failure->seal({1});});assert(!published);failure->cancel();
 Slots<int>base;auto a=base.with(0,10).with(127,20);auto b=a.with(0,30).with(128,40);assert(!base.find(0)&&*a.find(0)==10&&*b.find(0)==30&&*a.find(127)==20&&!a.find(128));a.swap(b);assert(*a.find(128)==40);
 assert(!decide(true,false,false,false,4).verify);
 assert(decide(false,false,true,false,4).end==End::None);
 assert(decide(false,true,true,false,4).end==End::EOS);
 assert(decide(false,false,false,true,4).committed_inputs==1);
 assert(decide(false,true,false,false,1).end==End::Cap);
}
