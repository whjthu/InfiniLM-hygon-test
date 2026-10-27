#include "deep_prefix_transaction_r128.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
using namespace infinilm::models::deepseek_v4::prefix_r128;
template<class F>void reject(F f){bool caught=false;try{f();}catch(...){caught=true;}assert(caught);}
void ready(Transaction&t,std::atomic<int>&published){
 for(size_t l=0;l<43;++l){t.decoder_done(l);t.stage([&](const Counts&){return [&]{++published;};});t.stage([&](const Counts&){return [&]{++published;};});}
 t.stage([&](const Counts&){return [&]{++published;};});t.forward_done();
}
void owners(int fault){
 Group g(7,2);if(fault==3)g.configure_fault_r129(3,2);std::atomic<int>published{0},errors{0};std::vector<std::thread>threads;
 for(size_t rank=0;rank<8;++rank)threads.emplace_back([&,rank]{try{
  int owner;auto epoch=std::make_shared<Epoch>();Transaction tx(&owner,epoch,{{0,127,2},{1,128,2}},7,2);ready(tx,published);
  if(fault==1&&rank==3)throw std::runtime_error("injected before seal");
  tx.seal(fault==2&&rank==7?Counts{2,2}:Counts{1,2});
  auto permit=g.arrive_sealed(rank,tx.stamp());assert(g.counts()[0]==8);
  g.before_publish_r129(rank);
  tx.publish(permit);g.published(permit);
 }catch(...){++errors;g.poison("injected or peer failure");}});
 for(auto&t:threads)t.join();
 if(!fault){assert(g.complete()&&!g.poisoned()&&published==8*87&&errors==0);}
 else{assert(g.poisoned()&&!g.complete()&&errors>0);if(fault!=3)assert(published==0);else assert(published>0);}
}
int main(){
 owners(0);owners(1);owners(2);owners(3);
 Group cancel(10,1);std::atomic<int>published{0};cancel.start_cancel();
 for(size_t rank=0;rank<8;++rank){int owner;auto e=std::make_shared<Epoch>();Transaction tx(&owner,e,{{0,2,2}},10,1);ready(tx,published);tx.cancel();cancel.cancelled(rank);assert(e->value==1);reject([&]{tx.seal({1});});}
 assert(cancel.fully_cancelled()&&published==0);
 int owner;auto e=std::make_shared<Epoch>();Transaction stale(&owner,e,{{0,126,2}},11,1);++e->value;reject([&]{stale.check();});
 auto e2=std::make_shared<Epoch>();Transaction missing(&owner,e2,{{0,3,2}},12,1);reject([&]{missing.check_owner(&e2);});reject([&]{missing.forward_done();});
 reject([&]{Transaction bad(&owner,e2,{{0,3,2},{0,3,2}},13,1);});
 std::cout<<"{\"passed\":true,\"scope\":\"Actual HOST transaction/group contracts only; no native GPU/state or model correctness proof\",\"cases\":9}"<<std::endl;
}
