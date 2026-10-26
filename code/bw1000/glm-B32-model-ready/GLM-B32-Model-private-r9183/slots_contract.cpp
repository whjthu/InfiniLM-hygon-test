#include "glm_slots_proposal_r9183.hpp"
#include <iostream>
#include <thread>
#include <cassert>
using namespace glm_model_resident_r5640;
int main(){Slots base;std::vector<Request>all;for(size_t i=0;i<32;++i){size_t row=i*13%32;all.push_back({row,0,55+row});}base=propose32_r9183(base,all,32);assert(base.active()==32);int tests=0;for(size_t B:{size_t(1),size_t(16),size_t(17),size_t(31),size_t(32)}){auto v=all;v.resize(B);for(auto&q:v){q.past=q.tokens;q.tokens=1;}auto next=propose32_r9183(base,v,32);for(auto q:v)assert(next.length(q.row)==q.past+1&&base.length(q.row)==q.past);++tests;}
int bad=0;for(int k=0;k<4;++k){auto v=all;for(auto&q:v){q.past=q.tokens;q.tokens=1;}if(k==0)v[20]=v[0];if(k==1)v[20].row=64;if(k==2)v[20].past++;if(k==3)v[20].tokens=640;try{propose32_r9183(base,v,32);}catch(const std::invalid_argument&){++bad;}for(auto q:all)assert(base.length(q.row)==q.tokens);}assert(bad==4);bool foreign=false;std::thread t([&]{try{propose32_r9183(base,all,32);}catch(const std::invalid_argument&){foreign=true;}});t.join();assert(foreign);base.reset();assert(base.empty());std::cout<<"{\"passed\":true,\"actualSlotsHeader\":true,\"batch_tests\":"<<tests<<",\"invalid_latechunk_preserved\":"<<bad<<",\"foreign_owner_rejected\":true,\"reset_cleared\":true}";}
