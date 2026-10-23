#include "glm-rows32-private-r5731-src/glm_model_resident_policy_r5640.hpp"
#include "glm-rows32-private-r5731-src/baseline_policy_CPU.hpp"
#include "glm-rows32-private-r5731-src/glm_rows32_policy_r5731.hpp"
#include <iostream>
#include <set>
namespace g=glm_model_resident_r5640;
int main(){
 size_t checks=0;auto require=[&](bool v){++checks;if(!v)throw std::runtime_error("CPU contract");};
 static_assert(sizeof(g::Slots)==sizeof(glm_policy_baseline_CPU::Slots));
 static_assert(alignof(g::Slots)==alignof(glm_policy_baseline_CPU::Slots));
 g::Slots slots;std::vector<g::Request> first;
 for(size_t row=0;row<32;row+=8){std::vector<g::Request>q;for(size_t j=row;j<row+8;++j)q.push_back({j,0,512});slots=slots.propose(q,32);}
 require(slots.active()==32);std::set<int>unique;for(size_t i=0;i<32;++i)unique.insert(slots.slot(i));require(unique.size()==32&&*unique.rbegin()==31);
 auto reject=[&](std::vector<g::Request>q,size_t maximum){bool threw=false;try{(void)slots.propose(q,maximum);}catch(const std::invalid_argument&){threw=true;}require(threw);require(slots.active()==32&&slots.length(31)==512);};
 reject({{32,0,1}},32);reject({{0,512,1},{0,512,1}},32);reject({{0,512,1},{31,511,1}},32);reject({{0,512,1}},24);
 for(size_t offset=0;offset<127;++offset){for(size_t row=0;row<32;row+=8){std::vector<g::Request>q;for(size_t j=row;j<row+8;++j)q.push_back({j,512+offset,1});auto next=slots.propose(q,32);require(next.length(row)==513+offset);require(next.length((row+8)%32)==slots.length((row+8)%32));slots=next;}}
 for(size_t i=0;i<32;++i)require(slots.length(i)==639);
 slots.reset();require(slots.empty());auto next=slots.propose({{31,0,512}},32);require(next.slot(31)==0&&next.length(31)==512);
 g::Slots legacy;glm_policy_baseline_CPU::Slots baseline;
 for(size_t row=0;row<16;row+=8){std::vector<g::Request>a;std::vector<glm_policy_baseline_CPU::Request>b;for(size_t j=row;j<row+8;++j){a.push_back({j,0,99});b.push_back({j,0,99});}legacy=legacy.propose(a);baseline=baseline.propose(b);}
 require(legacy.active()==baseline.active());for(size_t row=0;row<16;++row)require(legacy.slot(row)==baseline.slot(row)&&legacy.length(row)==baseline.length(row));
 bool rejected=false;try{(void)legacy.propose({{16,0,99}});}catch(const std::invalid_argument&){rejected=true;}require(rejected);
 require(glm_rows32_r5731::rows_supported(32));require(glm_rows32_r5731::transient_geometry(32,0)&&!glm_rows32_r5731::transient_geometry(32,1));
 require(glm_rows32_r5731::admission_bytes(32,40)==10ull*1024*1024*1024);
 std::cout<<"{\"checks\":"<<checks<<",\"sizeof_original_Slots\":"<<sizeof(glm_policy_baseline_CPU::Slots)<<",\"sizeof_candidate_Slots\":"<<sizeof(g::Slots)<<",\"layout_preserved\":true,\"GPU\":\"NOT_RUN\"}\n";
}
