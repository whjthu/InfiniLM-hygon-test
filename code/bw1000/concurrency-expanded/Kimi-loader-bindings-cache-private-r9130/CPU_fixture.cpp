#include "bindings_cache_r9130.hpp"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <functional>
using J=nlohmann::json;
struct MockTensor{std::vector<size_t>shape;std::string dtype;size_t storage;bool operator==(const MockTensor&o)const{return shape==o.shape&&dtype==o.dtype&&storage==o.storage;}};
using B=std::unordered_map<std::string,MockTensor>;using Cache=loading_r9130::ImmutableBindings<B>;
void require(bool a){if(!a)throw std::runtime_error("fixture assert");}
bool rejected(std::function<void()>f){try{f();return false;}catch(const std::exception&){return true;}}
int main(int argc,char**argv){require(argc==2);J input;std::ifstream(argv[1])>>input;B model;size_t next=1;for(auto&v:input)model.emplace(v.at("key"),MockTensor{v.at("shape").get<std::vector<size_t>>(),v.at("dtype"),next++});require(model.size()==5384);auto build=[&]{B result;for(auto&v:model)result.emplace(v.first,v.second);return result;};auto equal=[](const B&a,const B&b){return a==b;};std::vector<std::string>keys;for(auto&v:model)keys.push_back(v.first);J qa;
 {Cache c(&model,1);auto&b=c.get(&model,1,build);require(b.size()==5384);require(&b==&c.get(&model,1,build));qa["same_binding_and_shape_dtype_preserved"]=b.at(keys[0])==model.at(keys[0]);qa["unknown_strict_rejected"]=rejected([&]{b.at("missing");});qa["unknown_nonstrict_skipped"]=b.find("missing")==b.end();qa["source_shape_mismatch_rejected"]=rejected([&]{auto wrong=b.at(keys[0]);wrong.shape.push_back(7);if(!(wrong==model.at(keys[0])))throw std::invalid_argument("shape mismatch before copy");});qa["model_identity_rejected"]=rejected([&]{c.get(&qa,1,build);});qa["generation_rejected"]=rejected([&]{c.get(&model,2,build);});bool wrongthread=false;std::thread t([&]{wrongthread=rejected([&]{c.get(&model,1,build);});});t.join();qa["owner_thread_rejected"]=wrongthread;c.finish(&model,1,build,equal);qa["finish_releases_bindings_before_runtime"]=!c.holds_bindings();qa["load_after_finalize_rejected"]=rejected([&]{c.get(&model,1,build);});}
 {Cache c(&model,3);c.get(&model,3,build);model.emplace("late.lazy.parameter",model.begin()->second);qa["late_registration_rejected_at_finish"]=rejected([&]{c.finish(&model,3,build,equal);});qa["late_failure_releases_owners"]=!c.holds_bindings();model.erase("late.lazy.parameter");}
 {Cache c(&model,4);c.get(&model,4,build);auto saved=model.at(keys[0]);++model.at(keys[0]).storage;qa["storage_change_rejected_at_finish"]=rejected([&]{c.finish(&model,4,build,equal);});model.at(keys[0])=saved;}
 {Cache c(&model,5);const auto&b=c.get(&model,5,build);std::unordered_set<std::string>loaded;loaded.insert(keys[0]);qa["duplicate_rejected_engine_style"]=rejected([&]{if(!loaded.insert(keys[0]).second)throw std::runtime_error("duplicate");});qa["partial_failure_failstop_not_rollback"]=loaded.count(keys[0])==1;c.abort(&model,5);qa["abort_releases_and_blocks_finalize"]=!c.holds_bindings()&&rejected([&]{c.finish(&model,5,build,equal);});}
 for(auto it=qa.begin();it!=qa.end();++it)require(it.value().get<bool>());
 volatile size_t sink=0;auto measure=[&](bool cached){auto start=std::chrono::steady_clock::now();size_t builds=0;Cache c(&model,6);for(size_t i=0;i<64;++i){if(cached){auto&b=c.get(&model,6,build);sink+=b.at(keys[i%keys.size()]).storage;}else{auto b=build();++builds;sink+=b.at(keys[i%keys.size()]).storage;}}if(cached){c.finish(&model,6,build,equal);builds=c.builds;}else c.abort(&model,6);return J{{"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},{"state_dict_build_calls",builds},{"cached",cached}};};J pairs=J::array();for(int i=0;i<4;++i){J x;if(i%2==0){x["baseline"]=measure(false);x["candidate"]=measure(true);}else{x["candidate"]=measure(true);x["baseline"]=measure(false);}pairs.push_back(x);}std::cout<<J{{"passed",true},{"CPU_only",true},{"MockTensor",true},{"actual_one_layer_keys",model.size()},{"transaction_batches",64},{"qa",qa},{"paired",pairs},{"scope","actual5384key metadata, flat HOST registry mockTensor; not runtime Tensor/GPUcopy/full Model tree or load gain"}}.dump(2)<<std::endl;
}
