#include "models/kimi_k3/kimi_k3_trace.hpp"
#include <infini/rt/device_guard.h>
#include <string>
namespace rt=infinilm::runtime;
namespace {struct Fixture{std::shared_ptr<rt::Runtime>r;};thread_local std::string error;}
extern "C" const char*taps_fixture_error_r9023(){return error.c_str();}
extern "C" void*taps_fixture_create_r9023(){try{infini::rt::DeviceGuard g{rt::Device::Type::kHygon,0,true};auto*c=new Fixture{rt::Runtime::create({rt::Device::Type::kHygon,0})};c->r->enableOwnerDrainFailStop();return c;}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int taps_fixture_call_r9023(void*ctx,void**p,int after){try{
 auto*c=static_cast<Fixture*>(ctx);if(!c||!p||(after!=0&&after!=1))throw std::invalid_argument("9023 fixture input");auto&r=*c->r;r.assertOwnerThread();
 auto wrap=[&](size_t i,rt::Shape shape,rt::DataType dt){return rt::Tensor::from_blob(p[i],shape,dt,r.device());};
 auto x=wrap(0,{1,8,7168},rt::DataType::kBFloat16),ssm=wrap(1,{17,6,128,128},rt::DataType::kBFloat16),conv=wrap(2,{17,2304,3},rt::DataType::kBFloat16);
 infinilm::global_state::ForwardContext fc;fc.runtime=&r;fc.attn_metadata.past_sequence_lengths=wrap(3,{1},rt::DataType::kInt32);fc.attn_metadata.total_sequence_lengths=wrap(4,{1},rt::DataType::kInt32);fc.attn_metadata.input_offsets=wrap(5,{2},rt::DataType::kInt32);fc.mamba_metadata.input_offsets=fc.attn_metadata.input_offsets;fc.mamba_metadata.init_state_indices=wrap(6,{1},rt::DataType::kInt32);fc.mamba_metadata.final_state_indices=wrap(7,{1},rt::DataType::kInt32);fc.ssm_state_vec.resize(21);fc.conv_state_vec.resize(21);fc.ssm_state_vec[20]=ssm;fc.conv_state_vec[20]=conv;
 using infinilm::models::kimi_k3::trace_hidden;
 if(!after)trace_hidden(r,x,fc,20,"attn_norm");else{trace_hidden(r,x,fc,20,"attn_output");trace_hidden(r,x,fc,20,"exit");trace_hidden(r,x,fc,93,"final_norm");for(size_t l:{19,37,66,78,90})trace_hidden(r,x,fc,l,"decoder");}
 r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int taps_fixture_close_r9023(void*ctx){auto*c=static_cast<Fixture*>(ctx);try{if(!c)return 1;c->r->assertOwnerThread();c->r->syncStream();c->r->trimMemory();delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
