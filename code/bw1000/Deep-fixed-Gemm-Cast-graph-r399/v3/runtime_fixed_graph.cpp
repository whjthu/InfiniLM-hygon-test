#include "contract.hpp"
#include "provider.hpp"
#include "execution/submit.hpp"
#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
namespace deep399 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace gr=infinilm::execution::graph;
thread_local std::string last_error,report;
struct Context {
 Contract state;
 std::shared_ptr<rt::Runtime> runtime;
 rt::Tensor a,b,y,z,a_storage,b_storage,y_storage,z_storage;
 std::shared_ptr<rt::Memory> workspace;
 std::shared_ptr<PreparedOps> ops;
 std::shared_ptr<gr::Graph> graph;
 std::vector<float> host_a,host_b;
 bool graph_recording=false,inputs_ready=false;
 void assert_owner(){state.owner();runtime->assertOwnerThread();}
 void drain(){runtime->syncStream();}
 void poison()noexcept{state.fail();try{if(graph_recording){runtime->cancelGraphRecording();graph_recording=false;}drain();}catch(...){std::terminate();}}
 void on_error()noexcept{state.owner_cleanup([&]{poison();});}
 void inputs(const float*ap,size_t an,const float*bp,size_t bn){
  state.live();assert_owner();if(!ap||!bp||an!=16*2048||bn!=2048*512)throw std::invalid_argument("r399 input sizes");
  // Host storage is owned until the original stream drain even on copy failure.
  host_a.assign(ap,ap+an);host_b.assign(bp,bp+bn);
  runtime->memcpyH2D(a->data(),host_a.data(),an*4,false);runtime->memcpyH2D(b->data(),host_b.data(),bn*4,false);
  // Changed-input qualification cannot pass on the previous warm output if a
  // vendor capture records no GEMM or Cast work. Poisoning is outside timing.
  std::vector<float> yp(16*512,std::numeric_limits<float>::quiet_NaN());std::vector<uint16_t>zp(16*512,0x7fc0);
  runtime->memcpyH2D(y->data(),yp.data(),yp.size()*4,false);runtime->memcpyH2D(z->data(),zp.data(),zp.size()*2,false);inputs_ready=true;
 }
 void prepare(){state.prepare([&]{assert_owner();infini::rt::DeviceGuard guard{runtime->device().type(),runtime->device().index(),true};if(!inputs_ready||runtime->isGraphRecording())throw std::logic_error("r399 eager preparation with supplied input required");ops->prepare(runtime->stream(),workspace->data()+256,workspace->size()-512);drain();});}
 void check(){state.live();assert_owner();drain();
  std::vector<float> aa(host_a.size()),bb(host_b.size());runtime->memcpyD2H(aa.data(),a->data(),aa.size()*4);runtime->memcpyD2H(bb.data(),b->data(),bb.size()*4);
  if(aa!=host_a||bb!=host_b)throw std::runtime_error("r399 input mutated");
  std::vector<unsigned char> expected(256,0x5a),got(256);
  runtime->memcpyD2H(got.data(),workspace->data(),256);if(got!=expected)throw std::runtime_error("r399 workspace leading guard");
  runtime->memcpyD2H(got.data(),workspace->data()+workspace->size()-256,256);if(got!=expected)throw std::runtime_error("r399 workspace trailing guard");
  for(const auto&store:{a_storage,b_storage,y_storage,z_storage}){size_t tail=store->size(1)*rt::dsize(store->dtype());std::vector<unsigned char>v(tail);runtime->memcpyD2H(v.data(),store->data()+store->nbytes()-tail,tail);if(v!=std::vector<unsigned char>(tail,0x5a))throw std::runtime_error("r399 tensor tail guard");}
 }
 void capture(){state.capture([&]{
  assert_owner();infini::rt::DeviceGuard capture_guard{runtime->device().type(),runtime->device().index(),true};if(!ops->capture_ready())throw std::logic_error("r399 unprepared provider");
  record_two([&]{runtime->startGraphRecording(true);graph_recording=true;},[&]{
   // All buffer snapshots use the actual Runtime's blob/retention contract.
   if(!runtime->retainGraphMemory(workspace))throw std::logic_error("r399 Runtime workspace cannot be pinned");
   return std::make_tuple(ex::detail::snapshot(*runtime,a),ex::detail::snapshot(*runtime,b),ex::detail::snapshot(*runtime,y),ex::detail::snapshot(*runtime,z));
  },[&](const auto&recorded,int operation){
   auto lease=ops;auto mem=workspace;auto owner=runtime;auto views=recorded;
   runtime->addGraphOperator(std::make_shared<gr::RecordedCall>([lease,mem,owner,views,operation]{
    owner->assertOwnerThread();infini::rt::DeviceGuard guard{owner->device().type(),owner->device().index(),true};
    if(operation)lease->cast(owner->stream());else lease->gemm(owner->stream());
   },ops->capture_ready(),false));
  },[&]{graph=runtime->stopGraphRecording();graph_recording=false;},[&]{
   const auto q=graph->execution_report();if(q.operator_count!=2||q.device_graph_segments!=1||q.host_segments!=0||!q.capture_attempted||!q.fallback_reason.empty())throw std::logic_error("r399 requires two real captured operators, no host replay");
  },[&]{runtime->cancelGraphRecording();graph_recording=false;});
 });}
 double run(int mode,int loops){
  assert_owner();infini::rt::DeviceGuard run_guard{runtime->device().type(),runtime->device().index(),true};drain();const auto tick=std::chrono::steady_clock::now();
  state.run(mode,loops,[&](int m){if(m)graph->run();else{ops->gemm(runtime->stream());ops->cast(runtime->stream());}});
  drain();return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tick).count()/loops;
 }
 void close(){state.close([&]{assert_owner();infini::rt::DeviceGuard guard{runtime->device().type(),runtime->device().index(),true};if(graph_recording){runtime->cancelGraphRecording();graph_recording=false;}drain();graph.reset();ops.reset();workspace.reset();a.reset();b.reset();y.reset();z.reset();a_storage.reset();b_storage.reset();y_storage.reset();z_storage.reset();runtime->trimMemory();drain();runtime.reset();});}
 ~Context()noexcept{if(runtime)try{close();}catch(...){std::terminate();}}
};
}
extern "C" const char*deep_fixed_graph_error_r399(){return deep399::last_error.c_str();}
extern "C" void*deep_fixed_graph_create_r399(int explicit_optin,int device){
 using namespace deep399;try{
  if(explicit_optin!=1||device<0||device>=8)throw std::invalid_argument("r399 default OFF / device");
  infini::rt::DeviceGuard create_guard{rt::Device::Type::kHygon,device,true};auto c=std::make_unique<Context>();c->runtime=rt::Runtime::create({rt::Device::Type::kHygon,device});c->runtime->enableOwnerDrainFailStop();
  c->a_storage=rt::Tensor::empty(*c->runtime,{17,2048},rt::DataType::kFloat32);c->b_storage=rt::Tensor::empty(*c->runtime,{2049,512},rt::DataType::kFloat32);c->y_storage=rt::Tensor::empty(*c->runtime,{17,512},rt::DataType::kFloat32);c->z_storage=rt::Tensor::empty(*c->runtime,{17,512},rt::DataType::kBFloat16);
  c->a=c->a_storage->narrow({{0,0,16}});c->b=c->b_storage->narrow({{0,0,2048}});c->y=c->y_storage->narrow({{0,0,16}});c->z=c->z_storage->narrow({{0,0,16}});
  c->ops=make_prepared(true,ex::detail::view(c->a),ex::detail::view(c->b),ex::detail::view(c->y),ex::detail::view(c->z));c->workspace=c->runtime->allocateMemory(c->ops->workspace_bytes()+512);
  std::vector<unsigned char>g(256,0x5a);c->runtime->memcpyH2D(c->workspace->data(),g.data(),256,false);c->runtime->memcpyH2D(c->workspace->data()+c->workspace->size()-256,g.data(),256,false);
  for(auto store:{c->a_storage,c->b_storage,c->y_storage,c->z_storage}){size_t n=store->size(1)*rt::dsize(store->dtype());std::vector<unsigned char>tail(n,0x5a);c->runtime->memcpyH2D(store->data()+store->nbytes()-n,tail.data(),n,false);}return c.release();
 }catch(const std::exception&e){last_error=e.what();return nullptr;}}
extern "C" int deep_fixed_graph_inputs_r399(void*raw,const float*a,size_t an,const float*b,size_t bn){using namespace deep399;try{if(!raw)throw std::invalid_argument("r399 null context");static_cast<Context*>(raw)->inputs(a,an,b,bn);return 0;}catch(const std::exception&e){last_error=e.what();if(raw)static_cast<Context*>(raw)->on_error();return 1;}}
extern "C" int deep_fixed_graph_prepare_r399(void*raw){using namespace deep399;try{if(!raw)throw std::invalid_argument("r399 null context");static_cast<Context*>(raw)->prepare();return 0;}catch(const std::exception&e){last_error=e.what();if(raw)static_cast<Context*>(raw)->on_error();return 1;}}
extern "C" int deep_fixed_graph_capture_r399(void*raw){using namespace deep399;try{if(!raw)throw std::invalid_argument("r399 null context");static_cast<Context*>(raw)->capture();return 0;}catch(const std::exception&e){last_error=e.what();if(raw)static_cast<Context*>(raw)->on_error();return 1;}}
extern "C" int deep_fixed_graph_run_r399(void*raw,int mode,int loops,double*ms){using namespace deep399;try{if(!raw||!ms)throw std::invalid_argument("r399 null run");*ms=static_cast<Context*>(raw)->run(mode,loops);return 0;}catch(const std::exception&e){last_error=e.what();if(raw)static_cast<Context*>(raw)->on_error();return 1;}}
extern "C" int deep_fixed_graph_read_r399(void*raw,float*y,uint16_t*z,size_t count){using namespace deep399;try{if(!raw||!y||!z||count!=16*512)throw std::invalid_argument("r399 read buffers");auto&c=*static_cast<Context*>(raw);c.state.live();c.assert_owner();c.runtime->memcpyD2H(y,c.y->data(),count*4);c.runtime->memcpyD2H(z,c.z->data(),count*2);return 0;}catch(const std::exception&e){last_error=e.what();if(raw)static_cast<Context*>(raw)->on_error();return 1;}}
extern "C" const char*deep_fixed_graph_report_r399(void*raw){using namespace deep399;try{if(!raw)throw std::invalid_argument("r399 null report");auto&c=*static_cast<Context*>(raw);c.assert_owner();report="{\"prepares\":"+std::to_string(c.state.prepares())+",\"captures\":"+std::to_string(c.state.captures())+",\"eager_calls\":"+std::to_string(c.state.eager())+",\"replay_calls\":"+std::to_string(c.state.replay())+",\"workspace_bytes\":"+std::to_string(c.workspace->size())+",\"phase\":"+std::to_string(int(c.state.phase()))+"}";return report.c_str();}catch(const std::exception&e){last_error=e.what();return nullptr;}}
extern "C" int deep_fixed_graph_destroy_r399(void*raw){using namespace deep399;try{if(!raw)throw std::invalid_argument("r399 null destroy");auto*c=static_cast<Context*>(raw);c->close();delete c;return 0;}catch(const std::exception&e){last_error=e.what();return 1;}}

extern "C" int deep_fixed_graph_check_r399(void*raw){using namespace deep399;try{if(!raw)throw std::invalid_argument("r399 null check");static_cast<Context*>(raw)->check();return 0;}catch(const std::exception&e){last_error=e.what();if(raw)static_cast<Context*>(raw)->on_error();return 1;}}
