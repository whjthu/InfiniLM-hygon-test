#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include "execution/submit.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <typeinfo>
#include <cuda_runtime_api.h>
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;using J=nlohmann::json;
double wall(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
double cpu(){timespec t{};if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t))throw std::runtime_error("thread clock");return t.tv_sec+t.tv_nsec*1e-9;}
void save(const std::string&p,const J&q){std::ofstream f(p);if(!f)throw std::runtime_error("output");f<<q.dump(2)<<"\n";f.flush();if(!f)throw std::runtime_error("write");}
struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;J rows=J::array();
 rt::Tensor alloc(rt::Shape s){auto t=rt::Tensor::empty(r,s,rt::DataType::kFloat32);held.push_back(t);return t;}
 template<class Op>rt::Tensor binary(const rt::Tensor&a,const rt::Tensor&b,rt::Shape s,const char*name){double t=wall(),c=cpu();auto out=alloc(s);double aend=wall(),ac=cpu();ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,b,out);double end=wall(),ec=cpu();rows.push_back({{"op",name},{"allocation_wall_s",aend-t},{"allocation_thread_s",ac-c},{"submit_wall_s",end-aend},{"submit_thread_s",ec-ac}});return out;}
 template<class Op>rt::Tensor unary(const rt::Tensor&a){double t=wall(),c=cpu();auto out=alloc(a->shape());double aend=wall(),ac=cpu();ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,out);double end=wall(),ec=cpu();rows.push_back({{"op","Sigmoid"},{"allocation_wall_s",aend-t},{"allocation_thread_s",ac-c},{"submit_wall_s",end-aend},{"submit_thread_s",ec-ac}});return out;}
 ~Work(){try{r.syncStream();}catch(...){std::terminate();}}
};
template<class Op>J binary_metadata(rt::Runtime&r,rt::Tensor a,rt::Tensor b,rt::Tensor out){auto cfg=Op::DefaultConfig(r.device().type());double t=wall();auto op=Op::Make(cfg,a->view(),b->view(),out->view());return J{{"implementation_index",cfg.implementation_index()},{"provider_RTTI",typeid(*op).name()},{"workspace_bytes",op->workspace_size_in_bytes()},{"Make_wall_s",wall()-t}};}
int main(int argc,char**argv){if(argc!=3)return 2;int d=std::stoi(argv[1]);std::string path=argv[2];J q={{"completed",false},{"passed",false},{"rank",d},{"weights",false},{"Model",false},{"HTTP",0},{"frames",J::array()}};double begin=wall();std::weak_ptr<rt::Runtime>weak;
 try{if(d<0||d>=8)throw std::runtime_error("rank");if(cudaSetDevice(d)!=cudaSuccess)throw std::runtime_error("device");auto memory=[&](){size_t f=0,t=0;if(cudaMemGetInfo(&f,&t)!=cudaSuccess||f<16ULL*1024*1024*1024)throw std::runtime_error("16GiB floor");return J{{"free",f},{"total",t}};};q["memory_before_create"]=memory();
 {auto r=rt::Runtime::create(rt::Device{rt::Device::Type::kHygon,d});weak=r;r->enableOwnerDrainFailStop();r->assertOwnerThread();q["Runtime_create_s"]=wall()-begin;q["memory_after_create"]=memory();
 std::vector<float>mix(24),base(24),scale={.5f,.75f,.25f},constants={1e-6f,1e-6f,2.f};for(int i=0;i<24;++i){mix[i]=(i-12)*.03125f;base[i]=(i%7-3)*.015625f;}
 auto tensor=[&](rt::Shape s,const std::vector<float>&v){auto t=rt::Tensor::empty(*r,s,rt::DataType::kFloat32);r->memcpyH2D(t->data(),v.data(),v.size()*4,false);return t;};auto mixes=tensor({1,24},mix),b=tensor({24},base),sc=tensor({3},scale),cn=tensor({3},constants);
 {Work w{*r};auto a=mixes->narrow({{1,0,4}}),ss=sc->narrow({{0,0,1}}),out=w.alloc({1,4});q["metadata"]["Mul"]=binary_metadata<infini::ops::Mul>(*r,a,ss,out);q["metadata"]["Add"]=binary_metadata<infini::ops::Add>(*r,a,b->narrow({{0,0,4}}),out);auto cfg=infini::ops::Sigmoid::DefaultConfig(r->device().type());auto op=infini::ops::Sigmoid::Make(cfg,out->view(),out->view());q["metadata"]["Sigmoid"]={{"implementation_index",cfg.implementation_index()},{"provider_RTTI",typeid(*op).name()},{"workspace_bytes",op->workspace_size_in_bytes()}};q["Mul_Torch8_hypothesis"]=q["metadata"]["Mul"]["implementation_index"]==8;save(path,q);}
 for(int frame=0;frame<7;++frame){J row={{"frame",frame},{"cold",frame==0},{"warm",frame==1||frame==2}};
 {Work w{*r};double start=wall(),startcpu=cpu(),viewwall=0,viewcpu=0;
 auto narrow=[&](const rt::Tensor&t,size_t dim,size_t offset,size_t width){double a=wall(),c=cpu();auto v=t->narrow({{dim,offset,width}});viewwall+=wall()-a;viewcpu+=cpu()-c;return v;};
 auto affine=[&](size_t off,size_t width,size_t si){auto a=narrow(mixes,1,off,width),s=narrow(sc,0,si,1);auto product=w.binary<infini::ops::Mul>(a,s,{1,width},"Mul");return w.binary<infini::ops::Add>(product,narrow(b,0,off,width),{1,width},"Add");};
 auto pre_sig=w.unary<infini::ops::Sigmoid>(affine(0,4,0));auto pre=w.binary<infini::ops::Add>(pre_sig,narrow(cn,0,1,1),{1,4},"Add");auto post_sig=w.unary<infini::ops::Sigmoid>(affine(4,4,1));auto post=w.binary<infini::ops::Mul>(post_sig,narrow(cn,0,2,1),{1,4},"Mul");auto logits=affine(8,16,2);double vt=wall(),vc=cpu();auto logits_view=logits->view({1,4,4});viewwall+=wall()-vt;viewcpu+=cpu()-vc;double at=wall(),ac=cpu();auto prob=w.alloc({1,4,4});double ae=wall(),ace=cpu();ex::submit<infini::ops::Softmax>(*r,infini::ops::Softmax::DefaultConfig(r->device().type()),logits_view,int64_t{2},std::optional<infini::ops::DataType>{},prob);double se=wall(),sce=cpu();w.rows.push_back({{"op","Softmax"},{"allocation_wall_s",ae-at},{"allocation_thread_s",ace-ac},{"submit_wall_s",se-ae},{"submit_thread_s",sce-ace}});auto comb=w.binary<infini::ops::Add>(prob,narrow(cn,0,1,1),{1,4,4},"Add");double drain=wall();r->syncStream();double end=wall();row["whole_to_original_completion_s"]=end-start;row["thread_to_completion_s"]=cpu()-startcpu;row["final_drain_s"]=end-drain;row["narrow_view_wall_s"]=viewwall;row["narrow_view_thread_s"]=viewcpu;row["ops"]=w.rows;
 std::vector<float>pv(4),postv(4),cv(16);r->memcpyD2H(pv.data(),pre->data(),16);r->memcpyD2H(postv.data(),post->data(),16);r->memcpyD2H(cv.data(),comb->data(),64);double err=0;for(int i=0;i<4;++i){double ref=1/(1+std::exp(-(double(mix[i])*scale[0]+base[i])))+constants[1];err=std::max(err,std::abs(pv[i]-ref));ref=2/(1+std::exp(-(double(mix[i+4])*scale[1]+base[i+4])));err=std::max(err,std::abs(postv[i]-ref));}
 for(int i=0;i<4;++i){double den=0;for(int j=0;j<4;++j)den+=std::exp(double(mix[8+4*i+j])*scale[2]+base[8+4*i+j]);for(int j=0;j<4;++j){double ref=std::exp(double(mix[8+4*i+j])*scale[2]+base[8+4*i+j])/den+constants[1];err=std::max(err,std::abs(cv[i*4+j]-ref));}}
 for(float x:pv)if(!std::isfinite(x))throw std::runtime_error("nonfinite");for(float x:postv)if(!std::isfinite(x))throw std::runtime_error("nonfinite");for(float x:cv)if(!std::isfinite(x))throw std::runtime_error("nonfinite");if(err>2e-6||w.rows.size()!=12||w.held.size()!=12)throw std::runtime_error("CPUref/opcount");row["max_abs_CPUref"]=err;row["finite_CPUref_passed"]=true;row["output"]={{"pre",pv},{"post",postv},{"comb",cv}};}
 q["frames"].push_back(row);save(path,q);}
 r->syncStream();q["memory_after_frames"]=memory();}
 q["normal_Runtime_expired"]=weak.expired();if(!weak.expired())throw std::runtime_error("Runtime retained");q["passed"]=true;
 }catch(const std::exception&e){q["error"]=e.what();}q["completed"]=true;q["seconds"]=wall()-begin;save(path,q);return q["passed"].get<bool>()?0:1;}
