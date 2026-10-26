#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <mutex>
#include <thread>
namespace infinilm::models::deepseek_v4::csa_host_r9246 {
// Exclusive HOST call-scope durations, not GPU busy. A helper bucket may contain its existing waits.
enum Bucket: size_t { INDEXER,SELECTION_READ,SELECTION_VALIDATE,BANK_SUBMIT,DESCRIPTORS,EXPLICIT_DRAIN,PREPARE,COMPRESSOR,LATENT,INVERSE,OUTPUT,COMMIT,RESIDUAL,N };
struct Sample {uint64_t serial=0,tokens=0,requests=0,owner_thread=0,completed=0;std::array<double,N+1>wall{},cpu{};std::array<uint64_t,N>visits{};};
inline std::atomic<int> mode{0};inline std::array<Sample,8>samples{};inline std::mutex sample_mutex;
struct Stamp{double wall,cpu;};
inline Stamp now(){timespec t{};if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t))std::terminate();return {std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(),double(t.tv_sec)+double(t.tv_nsec)*1e-9};}
struct Frame {
 bool enabled=false;int device;size_t current=RESIDUAL;Stamp begin{},last{};Sample value{};
 Frame(int d,size_t t,size_t rows,bool eligible):enabled(mode.load(std::memory_order_relaxed)&&eligible),device(d){if(enabled){value.tokens=t;value.requests=rows;value.owner_thread=std::hash<std::thread::id>{}(std::this_thread::get_id());begin=last=now();++value.visits[current];}}
 void set(size_t b){if(!enabled)return;auto n=now();value.wall[current]+=n.wall-last.wall;value.cpu[current]+=n.cpu-last.cpu;last=n;current=b;++value.visits[b];}
 struct Scope{Frame&f;size_t previous;Scope(Frame&p,size_t b):f(p),previous(p.current){f.set(b);}~Scope(){f.set(previous);}};
 template<class F>decltype(auto)call(size_t b,F&&fn){Scope scope(*this,b);return fn();}
 void success(){if(enabled)value.completed=1;}
 ~Frame(){if(!enabled)return;auto n=now();value.wall[current]+=n.wall-last.wall;value.cpu[current]+=n.cpu-last.cpu;value.wall[N]=n.wall-begin.wall;value.cpu[N]=n.cpu-begin.cpu;std::lock_guard<std::mutex>lock(sample_mutex);value.serial=samples.at(device).serial+1;samples.at(device)=value;}
};
}
