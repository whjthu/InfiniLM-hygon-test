#include "phase_tape_r161.hpp"
#include <vector>
#include <iostream>
#include <cassert>
struct Driver {using Event=int;int creates=0,records=0,copies=0,destroys=0;bool done=false;std::vector<int32_t>buffer;
 void check_owner(){}Event create_event(){return creates++;}void destroy_event(Event){destroys++;}int32_t*pinned_i32(size_t n){buffer.resize(n);return buffer.data();}void free_pinned(void*){}
 void record(Event){records++;}bool query(Event){return done;}float elapsed(Event a,Event b){return float(b-a);}void copy_ids_async(int32_t*dst,const int32_t*src,size_t n){copies++;for(size_t i=0;i<n;i++)dst[i]=src[i];}
};
template<class F>void rejects(F f){bool yes=false;try{f();}catch(const std::logic_error&){yes=true;}assert(yes);}
int main(){Driver d;moe_phase_r161::Tape<Driver>t(d);t.arm(1,1);assert(!t.begin(47,64));assert(d.records==0);assert(t.begin(1,64));rejects([&]{t.mark(2);});for(size_t i=1;i<=14;i++)t.mark(i);std::vector<int32_t>ids(1024);for(size_t i=0;i<ids.size();i++)ids[i]=int(i%896);bool leased=false;t.finish(ids.data(),[&]{leased=true;});rejects([&]{t.read_after_original_completion();});rejects([&]{t.close_after_original_completion();});d.done=true;auto q=t.read_after_original_completion();size_t sum=0;for(auto v:q.hist)sum+=v;assert(sum==1024&&q.tokens==64&&q.layer==1);for(auto v:q.ms)assert(v==1);assert(d.records==16&&d.copies==1);rejects([&]{t.arm(1,1);});t.arm(47,2);rejects([&]{t.begin(47,8);});t.close_after_original_completion();assert(d.destroys==16);std::cout<<"{\"passed\":true,\"CPU_only\":true,\"GPU\":false,\"Native\":false,\"cases\":8,\"no_wait_driver\":true}\n";}
