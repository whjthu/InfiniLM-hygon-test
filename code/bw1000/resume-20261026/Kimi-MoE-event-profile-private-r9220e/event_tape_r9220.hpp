#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <stdexcept>
namespace moe_event_r9220 {
// Driver operations never wait. Caller invokes read only after ORIGINAL Work completion.
template<class Driver>class Tape {
 Driver&d;using E=typename Driver::Event;std::array<E,10>events{};int32_t*host=nullptr;
 int phase=0,layer=-1;uint64_t epoch=0,last_epoch=0;size_t T=0,next=0;bool created=false;std::function<void()>hold;
 void owner(){d.check_owner();}void require(bool x,const char*m){if(!x)throw std::logic_error(m);}
public:
 struct Result{uint64_t epoch;int layer;size_t tokens,routes;std::array<float,8>segment_ms;float hist_copy_ms;std::array<uint32_t,896>hist{};};
 explicit Tape(Driver&driver):d(driver){}
 void arm(int l,uint64_t e){owner();require(phase==0&&l>=0&&l<93&&e>last_epoch,"arm owner/layer/epoch/pending");if(!created){for(auto&v:events)v=d.create_event();host=d.pinned_i32(256*16);created=true;}layer=l;epoch=e;phase=1;}
 bool begin(int l,size_t tokens){owner();if(phase!=1||l!=layer)return false;require(tokens>0&&tokens<=256,"profile tokens1..256");T=tokens;next=0;phase=2;d.record(events[next++]);return true;}
 void mark(){owner();require(phase==2&&next>=1&&next<=8,"mark order");d.record(events[next++]);}
 void finish(const int32_t*ids,std::function<void()>lease){owner();require(phase==2&&next==9&&ids,"finish order/ids");hold=std::move(lease);d.copy_ids_async(host,ids,T*16);d.record(events[9]);phase=3;}
 Result read_after_original_work_completion(){owner();require(phase==3&&d.query(events[9]),"original Work not completed; no blocking read");Result r{};r.epoch=epoch;r.layer=layer;r.tokens=T;r.routes=T*16;for(size_t i=0;i<8;i++)r.segment_ms[i]=d.elapsed(events[i],events[i+1]);r.hist_copy_ms=d.elapsed(events[8],events[9]);for(size_t i=0;i<T*16;i++){require(host[i]>=0&&host[i]<896,"invalid expert id");++r.hist[size_t(host[i])];}last_epoch=epoch;phase=0;hold={};return r;}
 void close_after_original_work_completion(){owner();require(phase==0||phase==1||(phase==3&&d.query(events[9])),"close requires original completion");hold={};if(created){d.free_pinned(host);for(auto e:events)d.destroy_event(e);}host=nullptr;created=false;phase=0;}
};}
