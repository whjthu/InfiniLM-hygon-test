#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
namespace moe_phase_r155 {
template<class Driver> class Tape {
 Driver&d;using Event=typename Driver::Event;std::array<Event,16>events{};
 int32_t*host=nullptr;bool created=false;int phase=0,layer=-1;size_t next=0,T=0;
 uint64_t epoch=0,last=0;std::function<void()>hold;
 void require(bool x,const char*m){if(!x)throw std::logic_error(m);}
public:
 struct Result {int layer;uint64_t epoch;size_t tokens;std::array<float,15>ms{};std::array<uint32_t,896>hist{};};
 explicit Tape(Driver&driver):d(driver){}
 void arm(int l,uint64_t e){d.check_owner();require(phase==0&&l>=1&&l<93&&e>last,"arm state/layer/epoch");if(!created){for(auto&v:events)v=d.create_event();host=d.pinned_i32(64*16);created=true;}layer=l;epoch=e;phase=1;}
 bool begin(int l,size_t tokens){d.check_owner();if(phase!=1||l!=layer)return false;require(tokens==64,"T64 only");T=tokens;next=1;phase=2;d.record(events[0]);return true;}
 bool active()const{return phase==2;}
 void mark(size_t index){d.check_owner();require(phase==2&&index==next&&next<=14,"disjoint phase order");d.record(events[next++]);}
 void finish(const int32_t*ids,std::function<void()>lease){d.check_owner();require(phase==2&&next==15&&ids,"finish phase/ids");hold=std::move(lease);d.copy_ids_async(host,ids,T*16);d.record(events[15]);phase=3;}
 Result read_after_original_completion(){d.check_owner();require(phase==3&&d.query(events[15]),"original drain not complete; never wait here");Result r{};r.layer=layer;r.epoch=epoch;r.tokens=T;for(size_t j=0;j<15;j++)r.ms[j]=d.elapsed(events[j],events[j+1]);for(size_t j=0;j<T*16;j++){require(host[j]>=0&&host[j]<896,"expert id range");r.hist[size_t(host[j])]++;}last=epoch;phase=0;hold={};return r;}
 void close_after_original_completion(){d.check_owner();require(phase==0||phase==1||(phase==3&&d.query(events[15])),"close pending work");hold={};if(created){d.free_pinned(host);for(auto e:events)d.destroy_event(e);}created=false;phase=0;}
};
}
