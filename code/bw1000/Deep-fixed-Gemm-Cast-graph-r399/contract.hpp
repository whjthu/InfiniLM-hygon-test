#pragma once
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
namespace deep399 {
// The production bridge and CPU recording leaf execute this same transaction.
// Begin/retention/stop are actual Runtime APIs in production, recording callbacks in CPU.
template<class Begin,class Retain,class Add,class Stop,class Validate,class Cancel>
void record_two(Begin begin,Retain retain,Add add,Stop stop,Validate validate,Cancel cancel){
 bool started=false;
 try{begin();started=true;auto resources=retain();add(resources,0);add(resources,1);stop();started=false;validate();}
 catch(...){if(started)cancel();throw;}
}
enum class Phase { Fresh, Prepared, Captured, Failed, Closed };
// This exact state machine is used by the real Runtime bridge and CPU recording leaf.
class Contract {
 std::thread::id owner_=std::this_thread::get_id();
 Phase phase_=Phase::Fresh;
 uint64_t prepared_=0, captures_=0, eager_=0, replay_=0;
 public:
 void owner()const{if(owner_!=std::this_thread::get_id())throw std::invalid_argument("r399 owner thread");}
 template<class F>bool owner_cleanup(F&&f){try{owner();}catch(...){return false;}f();return true;}
 Phase phase()const{return phase_;}
 void live()const{owner();if(phase_==Phase::Failed||phase_==Phase::Closed)throw std::logic_error("r399 retired");}
 template<class F>void prepare(F&&f){owner();if(phase_!=Phase::Fresh)throw std::logic_error("r399 prepare once");try{f();phase_=Phase::Prepared;++prepared_;}catch(...){phase_=Phase::Failed;throw;}}
 template<class F>void capture(F&&f){owner();if(phase_!=Phase::Prepared)throw std::logic_error("r399 capture requires prepared");try{f();phase_=Phase::Captured;++captures_;}catch(...){phase_=Phase::Failed;throw;}}
 template<class F>void run(int mode,int loops,F&&f){owner();if(phase_!=Phase::Captured||mode<0||mode>1||loops<1||loops>64)throw std::invalid_argument("r399 run state/mode/loops");try{for(int i=0;i<loops;++i){f(mode);if(mode)++replay_;else ++eager_;}}catch(...){phase_=Phase::Failed;throw;}}
 void fail()noexcept{phase_=Phase::Failed;}
 template<class F>void close(F&&f){owner();if(phase_==Phase::Closed)throw std::logic_error("r399 already closed");try{f();phase_=Phase::Closed;}catch(...){phase_=Phase::Failed;throw;}}
 uint64_t prepares()const{return prepared_;}uint64_t captures()const{return captures_;}uint64_t eager()const{return eager_;}uint64_t replay()const{return replay_;}
};
}
