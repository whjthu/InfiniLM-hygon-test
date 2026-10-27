#pragma once
#include <stdexcept>
#include <exception>
namespace infinilm::models::deepseek_v4::prefix_r135 {
// Runtime owns a device; its API guards need not leave that device current.
// Bind only around raw driver queries and restore the owner's previous state.
template<class Api> class ScopedDevice {
 int previous_=-1,device_=-1; bool armed_=false;
public:
 explicit ScopedDevice(int device):device_(device){
  if(device<0||Api::get(&previous_)!=0)throw std::runtime_error("prefix device previous query");
  if(previous_!=device_){
   if(Api::set(device_)!=0)throw std::runtime_error("prefix device bind");
   armed_=true;
  }
  int bound=-1;
  if(Api::get(&bound)!=0||bound!=device_){
   if(armed_&&Api::set(previous_)!=0)std::terminate();armed_=false;
   throw std::runtime_error("prefix bound current device differs");
  }
 }
 ~ScopedDevice()noexcept{if(armed_&&Api::set(previous_)!=0)std::terminate();}
 ScopedDevice(const ScopedDevice&)=delete;ScopedDevice&operator=(const ScopedDevice&)=delete;
 int previous()const noexcept{return previous_;}
 int device()const noexcept{return device_;}
};
}
