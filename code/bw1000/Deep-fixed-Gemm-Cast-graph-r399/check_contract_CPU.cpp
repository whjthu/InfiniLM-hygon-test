#include "contract.hpp"
#include <iostream>
#include <memory>
#include <functional>
#include <vector>
using namespace deep399;
int main(){int checks=0;auto ok=[&](bool b){if(!b)throw std::runtime_error("CPU recording assertion");++checks;};auto refuses=[&](auto f){try{f();}catch(const std::exception&){++checks;return;}throw std::runtime_error("expected refusal");};
 Contract q;int prepare=0,capture=0,start=0;
 refuses([&]{q.capture([&]{++capture;});});ok(capture==0);
 refuses([&]{q.run(1,1,[&](int){++start;});});ok(start==0);
 q.prepare([&]{++prepare;});ok(prepare==1&&q.prepares()==1);
 refuses([&]{q.prepare([&]{++prepare;});});ok(prepare==1);
 auto lease=std::make_shared<int>(7);std::weak_ptr<int>w=lease;std::function<void()>recorded;
 q.capture([&]{++capture;recorded=[lease]{};});lease.reset();ok(!w.expired()&&capture==1&&q.captures()==1);
 q.run(0,3,[&](int m){ok(m==0);++start;});q.run(1,2,[&](int m){ok(m==1);recorded();++start;});ok(start==5&&q.eager()==3&&q.replay()==2);
 refuses([&]{q.run(2,1,[&](int){++start;});});ok(start==5);
 refuses([&]{q.run(1,65,[&](int){++start;});});ok(start==5);
 q.close([&]{recorded={};});ok(w.expired()&&q.phase()==Phase::Closed);refuses([&]{q.live();});
 for(int failure=0;failure<3;++failure){Contract r;if(failure==0)refuses([&]{r.prepare([]{throw std::runtime_error("prepare leaf");});});else{r.prepare([]{});if(failure==1)refuses([&]{r.capture([]{throw std::runtime_error("capture leaf");});});else{r.capture([]{});refuses([&]{r.run(1,1,[](int){throw std::runtime_error("replay leaf");});});}}ok(r.phase()==Phase::Failed);refuses([&]{r.live();});int closed=0;r.close([&]{++closed;});ok(closed==1&&r.phase()==Phase::Closed);}
 for(int failure=-1;failure<5;++failure){int begin=0,adds=0,stop=0,validate=0,cancel=0;std::vector<std::shared_ptr<int>>retained;auto source=std::make_shared<int>(9);std::weak_ptr<int>weak=source;
  auto transaction=[&]{record_two([&]{++begin;if(failure==0)throw std::runtime_error("begin");},[&]{if(failure==1)throw std::runtime_error("retention");return source;},[&](auto ptr,int index){++adds;if(failure==index+2)throw std::runtime_error("add");retained.push_back(ptr);},[&]{++stop;if(failure==4)throw std::runtime_error("stop");},[&]{++validate;},[&]{++cancel;retained.clear();});};
  if(failure<0){transaction();source.reset();ok(!weak.expired()&&begin==1&&adds==2&&stop==1&&validate==1&&cancel==0);retained.clear();ok(weak.expired());}
  else{refuses(transaction);ok(begin==1&&validate==0&&cancel==(failure==0?0:1));}
 }
 {Contract r;bool refused=false,cleaned=false;std::thread foreign([&]{try{r.live();}catch(...){refused=true;}r.owner_cleanup([&]{cleaned=true;r.fail();});});foreign.join();ok(refused&&!cleaned&&r.phase()==Phase::Fresh);r.close([]{});ok(r.phase()==Phase::Closed);}
 std::cout<<"{\"completed\":true,\"passed\":true,\"checks\":"<<checks<<",\"actual_shared_state_machine\":true,\"Runtime_and_vendor_leaf\":\"recording only, no device\",\"GPU\":false}"<<std::endl;
}
