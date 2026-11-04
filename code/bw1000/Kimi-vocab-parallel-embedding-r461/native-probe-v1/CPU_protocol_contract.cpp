#include "probe_protocol.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
using namespace kimi_embedding_probe_r461;
std::array<std::shared_ptr<Request>,8> requests(int*workers,const void*context,std::uint64_t epoch) {
    auto gate=std::make_shared<Gate>();std::array<std::shared_ptr<Request>,8> out;
    for(std::size_t d=0;d<8;++d) {
        auto q=std::make_shared<Request>();q->device=d;q->worker=&workers[d];q->context=context;
        q->epoch=epoch;q->ids={0,5119,5120,163839};q->gate=gate;out[d]=q;
    }
    return out;
}
int main() {
    int workers[8]{},context=0,outsider=0;std::size_t cases=0;
    {
        auto q=requests(workers,&context,1);install(q);
        assert(!contains(&outsider)&&!acquire(&outsider,0));
        std::array<std::thread,8> owners;std::atomic<int>done{0};
        for(std::size_t d=0;d<8;++d)owners[d]=std::thread([&,d]{
            auto task=acquire(&workers[d],d);assert(task==q[d]&&task->epoch==1);
            metadata_ready(task);await_admission(task);task->output={std::uint16_t(d)};task->completed=true;++done;
        });
        assert(await_all_metadata(q[0]->gate));release(q[0]->gate,false);
        for(auto&owner:owners){owner.join();}
        assert(done==8);
        std::weak_ptr<Request> retained=q[0];clear(q);assert(!retained.expired());
        q={};assert(retained.expired());++cases;
    }
    {
        auto q=requests(workers,&context,2);install(q);std::array<std::thread,8> owners;std::atomic<int>cancelled{0};
        for(std::size_t d=0;d<8;++d)owners[d]=std::thread([&,d]{
            auto task=acquire(&workers[d],d);
            if(d==3){release(task->gate,true);return;}
            metadata_ready(task);try{await_admission(task);}catch(const std::runtime_error&){++cancelled;}
        });
        assert(!await_all_metadata(q[0]->gate));release(q[0]->gate,true);
        for(auto&owner:owners){owner.join();}
        assert(cancelled==7);clear(q);++cases;
    }
    {
        auto q=requests(workers,&context,3);install(q);
        for(std::size_t d=0;d<3;++d){auto task=acquire(&workers[d],d);metadata_ready(task);}
        release(q[0]->gate,true);
        for(std::size_t d=0;d<3;++d){bool rejected=false;try{await_admission(q[d]);}catch(const std::runtime_error&){rejected=true;}assert(rejected);}
        clear(q);++cases;
    }
    for(int bad=0;bad<3;++bad) {
        auto q=requests(workers,&context,4);
        if(bad==0)q[4]->epoch=3;
        if(bad==1)q[4]->context=&outsider;
        if(bad==2)q[4]->worker=&workers[3];
        bool rejected=false;try{install(q);}catch(const std::invalid_argument&){rejected=true;}
        assert(rejected&&!contains(&workers[0]));++cases;
    }
    assert(cases==6);
    std::cout<<"{\"passed\":true,\"cases\":6,\"actual_std_threads\":8,\"Runtime_or_Tensor_executed\":false,\"SDK_execution\":false,\"GPU\":0}\n";
}
