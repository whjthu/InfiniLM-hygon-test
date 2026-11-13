#include <cstddef>
#include <exception>
#include <stdexcept>
#include <vector>

namespace rt {
enum class DataType { Mock };
struct Shape {};
struct Runtime {
 int syncs=0;
 bool leased=false;
 bool throw_on_first_fence=false;
};
struct Tensor {
 static Tensor empty(Runtime&, Shape, DataType) { return {}; }
};
}
namespace decode_leases_r1354 {
bool active(rt::Runtime& r) { return r.leased; }
void success_fence(rt::Runtime& r) {
 if (r.leased) return;
 ++r.syncs;
 if(r.throw_on_first_fence) throw std::runtime_error("sync failed");
}
void retire(rt::Runtime& r,std::vector<rt::Tensor>& held,int entered) noexcept {
 if(!r.leased||std::uncaught_exceptions()>entered) ++r.syncs;
 else held.clear();
}
}
struct Timing {
 bool enabled() const noexcept { return true; }
 void before_fence(bool,bool) noexcept {}
 void after_fence() noexcept {}
};
namespace {struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;int entered_exceptions=std::uncaught_exceptions();bool drained=false;
~Work(){
 if(!drained||std::uncaught_exceptions()>entered_exceptions)
  decode_leases_r1354::retire(r,held,entered_exceptions);
}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}};}

void projection(rt::Runtime& r, bool fail_after_fence) {
 Timing timing;
 Work work{r, {}};
 bool cached=false;
 const bool leased=decode_leases_r1354::active(r);
 if(timing.enabled())timing.before_fence(cached,leased);
 decode_leases_r1354::success_fence(r);
 // Outside an active lease, success_fence has drained this Runtime's stream.
 // No work is enqueued between the drain and retirement of these tensors.
 work.drained=!leased;
 timing.after_fence();
 if(fail_after_fence) throw std::runtime_error("unwind");
}

int main() {
 rt::Runtime normal;
 projection(normal,false);
 if(normal.syncs!=1) return 1;
 rt::Runtime leased;
 leased.leased=true;
 projection(leased,false);
 if(leased.syncs!=0) return 2;
 rt::Runtime failed;
 failed.throw_on_first_fence=true;
 try {projection(failed,false); return 3;} catch(const std::runtime_error&) {}
 if(failed.syncs!=2) return 4;
 rt::Runtime unwind;
 try {projection(unwind,true); return 5;} catch(const std::runtime_error&) {}
 if(unwind.syncs!=2) return 6;
 rt::Runtime other;
 {Work sibling{other,{}};}
 if(other.syncs!=1) return 7;
 return 0;
}
