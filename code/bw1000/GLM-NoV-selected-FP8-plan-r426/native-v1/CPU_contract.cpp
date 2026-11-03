// Actual private cache implementation is linked into this executable.
// Only the unchanged external776 CABI is recorded. No SDK/device calls.
#include "glm_fp8_cache_r426.hpp"
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace {
struct RecordedPlan {int device;void*stream;};
std::atomic<unsigned> created{0},destroyed{0},runs{0},drains{0};
bool fail_run=false,fail_destroy=false;std::vector<char>weights_seen;
void require(bool v,const char*where){if(!v)throw std::logic_error(where);}
std::array<uint64_t,10>stats(void*c){std::array<uint64_t,10>v{};require(!glm_fp8_cache_stats_r426(c,v.data(),v.size()),"stats");return v;}
int run(void*c,const void*id,void*stream,size_t ro,char*w){static char out,x,scale;return glm_fp8_cache_run_r426(c,id,stream,1,1,2048,3584,512,224,4,ro,0,&out,&x,&scale,w,&scale);}
}
extern "C" const char*glm_fp8_error_r776(){return "recording776forcedfailure";}
extern "C" void*glm_fp8_create_r776(int device,int dtype,size_t m,size_t n,size_t k,size_t sr,size_t sc,size_t ro,size_t co,void*stream){
 require(glm_selected_plan_r426::fits(dtype,m,n,k,sr,sc,ro,co),"original776callgeometry");++created;return new RecordedPlan{device,stream};
}
extern "C" int glm_fp8_run_r776(void*p,int mode,void*out,const void*x,const void*xs,const void*w,const void*ws){
 require(p&&mode==1&&out&&x&&xs&&w&&ws,"runmetadata");++runs;weights_seen.push_back(*static_cast<const char*>(w));return fail_run?1:0;
}
extern "C" int glm_fp8_destroy_r776(void*p){require(p,"destroyplan");++drains;if(fail_destroy)return 1;delete static_cast<RecordedPlan*>(p);++destroyed;return 0;}
int main(int argc,char**argv){try{
 int runtime_identity=1,other_identity=2,stream_marker=3,other_stream=4;char weight='A';
 void*c=glm_fp8_cache_create_r426(0,&stream_marker,&runtime_identity);require(c,"createcache");
 if(argc==2&&std::string(argv[1])=="--destroy-failure"){
  require(!run(c,&runtime_identity,&stream_marker,0,&weight),"preparefatal");fail_destroy=true;glm_fp8_cache_clear_r426(c,&runtime_identity,&stream_marker);return 99;
 }
 // Same geometry, distinct layer/weight pointers: one originalplan, realcachehits.
 std::array<char,40>layer_weights{};
 for(size_t layer=0;layer<layer_weights.size();++layer){layer_weights[layer]=char(layer+1);require(!run(c,&runtime_identity,&stream_marker,0,&layer_weights[layer]),"crosslayerrun");}
 auto s=stats(c);require(s[0]==1&&s[2]==39&&s[3]==1&&s[5]==1&&s[6]==20447232&&s[8]==40,"shapeonlyreuse");
 require(weights_seen.size()==40,"recordeveryweight");for(size_t i=0;i<40;++i)require(weights_seen[i]==char(i+1),"freshweightpayload");
 unsigned foreign=0;std::thread t([&]{std::array<uint64_t,10>tmp{};foreign=glm_fp8_cache_stats_r426(c,tmp.data(),tmp.size());});t.join();require(foreign==1,"wrongthreadreject");
 require(run(c,&other_identity,&stream_marker,0,&weight)==1,"wrongidentityreject");require(run(c,&runtime_identity,&other_stream,0,&weight)==1,"wrongstreamreject");
 static char out,x,scale;
 for(size_t m:{size_t{1},size_t{512},size_t{8192}})require(glm_fp8_cache_run_r426(c,&runtime_identity,&stream_marker,1,1,m,3584,512,224,4,0,0,&out,&x,&scale,&weight,&scale)==1,"ineligibleMreject");
 require(stats(c)[0]==1,"invaliddidnotcreate");
 // All eight legal rowOffsets are separategeometrykeys, not layeridentities.
 for(size_t rank=1;rank<8;++rank)require(!run(c,&runtime_identity,&stream_marker,3584*rank,&weight),"legalrowoffset");
 s=stats(c);require(s[5]==8&&s[6]==8*20447232ull&&s[6]<=192ull*1024*1024&&s[4]==0,"192MiBbound");
 require(!glm_fp8_cache_clear_r426(c,&runtime_identity,&stream_marker),"normalclear");s=stats(c);require(s[0]==s[1]&&!s[5]&&!s[6],"fullyreleased");
 require(!run(c,&runtime_identity,&stream_marker,0,&weight),"newsequencecold");
 fail_run=true;unsigned before=drains;require(run(c,&runtime_identity,&stream_marker,0,&weight)==1,"runfailreturned");require(drains==before+1,"runfaildrainedbeforecallerreturns");fail_run=false;
 unsigned before_runs=runs;require(run(c,&runtime_identity,&stream_marker,0,&weight)==1&&runs==before_runs,"poisonsticky");
 require(!glm_fp8_cache_clear_r426(c,&runtime_identity,&stream_marker),"poisonclearreleases");require(run(c,&runtime_identity,&stream_marker,0,&weight)==1,"clearcannotrevivefailedowner");
 require(!glm_fp8_cache_destroy_r426(c,&runtime_identity,&stream_marker),"normalcachedestroy");require(created==destroyed,"noPlanleak");
 std::cout<<"{\"passed\":true,\"actual_private_cache_logic\":true,\"recording_776_leaf\":true,\"cross_layer_invocations\":40,\"first_sequence_created\":1,\"first_sequence_hits\":39,\"logical_plan_scratch_bytes\":20447232,\"normal_clear_and_error_drain\":true,\"sticky_failstop\":true,\"Native\":0,\"GPU\":0,\"SDK_executed\":false}"<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
