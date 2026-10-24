import hashlib,json,pathlib
r=pathlib.Path('.');base=r/'GLM-PP3-B64-private-r8780-src/modern_glm_cached_model_bridge_r809.cpp'
s=base.read_bytes().decode();assert hashlib.sha256(base.read_bytes()).hexdigest()=='8fc424d1595fd30c59886dac33f847df3b4d08a40e8e7d348eaaff3d0a2be03e'
prefix=s[s.index(' const char*setting='):s.index(' infinilm::engine::distributed::RankInfo rank(r.device());')]
copyline=next(line for line in prefix.splitlines() if 'std::memcpy(&world,world_id' in line)
prefix=prefix.replace(copyline,copyline+'\n comm_metadata_r8783={{"IDfingerprint_fnv1a64",{{"world",comm_id_r8783(world_id)},{"TP",comm_id_r8783(tp_id)},{"control_input",comm_id_r8783(control_id)}}},{"control_shared_world",c->shared_control_r4279}};')
# The precise existing setup statements are reused; insert only owner-local logs.
for phase,statement in [('world','if(infinicclCommInitRank(&c->world'),('TP','if(infinicclCommInitRank(&c->tp')]:
    line=next(line for line in prefix.splitlines() if statement in line)
    prefix=prefix.replace(line,' comm_phase_r8783(journal,stage,device,"'+phase+'.enter");\n'+line+'\n comm_phase_r8783(journal,stage,device,"'+phase+'.return");')
line=next(line for line in prefix.splitlines() if 'if(c->shared_control_r4279)c->control=c->world;' in line)
prefix=prefix.replace(line,' comm_phase_r8783(journal,stage,device,"control.enter");\n'+line+'\n comm_phase_r8783(journal,stage,device,"control.return");')
extra=r'''
// Private comm-only diagnostic. Original model create and guard behavior above
// is byte-preserved. No module is constructed, weights loaded, or forward run.
#include <fstream>
#include <chrono>
#include <cstdio>
#include <unistd.h>
namespace {
thread_local nlohmann::json comm_metadata_r8783;
std::string comm_id_r8783(const void*p){const auto*b=static_cast<const unsigned char*>(p);uint64_t h=14695981039346656037ull;for(size_t i=0;i<sizeof(infinicclUniqueId);++i){h^=b[i];h*=1099511628211ull;}return std::to_string(h);}
void comm_phase_r8783(const char*path,int stage,int device,const char*phase){
 if(!path||!*path)throw std::invalid_argument("8783 journal path required");
 const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
 auto item=comm_metadata_r8783;item["phase"]=phase;item["stage"]=stage;item["rank"]=device;item["device"]=device;item["participant"]=stage*8+device;item["PID"]=int(getpid());item["native_steady_ns"]=ns;const auto row=item.dump();
 std::ofstream out(path,std::ios::app);if(!out)throw std::runtime_error("8783 journal open");out<<row<<'\n';out.flush();if(!out)throw std::runtime_error("8783 journal flush");
 std::printf("GLM_COMM_PHASE_R8783 %s\n",row.c_str());std::fflush(stdout);
}
}
extern "C" void* modern_glm_comm_only_create_r8783(int device,int stage,size_t rows,size_t capacity,const char*cfg,const void*world_id,const void*tp_id,const void*control_id,const char*journal){std::unique_ptr<Context>c;try{
'''+prefix+r'''
 // Same native stream/Runtime, before RankInfo/TextModel allocation boundary.
 r.syncStream();comm_phase_r8783(journal,stage,device,"comm_only.return.before_TextModel");return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());try{comm_phase_r8783(journal,stage,device,"comm_only.error");}catch(...){}return nullptr;}}
extern "C" int modern_glm_comm_only_destroy_r8783(void*raw,const char*journal){
 auto*c=static_cast<Context*>(raw);try{
  if(!c||!c->owner||c->module||c->ready||!c->loaded.empty())throw std::invalid_argument("8783 destroy requires unready nullmodel context");
  c->owner->assertOwnerThread();const int stage=c->stage,device=c->owner->device().index();
  comm_phase_r8783(journal,stage,device,"cleanup.enter");
  const int result=modern_glm_model_destroy(raw);
  if(result)return result;
  comm_phase_r8783(journal,stage,device,"cleanup.return");return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
'''
p=r/'modern_glm_comm_bridge_r8783.cpp';p.write_bytes((s+extra).encode());assert p.read_bytes().startswith(base.read_bytes())
q={'scope':'Onebridge hostTU; originalcreate preserved; comm-only beforeTextModel, exactworldTPcontrol sequence; noModelparams/weights/resident/forward',
   'base_SHA256':hashlib.sha256(base.read_bytes()).hexdigest(),'source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),
   'original_create_unchanged':True,'nullmodel_destroy':'Originaldestroy calls module.reset() only; no module dereference. Newcommdestroy validates nullmodule/unready and invokes originaldestroy, followed by originalshutdown in consumer.'}
(r/'GLM-comm-bridge-source-r8783.json').write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
