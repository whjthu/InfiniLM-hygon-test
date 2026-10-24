import hashlib,json,pathlib
r=pathlib.Path('.');base=r/'modern_glm_comm_bridge_r8783.cpp';s=base.read_text();assert hashlib.sha256(base.read_bytes()).hexdigest()=='dab58a23e72cd15b3d498f212c6ae43d9930ea9b5d1f36ae21382d6474b4077c'
comm=s.index('extern "C" void* modern_glm_comm_only_create_r8783(')
start=s.index('std::unique_ptr<Context>c;try{',comm)+len('std::unique_ptr<Context>c;try{')
end=s.index(' // Same native stream/Runtime',start);prefix=s[start:end]
original=s.index('extern "C" void*modern_glm_model_create(')
start=s.index(' infinilm::engine::distributed::RankInfo rank(r.device());',original);end=s.index('\n }catch(',start);tail=s[start:end]
line=next(v for v in tail.splitlines() if 'c->module=std::make_unique<model::TextModelProjectedR22>' in v)
assert line.endswith('r.syncStream();return c.release();')
make=line[:-len('r.syncStream();return c.release();')]
tail=tail.replace(line,' comm_phase_r8783(journal,stage,device,"TextModel.enter");\n'+make+'\n comm_phase_r8783(journal,stage,device,"TextModel.return");\n comm_phase_r8783(journal,stage,device,"sync.enter");r.syncStream();comm_phase_r8783(journal,stage,device,"sync.return");\n comm_phase_r8783(journal,stage,device,"constructor_only.return.unloaded");return c.release();')
extra='''
// Conditional53 unloaded constructor: same original order, no new barrier,
// stream, ready bypass, checkpoint I/O, weight processing or forward.
extern "C" void* modern_glm_constructor_only_create_r8790(int device,int stage,size_t rows,size_t capacity,const char*cfg,const void*world_id,const void*tp_id,const void*control_id,const char*journal){std::unique_ptr<Context>c;try{
'''+prefix+tail+'''
 }catch(const std::exception&e){error=e.what();drain(c.get());try{comm_phase_r8783(journal,stage,device,"constructor_only.error");}catch(...){}return nullptr;}}
extern "C" int modern_glm_constructor_only_destroy_r8790(void*raw,const char*journal){
 auto*c=static_cast<Context*>(raw);try{
  if(!c||!c->owner||!c->module||c->ready||!c->loaded.empty())throw std::invalid_argument("8790 destroy requires unloaded constructor context");
  c->owner->assertOwnerThread();const int stage=c->stage,device=c->owner->device().index();
  comm_phase_r8783(journal,stage,device,"cleanup.enter");const int result=modern_glm_model_destroy(raw);if(result)return result;
  comm_phase_r8783(journal,stage,device,"cleanup.return");return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
'''
p=r/'modern_glm_constructor_bridge_r8790.cpp';p.write_bytes((s+extra).encode());assert p.read_bytes().startswith(base.read_bytes())
j={'scope':'53conditional same260s unloadedconstructor only; samecommorder thenoriginalRankInfo/rope/TextModel/sync withimmediatephases; noextra barrier, weights/finish/resident/forward',
 'base_SHA256':hashlib.sha256(base.read_bytes()).hexdigest(),'source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_create_and_comm_only_unchanged':True,'condition_met':'531all24normal/audit0/532postempty'}
(r/'GLM-constructor-bridge-source-r8790.json').write_bytes((json.dumps(j,indent=2)+'\n').encode());print(json.dumps(j))
