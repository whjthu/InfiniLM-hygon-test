from pathlib import Path
import hashlib,json
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
FLAG='infinilm_kimi_k192_v128_private'
def prepare(allocate_source,mla_source,pair_source,out):
 out=Path(out);out.mkdir(exist_ok=False);proof={}
 expected={'allocate':'9646d436b0795e976ca2e1db87c61a540c25a5c877cbe901841f2b7bc51bfe9e','mla':'788c905159b7cba1f5db194e4f6817e452ce5fc8887f8e0516827de7bf2f4b80','pair':'17c92773cfe9af85676aac104bb1786c7872d916f5a276629d7eef9f007523b3'}
 def change(s,old,new):
  assert s.count(old)==1,old[:120]
  return s.replace(old,new)
 for tag,path in [('allocate',allocate_source),('mla',mla_source),('pair',pair_source)]:
  assert H(path)==expected[tag],str(path);s=Path(path).read_text()
  if tag=='allocate':
   s='#include "cache.hpp"\n'+s
   old='                kv[i] = ar_storage_r456'
   new="""                const bool split=model_config->get_config_json().value("infinilm_kimi_k192_v128_private",false);
                if(split&&(!ar_storage_r456||tp_size!=32||rank_info.pp_size!=1||head_dim!=192||num_heads!=96||model_config->get<size_t>("v_head_dim")!=128||config->num_blocks()!=132||config->block_size()!=256||config->max_batch_size()!=1||model_config->get_kv_cache_dtype()!=infinilm::runtime::DataType::kBFloat16))throw std::invalid_argument("private splitV exact AR allocator");
                kv[i] = split ? kimi_split_v128::allocate(runtime) : ar_storage_r456"""
   s=change(s,old,new)
  elif tag=='mla':
   s='#include "backend.hpp"\n'+s
   begin=s.index('    // Flash/paged attention caches use the Q/K head width.')
   end=s.index('    // Explicit candidate arithmetic:',begin)
   old=s[begin:end]
   # Preserve all original padding/attention code inside default branch.
   baseline=old[:old.index('    auto output = infinilm::runtime::Tensor::empty(')]
   new="""    auto storage=forward_context.kv_cache_vec.at(layer_idx_);
    const bool split=storage&&storage->ndim()==4&&storage->shape()==infinilm::runtime::Shape{132,256,3,320};
    infinilm::runtime::Tensor padded_output,split_output;
    if(split){
        if(local_num_heads_!=3||q_head_dim_!=192||v_head_dim_!=128||batch_size!=1)throw std::invalid_argument("splitV private caller geometry");
        const char*trace=std::getenv("INFINILM_KIMI_TRACE_KV_POOL_R8511");if(trace&&std::string(trace)=="1")throw std::invalid_argument("legacy8511 trace does not describe splitV schema");
        if(infinilm::models::kimi_native::spec_math_mode_r3676(runtime)!=0)throw std::invalid_argument("splitV only ordinary AR Math0");
        split_output=kimi_split_v128::forward(runtime,q,key,value,storage,forward_context.attn_metadata,softmax_scale_)->view({batch_size,seq_len,local_num_heads_,v_head_dim_});
    }else{
"""+baseline.replace('    auto padded_output =','    padded_output =')+"""    }
    auto output=infinilm::runtime::Tensor::empty(runtime,{batch_size,seq_len,local_num_heads_,v_head_dim_},split?split_output->dtype():padded_output->dtype());
"""
   s=s[:begin]+new+s[end:]
   s=change(s,'const auto ops_input = padded_output->narrow({{3, 0, v_head_dim_}});','const auto ops_input = split ? split_output : padded_output->narrow({{3, 0, v_head_dim_}});')
  else:
   s='#include "contract.hpp"\n'+s
   s=change(s,'thread_local bool authorized_vocab_embedding_r461=false;', """thread_local bool authorized_vocab_embedding_r461=false;
thread_local bool authorized_split_v128=false;
struct SplitScope{SplitScope(){if(authorized_split_v128)throw std::invalid_argument("nested splitV create");authorized_split_v128=true;}~SplitScope(){authorized_split_v128=false;}};""")
   s=change(s,'struct ProbeEngine:en::InferEngine{','struct ProbeEngine:en::InferEngine{bool split_profile()const{return model_config_->get_config_json().value("infinilm_kimi_k192_v128_private",false);}')
   s=change(s,'auto&text=cfg.at("text_config");text["infinilm_kimi_long_agent_r179"]', """const bool split_v128=cfg.value("infinilm_kimi_k192_v128_private",false);kimi_split_v128::authorize(split_v128,authorized_split_v128,ar_storage_r456,vocab_r461,tp_nodes,pp_size,max_rows);auto&text=cfg.at("text_config");text["infinilm_kimi_k192_v128_private"]=split_v128;text["infinilm_kimi_long_agent_r179"]""")
   start=s.index('extern "C" const char*modern_kimi_ar_storage_capabilities_r456(');stop=s.index('\n\nextern "C" void* modern_kimi_engine_create_vocab_parallel_r461',start)
   old=s[start:stop];newcaps=old.replace('modern_kimi_ar_storage_capabilities_r456','modern_kimi_k192_v128_capabilities_private').replace('if(!geometry_context_r453(c)', 'if(!c||!c->engine->split_profile()||!geometry_context_r453(c)').replace('{"MLA_shape",{2,132,256,3,192}}','{"MLA_shape",{132,256,3,320}},{"K_dim",192},{"V_dim",128},{"private_FA_query_tile",32},{"private_FA_key_tile",3072}')
   guarded=old.replace('if(!geometry_context_r453(c)','if((c&&c->engine->split_profile())||!geometry_context_r453(c)')
   s=s[:start]+guarded+s[stop:]
   s+='\n'+newcaps+"""
extern "C" void* modern_kimi_engine_create_k192_v128_private(const char*raw,int node,const char*address,int port){try{
 const char*opt=std::getenv("INFINILM_KIMI_K192_V128_PRIVATE");if(!opt||std::string(opt)!="1"||!raw)throw std::invalid_argument("splitV explicit private defaultOFF factory");
 auto cfg=nlohmann::json::parse(raw);if(!cfg.value("infinilm_kimi_k192_v128_private",false))throw std::invalid_argument("splitV explicit config required");
 SplitScope authorization;return modern_kimi_engine_create_vocab_parallel_r461(raw,node,address,port);
}catch(const std::exception&e){error=e.what();return nullptr;}}
"""
  target=out/(tag+'.cpp');target.write_text(s,encoding="utf-8",newline="\n");proof[tag]=dict(parent=str(path),parent_SHA256=expected[tag],private=str(target),private_SHA256=H(target))
 return proof
