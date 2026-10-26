namespace infinilm::models::glm_moe_dsa {void glm_fp8_m15_policy_r119(rt::Runtime&,bool);}
extern "C" int modern_glm_model_m15_cache_policy_r119(void*raw,int enabled){
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try{if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("r119 policy context/value");
  auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("r119 policy requires reset and idle owner");
  r.syncStream();model::glm_fp8_m15_policy_r119(r,enabled!=0);return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}
