from pathlib import Path
s=Path('kimi_TP16_EP2_full_MoE_r8669zj.cpp').read_text().replace('r8669zj','r8669zw')
s=s.replace('modern_kimi_EP2_counts_r8669zw(int,uint64_t*,size_t)','modern_kimi_official_counts_r8669zu(int,uint64_t*,size_t)')
needle='for(int kind=0;kind<2;++kind){auto config='
s=s.replace(needle,'for(int kind=0;kind<2;++kind){phase.wait();if(rank==0){setenv("INFINILM_KIMI_OFFICIAL_WFP4_R8669ZT",kind?"1":"0",1);setenv("INFINILM_KIMI_W13_REPACKED_R8669ZT",kind?"1":"0",1);}phase.wait();auto config=')
s=s.replace('if(kind){ep.backend=infinilm::layers::moe::EPBackend::LocalAllReduce;ep.ep_size=2;ep.ep_rank=node;}','')
needle='for(int kind=0;kind<2;++kind){auto&module=modules[kind];'
new='''for(int kind=0;kind<2;++kind){
 phase.wait();if(rank==0){
  setenv("INFINILM_KIMI_OFFICIAL_WFP4_R8669ZT",kind?"1":"0",1);setenv("INFINILM_KIMI_W13_REPACKED_R8669ZT",kind?"1":"0",1);
  if(kind){size_t keys=0,bytes=0;auto begin=std::chrono::steady_clock::now();
   for(auto&kv:weights)if(kv.first.rfind("experts.",0)==0&&(kv.first.find(".w1.weight_packed")!=std::string::npos||kv.first.find(".w3.weight_packed")!=std::string::npos)){
    auto&t=kv.second;if(t->dtype()!=rt::DataType::kUInt8||t->shape()!=rt::Shape{3072,1792}||!t->is_contiguous())throw std::runtime_error("actualprivatecheckpointW13repackshape");
    auto*p=static_cast<uint8_t*>(t->data());for(size_t j=0;j<t->nbytes();j+=4){const uint8_t a=p[j],b=p[j+1],c=p[j+2],d=p[j+3];p[j]=(a&15)|((b&15)<<4);p[j+1]=(c&15)|((d&15)<<4);p[j+2]=(a>>4)|(b&240);p[j+3]=(c>>4)|(d&240);}++keys;bytes+=t->nbytes();
   }
   if(keys!=1792||bytes!=size_t(896)*2*3072*1792)throw std::runtime_error("repackall896expertFQNcoverage");row["CPU_private_repack"]={{"keys",keys},{"bytes",bytes},{"milliseconds",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()},{"temporary_scalar_bytes",4},{"after_original_context_all_owner_load_drain",true}};
  }
 }phase.wait();auto&module=modules[kind];'''
assert needle in s;s=s.replace(needle,new)
s=s.replace('{"global_expert_start",kind?node*448:0},{"local_experts",kind?448:896},{"expert_TP_rank",kind?info.tp_rank%8:info.tp_rank},{"expert_TP_size",kind?8:16}','{"global_expert_start",0},{"local_experts",896},{"expert_TP_rank",info.tp_rank},{"expert_TP_size",16},{"W13_repacked",bool(kind)}')
s=s.replace('if(rank==0)setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","1",1);','if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","1",1);setenv("INFINILM_KIMI_OFFICIAL_WFP4_R8669ZT",plan[arm]?"1":"0",1);setenv("INFINILM_KIMI_W13_REPACKED_R8669ZT",plan[arm]?"1":"0",1);}')
s=s.replace('uint64_t ep_before[3],ep_after[3];if(modern_kimi_EP2_counts_r8669zw(rank,ep_before,3))','uint64_t ep_before[2],ep_after[2];if(modern_kimi_official_counts_r8669zu(rank,ep_before,2))')
s=s.replace('{uint64_t(!plan[arm]),uint64_t(!plan[arm]*16)}','{1,16}')
old='if(modern_kimi_EP2_counts_r8669zw(rank,ep_after,3)||ep_after[0]-ep_before[0]!=uint64_t(plan[arm])||ep_after[1]-ep_before[1]!=uint64_t(plan[arm]*16)||(plan[arm]&&ep_after[2]!=uint64_t(node*448)))'
s=s.replace(old,'if(modern_kimi_official_counts_r8669zu(rank,ep_after,2)||ep_after[0]-ep_before[0]!=uint64_t(plan[arm])||ep_after[1]-ep_before[1]!=uint64_t(plan[arm]*16))')
s=s.replace('"EP2_counter_after",std::array<uint64_t,3>{ep_after[0],ep_after[1],ep_after[2]}','"official_W13_counter_after",std::array<uint64_t,2>{ep_after[0],ep_after[1]}')
s=s.replace('if(rank==0)setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);','if(rank==0){setenv("INFINILM_KIMI_I192_GROUPED_R8669Y","0",1);setenv("INFINILM_KIMI_OFFICIAL_WFP4_R8669ZT","0",1);setenv("INFINILM_KIMI_W13_REPACKED_R8669ZT","0",1);}')
assert 'modern_kimi_EP2_counts'not in s and 'EPBackend::LocalAllReduce'not in s and 'ep_after[2]!='not in s
Path('kimi_TP16_official_WFP4_r8669zw.cpp').write_bytes(s.encode());print('paired adapter ready',len(s))
