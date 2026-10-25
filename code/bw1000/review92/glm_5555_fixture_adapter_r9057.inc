// Qualification-only adapter on the existing Context/Runtime/TP communicator.
// No model weights are loaded or read. Q/K/V are explicit caller fixtures.
namespace infinilm::models::glm_moe_dsa {rt::Tensor sparse_attention_exact_rows_r5555(rt::Runtime&,const rt::Tensor&,const std::vector<rt::Tensor>&,const std::vector<rt::Tensor>&,int);}
extern "C" int glm_attention_fixture_r9057(void*raw,size_t B,size_t P,const void*qptr,const void*kptr,const void*vptr,void*out){
 auto*c=static_cast<Context*>(raw);bool own=false;
 try{owner(c);own=true;if(B!=16||P!=513)throw std::invalid_argument("9057 bounded5555fixture");for(auto p:{qptr,kptr,vptr,static_cast<const void*>(out)})input(c,p);
 Work w{*c->r};w.held={rt::Tensor::from_blob(const_cast<void*>(qptr),{B,8,256},rt::DataType::kBFloat16,c->r->device()),rt::Tensor::from_blob(const_cast<void*>(kptr),{B,P,8,256},rt::DataType::kBFloat16,c->r->device()),rt::Tensor::from_blob(const_cast<void*>(vptr),{B,P,8,256},rt::DataType::kBFloat16,c->r->device()),rt::Tensor::from_blob(out,{B,8,256},rt::DataType::kBFloat16,c->r->device())};
 std::vector<rt::Tensor>keys,values;for(size_t row=0;row<B;++row){keys.push_back(w.held[1]->narrow({{0,row,1}})->view({P,8,256}));values.push_back(w.held[2]->narrow({{0,row,1}})->view({P,8,256}));}
 auto begin=Clock::now();double cp=cpu();auto result=model::sparse_attention_exact_rows_r5555(*c->r,w.held[0],keys,values,1);w.held.push_back(result);w.held[3]->copy_from(*c->r,result);c->r->syncStream();serialized=nlohmann::json{{"N",B},{"B",B},{"P",P},{"entry",5555},{"backend",1},{"wall_ms",std::chrono::duration<double,std::milli>(Clock::now()-begin).count()},{"threadCPU_ms",(cpu()-cp)*1000},{"model_weights_loaded",false},{"profile",0},{"stages",nlohmann::json::array()}}.dump();return 0;
 }catch(const std::exception&e){error=e.what();if(own)drain(c);return 1;}
}
