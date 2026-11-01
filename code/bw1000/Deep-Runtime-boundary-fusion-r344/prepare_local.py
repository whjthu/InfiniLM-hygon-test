import pathlib,json,hashlib
repo=pathlib.Path('.');S=repo/'code/bw1000/Deep-Runtime-boundary-fusion-r344';ctx=json.loads((S/'remote-source-context.json').read_text());base=next(v['text']for n,v in ctx.items()if n.endswith('_r19.cpp'));assert next(v['SHA256']for n,v in ctx.items()if n.endswith('_r19.cpp'))=='8e3160a8d3defff085c888dea3d78bf1d09f0b8d3c4d6cbf2d8f85735041feee'
a=base.index('void TemporalCompressor::rotate(');z=base.index('\nTemporalCompressor::Batch TemporalCompressor::prepare(',a);rotate=base[a:z];body=rotate[rotate.index('{')+1:rotate.rfind('}')];k=body.index('    std::vector<int64_t>rows');body=body[:k]+'''    const int rc=run340(input->data(),weight->data(),x->data(),cache->data(),int(n),int(dim_),256,1,1,eps,r.stream());
    if(rc)throw std::runtime_error("Deep344 fused launch error="+std::to_string(rc));
    r.syncStream();
''';helper='''static void fused344(rt::Runtime&r,const rt::Tensor&input,const rt::Tensor&weight,rt::Tensor x,const std::vector<int64_t>&positions,size_t dim_,double theta_,double factor_,double beta_fast_,double beta_slow_,size_t original_,float eps){'''+body+'}\n'
base=base[:a]+helper+base[a:];needle='''        batch.rotated=norm_->forward(r,selected);work.held.push_back(batch.rotated);
        rotate(r,batch.rotated,rotary_positions);''';assert base.count(needle)==1
replacement='''        if(deep344_policy.load(std::memory_order_relaxed)==1 && dim_==512){
            batch.rotated=work.alloc(selected->shape(),selected->dtype());
            fused344(r,selected,norm_->weight(),batch.rotated,rotary_positions,dim_,theta_,factor_,beta_fast_,beta_slow_,original_,float(norm_->eps()));
            deep344_fused.fetch_add(1,std::memory_order_relaxed);
        }else{
            batch.rotated=norm_->forward(r,selected);work.held.push_back(batch.rotated);
            rotate(r,batch.rotated,rotary_positions);
            deep344_baseline.fetch_add(1,std::memory_order_relaxed);
        }''';base=base.replace(needle,replacement);insert='''extern "C" int run340(const void*,const void*,void*,const void*,int,int,int,int,int,float,void*);
''';base=insert+base;base=base.replace('namespace infinilm::models::deepseek_v4 {','namespace infinilm::models::deepseek_v4 {\nnamespace {std::atomic<int>deep344_policy{0};std::atomic<uint64_t>deep344_fused{0},deep344_baseline{0};}\n',1);base+='''
extern "C" int modern_deep344_mode(int mode){if(mode<0||mode>1)return 1;infinilm::models::deepseek_v4::deep344_policy.store(mode);return 0;}
extern "C" uint64_t modern_deep344_calls(int fused){return fused?infinilm::models::deepseek_v4::deep344_fused.load():infinilm::models::deepseek_v4::deep344_baseline.load();}
''';(S/'compressor344.cpp').write_bytes(base.encode())
bridge=(repo/'code/bw1000/deep-c128-weight-cache/deep_C128_component_bridge_r8694.cpp').read_bytes();(S/'bridge344.cpp').write_bytes(bridge)
c=json.loads((repo/'code/bw1000/resume-20261026/Deep-C128-weight-cache-r172/source-context-r172.json').read_text());u=next(u for u in c['units']if u['source'].endswith('compressor_fp32_c128_r19.cpp'));(S/'build-context.json').write_bytes((json.dumps({'unit':u,'parent_receipt':'/refactor/candidates/deep-c4-smallt-model-r9254/CPU-build.json','parent_receipt_SHA256':c['current_CPU_receipt_SHA256'],'kernel':'/refactor/incoming/Deep-operator-search-r340/lib340.so','kernel_SHA256':'7c5690fd55ca0a7894884b9c4dbddefd373740398b5d21a153becd51cea1a5f3','parent_compressor_SHA256':'8e3160a8d3defff085c888dea3d78bf1d09f0b8d3c4d6cbf2d8f85735041feee'},indent=2)+'\n').encode());print('SOURCE_PREPARED',len(base))
