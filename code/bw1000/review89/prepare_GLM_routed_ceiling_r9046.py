from pathlib import Path
import json,hashlib,ast
r=Path(__file__).resolve().parent
s=json.loads((r/'GLM-decode-MoE-source-r8756-b.json').read_bytes())['files'][0]['text']
assert hashlib.sha256(s.encode()).hexdigest()=='c716b37fd3fe786cece00a5b4a309d9eb13b76e4bbdb8cc8843be936c708e64b'
state='''
// Fixed-fixture diagnostic only. Default OFF, no production reuse policy.
struct RoutedCeiling9046 { int mode=0; rt::Runtime*owner=nullptr; const void*module=nullptr; size_t offset=0; rt::Tensor local,current_x,current_ids; std::vector<unsigned char>reference_x,reference_ids; uint64_t counts[3]={0,0,0}; };
thread_local RoutedCeiling9046 routed_ceiling9046;
'''
s=s.replace('struct ActualCaptureState5164',state+'struct ActualCaptureState5164',1)
a=s.index('  glm_layer_mark_r5670(r,"FFN.gateup.begin")');z=s.index('  actual_capture_field5164(r,capture.get(),"local_down_BF16"',a)
old=s[a:z]
old=old.replace(',local=work.alloc({routes,6144},rt::DataType::kBFloat16)','')
assert 'local=work.alloc' not in old
new='''   auto& diagnostic=routed_ceiling9046;
   const bool eligible=tokens==8 && routes==64 && routed_offset_%256==0;
   rt::Tensor local;
   if(eligible){diagnostic.current_x=x;diagnostic.current_ids=indices;}
   if(eligible && diagnostic.mode==1) {
    if(!diagnostic.local || diagnostic.owner!=&r || diagnostic.module!=this || diagnostic.offset!=routed_offset_)throw std::logic_error("routed ceiling fixture not captured");
    local=diagnostic.local;work.held.push_back(local);++diagnostic.counts[1];
   } else {
    local=work.alloc({routes,6144},rt::DataType::kBFloat16);
'''+old+'''    if(eligible) {
     if(diagnostic.mode==2) { diagnostic.local=local;diagnostic.owner=&r;diagnostic.module=this;diagnostic.offset=routed_offset_;++diagnostic.counts[2]; }
     else ++diagnostic.counts[0];
    }
   }
'''
s=s[:a]+new+s[z:]
s+='''
extern "C" int glm_routed_ceiling_policy_r9046(int mode) {
 using namespace infinilm::models::glm_moe_dsa;
 if(mode==-1){routed_ceiling9046.local.reset();routed_ceiling9046.current_x.reset();routed_ceiling9046.current_ids.reset();routed_ceiling9046.reference_x.clear();routed_ceiling9046.reference_ids.clear();routed_ceiling9046.owner=nullptr;routed_ceiling9046.module=nullptr;routed_ceiling9046.mode=0;return 0;}
 if(mode<0||mode>2)return -1;routed_ceiling9046.mode=mode;return 0;
}
extern "C" int glm_routed_ceiling_counts_r9046(uint64_t*out,size_t n) {
 using namespace infinilm::models::glm_moe_dsa;if(!out||n!=3)return -1;
 for(size_t i=0;i<3;++i)out[i]=routed_ceiling9046.counts[i];return 0;
}
extern "C" int glm_routed_ceiling_validate_r9046(uint64_t*out,size_t n) {
 using namespace infinilm::models::glm_moe_dsa;auto&s=routed_ceiling9046;
 if(!out||n!=3||!s.owner||!s.current_x||!s.current_ids||!s.local)return -1;
 try { s.owner->assertOwnerThread();std::vector<unsigned char>x(s.current_x->nbytes()),ids(s.current_ids->nbytes());
 s.owner->memcpyD2H(x.data(),s.current_x->data(),x.size());s.owner->memcpyD2H(ids.data(),s.current_ids->data(),ids.size());
 if(s.reference_x.empty()){s.reference_x=x;s.reference_ids=ids;}
 out[0]=x==s.reference_x;out[1]=ids==s.reference_ids;out[2]=x.size()+ids.size();return 0;
 } catch(...){return -2;}
}
'''
(r/'glm_routed_ceiling_r9046.cpp').write_bytes(s.encode())
print(json.dumps({'source':'glm_routed_ceiling_r9046.cpp','SHA256':hashlib.sha256(s.encode()).hexdigest(),'HOST_TUs':1,'GPU_TUs':0,'copy_bytes':0,'capture':'First warm only;zero-copy completed local BF16 Tensor alias; release on original owner before destroy','skip':'gateup/Silu/Mul/downquant/down;router/inputquant/merge/shared/TP/drain unchanged'}))

