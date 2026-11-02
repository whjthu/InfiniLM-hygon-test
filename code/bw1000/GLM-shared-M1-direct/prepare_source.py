"""Mechanical one-HOST-TU patch; original source is byte-pinned and recoverable."""
from pathlib import Path
import hashlib,json
D=Path(__file__).resolve().parent
original=(D/'original_ffn.cpp').read_bytes()
assert hashlib.sha256(original).hexdigest()=='c716b37fd3fe786cece00a5b4a309d9eb13b76e4bbdb8cc8843be936c708e64b'
s=original.decode();replacements=[]
def patch(old,new):
 global s
 assert s.count(old)==1,old[:120]
 s=s.replace(old,new,1);replacements.append((old,new))
patch('Fp8MlpR19::Fp8MlpR19(', '#include "shared_M1_adapter.inc"\nFp8MlpR19::Fp8MlpR19(')
patch('size_t tokens=x->size(0);auto gate=gate_->forward(r,x),up=up_->forward(r,x);',
      'size_t tokens=x->size(0);bool direct=shared_m1_direct_v1::select(r,tokens,local_,global_);rt::Tensor gate,up;if(direct){auto pair=shared_m1_direct_v1::gateup(r,work,*gate_,*up_,x,offset_);gate=pair.first;up=pair.second;}else{gate=gate_->forward(r,x);up=up_->forward(r,x);}')
old='ex::submit<infini::ops::Fp8BlockLinear>(r,ex::RecordingPolicy::HostOnly,infini::ops::Fp8BlockLinear::DefaultConfig(r.device().type()),out,packed,global_scale,down_,scale_,size_t{0},offset_);decode_leases_r1354::success_fence(r);return out;'
patch(old,'if(direct)shared_m1_direct_v1::down(r,work,out,packed,local_scale,down_,scale_,offset_);else '+old)
api='''
extern "C" int glm_shared_M1_mode_v1(int device,int mode){if(device<0||device>=8||mode<0||mode>1)return 1;infinilm::models::glm_moe_dsa::shared_m1_direct_v1::states[device].enabled.store(mode);return 0;}
extern "C" int glm_shared_M1_counts_v1(int device,uint64_t*out,size_t n){if(device<0||device>=8||!out||n!=5)return 1;for(size_t i=0;i<5;++i)out[i]=infinilm::models::glm_moe_dsa::shared_m1_direct_v1::states[device].count[i].load();return 0;}
'''
s+=api
recover=s.removesuffix(api)
for old,new in reversed(replacements):
 assert recover.count(new)==1;recover=recover.replace(new,old,1)
assert recover.encode()==original
(D/'candidate_ffn.cpp').write_text(s,encoding='utf8',newline='\n')
print(json.dumps({'passed':True,'changed_HOST_TUs':1,'header_or_class_layout_changed':False,'GPU_TUs_changed':0,'original_byte_recovery':True,'original_SHA256':hashlib.sha256(original).hexdigest(),'candidate_SHA256':hashlib.sha256(s.encode()).hexdigest()}))
