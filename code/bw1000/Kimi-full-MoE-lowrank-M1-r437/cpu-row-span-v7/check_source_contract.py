from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent;P=D.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
 for p,h in json.loads((D/'SOURCE_REFERENCES.json').read_bytes()).items():assert sha(p)==h,p
 assert (D/'private_moe.cpp').read_bytes()==(P/'private_moe.cpp').read_bytes(),'MoE body/router/experts/norm/TP/shared changed'
 c=(D/'component.cpp').read_text();h=(D/'projection.hpp').read_text();r=(D/'row_contract.hpp').read_text()
 parent=(P/'cpu-first-submit-v5/component.cpp').read_text()
 reversed_c=c.replace('kimi_r437::DiagnosticScope descriptor_scope(storage,stem,rank,info.tp_rank,node);','').replace('kimi_r437::diagnostic.arm=arm;','')
 assert reversed_c==parent,'Extra producer/math/owner ordering change'
 assert c.index('auto input=storage->narrow')<c.index('DiagnosticScope descriptor_scope')<c.index('auto y=plan[arm]?')
 assert 'class'not in r and 'rows==1&&columns==k&&rowpitch>0&&inner==1'in r
 assert 'rowpitch==static_cast<ptrdiff_t>(k)&&inner==1'in r
 assert 'if(!selected)return layer->forward(r,input);'in h
 assert h.index('rt::Tensor weight,output;')<h.index('struct FailureDrain')<h.index('output=rt::Tensor::empty')
 for name in ['input_exact_2D_1K','input_BF16','input_same_device','input_positive_unused_rowpitch_and_inner1','weight_exact_NK_row_contiguous','current_cuda_device_owned','not_graph','actual_local_rank_device','down_actual_data_span_within_strong_storage']:
  assert name in h
 assert 'pred.json()'in h and 'if(diagnostic.arm==1||'in h and 'std::array<std::pair<const char*,bool>,24>'in h
 assert h.index('persist(r,which,input,weight,output,pred,"guard_rejected"')<h.index('throw std::invalid_argument("437 exact physical single-row')
 assert 'UNKNOWN_not_exposed_publicly'in h and 'observed_byte_displacement_from_canary_storage'in h and 'logical_offset0_source_guarantee'in h
 assert 'cudaMemGetAddressRange'not in h and 'affinity_r2392::apply_once'not in c and 'kimi_r437_owner::Pin pin'not in c
 return dict(passed=True,actual_new_source_order_and_lease=True,original_MoE_body_byte_identical=True,all_source_refs_SHA_verified=True,only_single_row_physical_span_contract=True,formal_predicates_stack_only=True,descriptor_io_first_ON_warm_or_fail=True,firstsubmit_unchanged=True,Native_or_device_executed=False,GPU=0)
if __name__=='__main__':print(json.dumps(check()))
