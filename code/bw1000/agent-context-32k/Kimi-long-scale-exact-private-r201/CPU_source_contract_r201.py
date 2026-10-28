"""Source-only equivalence/binding; no imports of Torch/Native or job tools."""
import ast,hashlib,json,pathlib,sys
D=pathlib.Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=(D/'original_kimi_moe_batch_r9129.cpp').read_text();new=(D/'kimi_moe_scale_exact_r201.cpp').read_text()
assert sha(D/'original_kimi_moe_batch_r9129.cpp')=='a70701b1070fa999c30834cd39a34c182049922e52d4a6473692cdcb53779576'
t=new.replace('#include "scale_exact_scope_r201.hpp"\n','')
start=t.index('        kimi_scale_exact_r201::Scope scale_scope_r201(');end=t.index('        register_mxfp4_experts(runtime);',start);t=t[:start]+t[end:]
start=t.index('    auto allocate_scale_r201 = ');end=t.index('    auto w13 = ',start);t=t[:start]+t[end:]
for name,shape in [('w13_scale','{num_experts_, 2 * local_intermediate_size_, hidden_size_ / 32}'),('w2_scale','{num_experts_, hidden_size_, local_intermediate_size_ / 32}')]:
 old='    auto '+name+' = infinilm::runtime::Tensor::empty(\n        runtime, '+shape+',\n        infinilm::runtime::DataType::kUInt8);'
 candidate='    auto '+name+' = allocate_scale_r201('+shape+');';assert t.count(candidate)==1;t=t.replace(candidate,old)
assert t==original,'unexpected source change beyond scoped scale allocator selection'
suffix='    auto register_packed = '
assert new[new.index(suffix):]==original[original.index(suffix):],'actual5376aliases/forward/loader binding suffix changed'
assert 'Tensor::empty_parameter(runtime,shape,'in new and 'pin_memory'not in new
assert all(b'\r'not in p.read_bytes()for p in D.glob('*.sh'))
for p in D.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
assert 'torch'not in sys.modules and 'ctypes'not in sys.modules
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'Runtime_created':False,'source_reverse_exact':True,'actual5376aliases_and_allforward_bytes_unchanged':True,'only_two_persistent_scale_allocations_selected':True,'public_header_layout_unchanged':True,'fixture_not_device_allocation_proof':True}))
