from pathlib import Path
import json,hashlib,sys,math
D=Path(__file__).resolve().parent;S=Path(sys.argv[1])if len(sys.argv)>1 else Path('/refactor/incoming/Kimi-TP32-I96-template-component-r163')
pins=json.loads((D/'parent-pins-r168.json').read_bytes());old=(S/'kimi_moe_I96_r163.cpp').read_text()
assert hashlib.sha256((S/'kimi_moe_I96_r163.cpp').read_bytes()).hexdigest()==pins['parent_regular_source_SHA256']
regular=(D/'kimi_moe_storage_regular_r168.cpp').read_text();exact=(D/'kimi_moe_storage_exact_r168.cpp').read_text()
assert regular.count('Tensor::empty_parameter(')==0 and exact.count('Tensor::empty_parameter(')==4
assert exact.replace('Tensor::empty_parameter(','Tensor::empty(')==regular,'allocator-only difference between variants'
begin=old.index('void KimiK3Experts::register_mxfp4_experts');end=old.index('const KimiK3Mxfp4MoeWeights &KimiK3Experts::mxfp4_weights()',begin)
original=old[begin:end];originalbody=original[original.index('{')+1:original.rfind('}')]
body=regular.split('    KimiK3Mxfp4MoeWeights output;\n',1)[1].split('    return output;\n',1)[0]
for a,b in reversed(list(pins['helper_semantic_replacements'].items())):body=body.replace(b,a)
assert body==originalbody,'shared helper must be exactly original allocator+narrow+squeeze+Weight bind order'
assert regular.startswith('#include <cuda_runtime_api.h>\n#include <infini/rt/device_guard.h>\n#include <set>\n'+old[:begin])
assert old[end:] in regular
assert regular.count('make_packed_banks_r168(')==3,'one helper, real member callsite, probe callsite'
assert regular.count('(rank_info.tp_size > 1 && !rank_info.comm)')==old.count('(rank_info.tp_size > 1 && !rank_info.comm)')==3
probe=regular.split('// Probe is the same helper',1)[1]
for forbidden in ['RankInfo ','CommunicationGroup ','module->forward','load_parameters','register_mxfp4_experts(']:assert forbidden not in probe,forbidden
assert 'state_dict()'in probe and 'leaf_aliases_checked'in probe and 'offset+t->nbytes()'in probe
assert all(s in probe for s in ['before_Runtime','after_Runtime','after_shared_banks_construct','after_shared_banks_destroy','after_Runtime_destroy'])
req=[896*192*1792,896*192*112,896*3584*48,896*3584*3];alloc=[req[0],2**25,2**28,2**24]
assert sum(req)==491323392 and sum(alloc)-sum(req)==135725056
print(json.dumps({'passed':True,'CPU_only':True,'Runtime_created':False,'same_shared_helper_actual_two_callsites':True,'original_register_body_equivalent':True,'unchanged_real_ctor_comm_guards':True,'no_fake_TP_world':True,'variants_only_four_allocator_calls_different':True,'no_weight_payload_or_forward':True,'one_owner_probe_only_banks_not_fullMoE':True,'expected_single_layer_class_delta_bytes':135725056,'all92_extrapolation_bytes':12486705152,'actual_GPU_bytes':'UNKNOWN','fullModel_peak':'UNKNOWN_STOP'}))
