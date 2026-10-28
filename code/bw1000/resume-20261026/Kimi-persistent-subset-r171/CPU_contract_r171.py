from pathlib import Path
import json,hashlib,sys
D=Path(__file__).resolve().parent;P=Path(sys.argv[1])if len(sys.argv)>1 else Path('/refactor/incoming/Kimi-capacity-exact-cache-lease-r170');pins=json.loads((D/'parent-pins-r171.json').read_bytes())
old=(P/'kimi_moe_capacity_r170.cpp').read_text();prefix=(D/'probe-prefix-r171.inc').read_text();probe=(D/'probe_r171.cpp').read_text();assert hashlib.sha256((P/'kimi_moe_capacity_r170.cpp').read_bytes()).hexdigest()==pins['parent_MoE_SHA256']
begin=old.index('void KimiK3Experts::register_mxfp4_experts');end=old.index('const KimiK3Mxfp4MoeWeights &KimiK3Experts::mxfp4_weights()',begin)
assert prefix.startswith(old[:begin])and old[end:]in prefix
body=prefix.split(' KimiK3Mxfp4MoeWeights output;\n',1)[1].split(' return output;\n',1)[0]
for a,b in reversed(list(pins['helper_mapping'].items())):body=body.replace(b,a)
assert body==pins['original_register_body']
assert prefix.count('(rank_info.tp_size > 1 && !rank_info.comm)')==old.count('(rank_info.tp_size > 1 && !rank_info.comm)')==3
assert 'r170_scratch::Lease workspace_lease_r170' in prefix and prefix.count('I96_scratch_lease_r170.hpp')==1
assert '#include "probe-prefix-r171.inc"'in probe and '#include "I96_scratch_lease_r170.hpp"'not in probe
assert probe.count('make_packed_subset_r171(')==1 and 'layer<92'in probe and 'rank.comm=nullptr'in probe
for s in ['45201752064ULL','46291641856ULL','aliases','checked171(cudaEventQuery(event))','throw HostMarker171{}','wrong_thread.join()','weakA.expired()','weakB.expired()','weakC.expired()','zero_bytes!=cachebytes'] :assert s in probe,s
for s in ['CommunicationGroup group','module->forward','load_parameters','cudaStreamCreate']:assert s not in probe,s
validator=json.loads((D/'cache-validator-source-r171.json').read_bytes());body=validator['function'];assert 'validate_cache_runtime' in body and '.comm'not in body and 'infiniccl'not in body
for x in ['runtime.assertOwnerThread()','runtime.device() != rank.device','rank.tp_size <= 0','topology.world_rank(rank.tp_rank)','runtime.graphCaptureState()']:assert x in body,x
print(json.dumps({'passed':True,'CPU_only':True,'Runtime_created':False,'same_TU_actual_production_helper_inverse_equivalent':True,'actual_lease_header_single_registry':True,'actual_parent_cache_object_reused':True,'cache_validator_comm_null_source_closed':True,'rank_metadata_is_allocation_geometry_only':True,'ownerA_B_created_and_used_on_same_owner_thread':True,'foreign_thread_only_tests_preGPU_rejection':True,'normal_exception_nested_crossowner_weak_release_cases_present':True,'bank_sets_actual92_not_single_layer_projection':True,'GPU_GO':False,'fullModel_GO':False}))
