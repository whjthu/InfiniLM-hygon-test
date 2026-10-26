import pathlib,json,copy,hashlib,sys
from deep_owner_consumer_r9234 import native_expected_kinds,validate_inner,validate_owner
r=pathlib.Path('/refactor');cfg=json.loads(pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json').read_text());kinds=native_expected_kinds(cfg);seed=r/'results/deep-attention-inner-r9193b-910374.1073.json';source=json.loads(seed.read_bytes());profile=source['owner_prefill_profiles'][0];inner=profile['inner_attention']['raw_inner_events'];old=profile['raw_owner_events'];validate_owner(json.dumps(old),1);cases=[]
for T in [32,7,2]:
 q=copy.deepcopy(inner)
 for row in q['rows']:row['tokens']=T;row['requests']=T
 result=validate_inner(json.dumps(q),kinds,T,T);assert result['owner_rows']==344 and result['all_GPUevents_destroyed_before_CAPIreturn'];cases.append(T)
 try:validate_inner(json.dumps(q),kinds,T+1,T+1)
 except AssertionError:pass
 else:raise AssertionError('geometry mismatch not rejected')
import subprocess
nm=subprocess.check_output(['nm','-D','--defined-only','/refactor/candidates/deep-mode1-sink-r9207/libdeep_mode1_sink_r9207.so'],text=True)
assert all(v in nm for v in ['deep_owner_arm_r8900','deep_owner_read_r8900','deep_attention_inner_arm_r9193','deep_attention_inner_read_r9193','modern_deep_precision_stats_r3443','modern_deep_FP32_calls_r3597','modern_deep_latent_fused_calls_r681','modern_deep_latent_fallback_calls_r681'])
assert 'torch'not in sys.modules
print(json.dumps({'CPU_only':True,'schema_passed':True,'source_scalar_packet_SHA256':hashlib.sha256(seed.read_bytes()).hexdigest(),'mock_geometry_cases':cases,'scope':'RecordedPF scalarpacket structure reused only forCPU schema geometry tests; noactualdecodeGPUtimingqualification.','native_main_layers':len(kinds),'noTorch':True,'noGPU':True}))
