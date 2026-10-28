"""Local CPU source contracts; imports no torch/Native and submits no jobs."""
import ast,json,pathlib,sys,importlib.util
D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(D/'runtime'))
for p in D.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
source=(D/'runtime/probe_r183.py').read_text();tree=ast.parse(source)
calls=[node.func.attr for node in ast.walk(tree)if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)]
assert calls.count('modern_kimi_engine_create_long_agent_r179')==1
for forbidden in ['modern_kimi_engine_create_batch_r9122c','modern_kimi_engine_load_r2475','modern_kimi_engine_finish_r2475','modern_kimi_engine_forward_r2475','modern_kimi_batch_forward_r9122','arange','zeros','ones','randn','tensor']:assert forbidden not in calls,forbidden
assert 'for ordinal in [1,2]'in source and 'initial_create_cache_allocation' in source and 'UNKNOWN_NOT_USED_AS_CACHE_FIT_PROOF'in source
assert source.index('modern_kimi_batch_reset_r9122(ctx,0)')<source.index("actual_cache_constructed_after_first_reset']=True")
assert 'Native_pool_reserved' in source and 'exact_transient_peak'in source
import probe_r183
fixture={'text_config':{'linear_attn_config':{'kda_layers':list(range(1,70))}}}
a=probe_r183.cache_plan(fixture,0);b=probe_r183.cache_plan(fixture,1);assert a['layers']+b['layers']==93 and a['KDA_layers']+b['KDA_layers']==69 and a['MLA_layers']+b['MLA_layers']==24
assert a['MLA_each_bytes']==2*260*256*6*192*2 and a['KDA_ssm_each_bytes']==3*6*128*128*2 and a['KDA_conv_each_bytes']==3*2304*3*2
pins=json.loads((D/'pins-r183.json').read_bytes());assert pins['Native_SHA256']=='ba6f9a01e459c5b57839677160cd916cba29312ee8355c939b898bc591ac7d81'and pins['CPU_receipt_SHA256']=='37ac44052ffb3b0e43cf7e440178aad4096f09dcd97a8b12d6eca6fc65cd827d'
from contract_r183 import NODES,PARENT
assert PARENT=='910380'and NODES==['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
assert 'torch'not in sys.modules and 'infinilm'not in sys.modules
print(json.dumps({'passed':True,'CPU_only':True,'GPU':False,'Native_dlopen':False,'explicit_long_factory_only':True,'two_legal_Loading_resets':True,'no_payload_finish_forward_or_fake_GPU_math':True,'initial_cache_UNKNOWN_then_real_reset_allocation_evidence':True,'sampler_min_is_observed_exact_peak_UNKNOWN':True,'cache_plan_fixture_is_not_actual_device_proof':True,'all32_cleanup_postKFD_required':True,'Model_GO':False}))
