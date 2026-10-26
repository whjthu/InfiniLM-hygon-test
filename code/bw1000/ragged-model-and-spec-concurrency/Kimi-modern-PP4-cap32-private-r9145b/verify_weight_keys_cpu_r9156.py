import json,pathlib,copy
from weight_key_guard_r9156 import summarize_keys,validate_all_stages
ranges=[[0,24],[24,47],[47,71],[71,93]];packets=[];checks=0
for node in range(8):
 stage=node//2;keys={'language_model.model.layers.%d.self_attn.o_proj.weight'%i for i in range(*ranges[stage])}
 if stage==0:keys.add('language_model.model.embed_tokens.weight')
 if stage==3:keys.update(['language_model.lm_head.weight','language_model.model.norm.weight','language_model.model.output_attn_res_norm.weight'])
 packet={'node':node,'keys':summarize_keys(keys,{k:'shard'for k in keys},stage,ranges)};packets.append(packet);checks+=1
s=validate_all_stages(packets,ranges);checks+=1
for kind in ['wronglayer','embedwrongstage','headmissing','unknownnonlayer','checkpointunknown']:
 stage=0;keys={'language_model.model.layers.%d.self_attn.o_proj.weight'%i for i in range(24)}|{'language_model.model.embed_tokens.weight'};mapping={k:'x'for k in keys}
 if kind=='wronglayer':keys.remove('language_model.model.layers.23.self_attn.o_proj.weight')
 if kind=='embedwrongstage':stage=1
 if kind=='headmissing':stage=3;keys={'language_model.model.layers.%d.self_attn.o_proj.weight'%i for i in range(71,93)};mapping={k:'x'for k in keys}
 if kind=='unknownnonlayer':keys.add('language_model.unknown.weight');mapping={k:'x'for k in keys}
 if kind=='checkpointunknown':mapping.pop('language_model.model.embed_tokens.weight')
 try:summarize_keys(keys,mapping,stage,ranges)
 except AssertionError:checks+=1
 else:raise AssertionError(kind)
for kind in ['missingnode','duplicatenode','wrongstage','lostcoverage']:
 p=copy.deepcopy(packets)
 if kind=='missingnode':p.pop()
 if kind=='duplicatenode':p[-1]['node']=6
 if kind=='wrongstage':p[2]['keys']['stage']=0
 if kind=='lostcoverage':p[4]['keys']['actual_layer_ids'].pop()
 try:validate_all_stages(p,ranges)
 except AssertionError:checks+=1
 else:raise AssertionError(kind)
q={'CPU_only':True,'GPU':False,'math_mocked':True,'checks':checks,'source_validator_executed':True,'all8stagecoverage_positive_and_negative':True,'read_payload':False};pathlib.Path(__file__).with_name('CPU-weight-keys-r9156.json').write_text(json.dumps(q,indent=2));print(json.dumps(q))
