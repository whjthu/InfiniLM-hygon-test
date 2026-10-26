"""Read-only stage ownership proof from actual engine parameter keys before payload load."""
import re,json,hashlib

def summarize_keys(keys,mapping,stage,ranges):
 keys=set(keys);assert keys and keys<=set(mapping),'engine keys not in actual checkpoint index'
 layers=sorted({int(x[1])for k in keys if(x:=re.search(r'^language_model\.model\.layers\.(\d+)\.',k))})
 expected=list(range(*ranges[stage]));assert layers==expected,('actual engine layer ownership mismatch',stage,layers,expected)
 other=sorted(k for k in keys if not re.search(r'^language_model\.model\.layers\.(\d+)\.',k))
 embed='language_model.model.embed_tokens.weight';head='language_model.lm_head.weight'
 assert(embed in keys)==(stage==0),'embedding stage ownership'
 assert(head in keys)==(stage==len(ranges)-1),'LMhead stage ownership'
 for k in other:
  assert k==embed or k==head or k.startswith(('language_model.model.norm.','language_model.model.output_attn_res_')),('unknown nonlayer engine key',k)
  if k!=embed:assert stage==len(ranges)-1,('tail parameter in non-tail stage',k,stage)
 return {'stage':stage,'actual_layer_ids':layers,'actual_range':[layers[0],layers[-1]+1],'key_count':len(keys),'keys_SHA256':hashlib.sha256(json.dumps(sorted(keys),separators=(',',':')).encode()).hexdigest(),'embedding_present':embed in keys,'lm_head_present':head in keys,'nonlayer_keys':other,'checkpoint_membership_validated':True}

def validate_all_stages(packets,ranges):
 assert len(packets)==2*len(ranges)and{p['node']for p in packets}==set(range(2*len(ranges)))
 by={p['node']:p['keys']for p in packets}
 for node,s in by.items():
  stage=node//2;assert s['stage']==stage and s['actual_layer_ids']==list(range(*ranges[stage]));assert s['embedding_present']==(stage==0)and s['lm_head_present']==(stage==len(ranges)-1)
 coverage=[i for stage in range(len(ranges))for i in by[2*stage]['actual_layer_ids']];assert coverage==list(range(93)),'full 93-layer exactly-once stage coverage'
 return {'model_weight_keys_validated':True,'all93_layer_ids_exactly_once_across4stages':True,'all8_nodes_verified':True,'PP_ranges':ranges,'pernode':by,'weight_payload_opened':False}
