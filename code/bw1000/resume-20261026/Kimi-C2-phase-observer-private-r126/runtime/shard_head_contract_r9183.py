"""Actual private owner/head type and CPU-source row checks, before any model adoption."""
import json

def read_head(lib,node,pp,loaded=False):
 raw=lib.modern_kimi_shard_head_info_r9183();assert raw,'missingactualheadinfo';q=json.loads(raw);assert q['schema']==1 and q['head_kind']=='ShardOnlyVocabHeadR9183'and not q['legacy_full_shard_policy_switch_supported'];assert len(q['owners'])==8
 tail=node//2==pp-1
 for dev,o in enumerate(q['owners']):
  assert o['device']==dev and o['model_constructed']and 'KimiK3ForConditionalGenerationShardR9183'in o['model_RTTI']and o['TP']==16 and o['tp_rank']==(node%2)*8+dev and o['PP']==pp and o['PP_stage']==node//2 and o['head_expected']==tail
  if tail:
   assert o['head_constructed']and 'ShardOnlyVocabHeadR9183'in o['head_RTTI']and not o['head_is_ReplicatedLinear']and not o['full_GPU_head_retained'];assert o['physical_head_shape']==[10240,7168]and o['head_parameter_bytes']==146800640 and o['global_row_begin']==o['tp_rank']*10240 and o['global_row_end_exclusive']==(o['tp_rank']+1)*10240
   if loaded:assert q['expected_CPU_source_registered']and o['postload_rows_verified']and o['CPU_expected_row_digests']==o['GPU_actual_row_digests']
  else:assert not o.get('head_constructed',False)
 return q

def expected_from_CPU(lib,w,torch):
 assert w.device.type=='cpu'and w.dtype==torch.bfloat16 and list(w.shape)==[163840,7168]and w.is_contiguous();bits=w.view(torch.uint16).numpy();ranks=[]
 for rank in range(16):
  digests=[]
  for row in [0,5120,10239]:
   h=1469598103934665603
   for value in bits[rank*10240+row]:h=((h^int(value))*1099511628211)&0xffffffffffffffff
   digests.append(h)
  ranks.append({'rank':rank,'row_begin':rank*10240,'digests':digests})
 rc=lib.modern_kimi_shard_head_expected_r9183(json.dumps({'key':'language_model.lm_head.weight','ranks':ranks}).encode())
 if rc:raise RuntimeError(lib.modern_kimi_shard_head_error_r9183().decode())
 return {'registered_ranks':16,'checked_local_rows':[0,5120,10239],'digest':'FNVword16 diagnostic, no rawweights stored','global_CPU_shape':[163840,7168],'existing_Weight_select_source_CPU_before_H2D':True,'source_CPU_lifetime':'originalpendingbuffers retained until Engine load flush ownerfence'}
