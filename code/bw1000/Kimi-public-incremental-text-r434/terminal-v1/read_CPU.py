from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('observation');p.add_argument('--output');a=p.parse_args();raw=Path(a.observation).read_bytes();o=json.loads(raw);x=o['receipt'];b=x['rawtext'].encode();assert len(b)==x['bytes']==11341 and hashlib.sha256(b).hexdigest()==x['SHA256']=='e635bb278970a9c3f02c4e1e25eb9f997a1cf150be78dc0bea149700c825900d';q=json.loads(b)
assert q['completed']and q['passed']and q['actual_CPU']=='910407.876'and q['CPU_only']and q['GPU']==0 and not q['Native_dlopen']and not q['weight_payload_read']and not q['Model_READY'];assert q['source_manifest_SHA256']=='f0821642a154b0ca51172093b8bd4119b117c9977b5a2d22fbbddb53136be7b4'
assert len(q['cases'])==11 and all(v['passed']for v in q['cases']);assert '910407.876|'in o['accounting']['stdout']and '|COMPLETED|0:0|00:00:03|'in o['accounting']['stdout']
for name in ['initial_module_maps','final_module_maps']:
 m=q[name];assert not m['project_Native_maps']and not m['Torch_imported']and not m['Transformers_imported']and m['Torch_cuda_initialized']in(None,False);assert set(m['visible_env'].values())=={'-1'}
codec=q['actual_codec_identity'];assert codec['tokenizer_config_SHA256']=='5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e'and codec['tiktoken_model_SHA256']=='b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103'
rows=[v for v in q['cases']if v['case']=='actual_new_prepared_P0P1_original572_replay'];assert len(rows)==4
expected=[(8635,58,4096,'66090df28f4a0b8024a03f769dd9d1f76c91a0348ea7141a31c48ae96f373934'),(9613,121,8192,'520bd483280ac3da16fbc2cda5b9242a68176f3f93e95de1fac61035e479960a')]
for index,(tokens,visible,budget,h)in enumerate(expected):
 group=[v for v in rows if v['index']==index];assert[v['mode']for v in group]==['P0','P1']
 for v in group:
  assert(v['prompt_tokens'],v['visible_tokens'],v['output_budget'],v['prompt_IDs_SHA256'])==(tokens,visible,budget,h)
  assert v['all_generated_including_EOS']==visible+1 and v['full_history']and v['owned_finish_after_recorded_reset']and v['queue_retired']and v['reasoning_content_and_typed_tool_arguments_match_original_wire']and v['fresh_tool_IDs_valid']
  assert v['early_raw_before_owned_finish']==v['early_public_semantic_event_before_owned_finish']==(v['mode']=='P1')
  assert v['selection_before_FIFO']==[dict(text_delivery=v['mode'],history_qualified=True)]
prefix=[v for v in q['cases']if v['case']=='actual_K3_allprefix_and_public_typed_parser'];assert[v['prefixes']for v in prefix]==[58,121]
cli=q['cases'][0]['roles'];assert len(cli)==4 and[v['topology']['node']for v in cli]==[0,1,2,3]and all(v['source_and_config_valid']and not v['GPU']and not v['Native_dlopen']for v in cli)
assert q['actual_checkpoint_codec']and q['actual_public_protocol_and_parser']and q['actual_new_prepared_and_original_history_queue']and q['original_572_861_not_reexecuted']and q['whole_local_recording6cases_not_repeated']
v=dict(passed=True,state='CPU_INTERFACE_QUALIFIED_NOT_DEVICE_HTTP_MODEL',actual_CPU=q['actual_CPU'],receipt_SHA256=x['SHA256'],source_manifest_SHA256=q['source_manifest_SHA256'],cases=11,allprefix_tokens=179,actual_new_prepared_rows=rows,module_maps=q['final_module_maps'],codec_identity=codec,private_artifacts=q['private_artifacts'],wall_s=q['wall_s'],GPU_GO=False,Model_GO=False,default_P0=True,explicit_P1=True,limits=q['scaffolding'],prefix_timestamps_not_token_client_times=True,private_payload_uploaded=False)
text=json.dumps(v,indent=2)+'\n'
if a.output:Path(a.output).write_text(text,encoding='utf8',newline='\n')
else:print(text,end='')
