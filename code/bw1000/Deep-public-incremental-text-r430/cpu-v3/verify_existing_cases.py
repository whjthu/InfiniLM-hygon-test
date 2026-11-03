"""Read original863 nine results without changing its FAILED producer terminal."""
import argparse,hashlib,json
from pathlib import Path
def verify(path):
 archive=json.loads(Path(path).read_bytes());assert archive['accounting']==[next(r for r in archive['accounting']if r[0]=='910407.863')]
 assert archive['accounting'][0][2:4]==['FAILED','1:0']
 records=list(archive['files'].values());assert len(records)==1;r=records[0];raw=r['text'].encode();assert len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['SHA256']=='9b891f22c5ae5db932afa38e0df292b2db50f6e16ef7f58646e770fa16b4e083'
 q=json.loads(raw);assert q['completed']and q['passed']is True and q['error']=='AssertionError()'and 'maps=Path'in q['traceback']
 assert q['source_manifest_SHA256']=='e772efb87a321a07616d4f789f77d24eadec9ca90e54fa8e1c20c126834002bb'and q['production_source_manifest_SHA256']=='4ec029919491a59987dce840975898727149209749f9de322da9f3af73171a91'
 rows=q['cases'];assert len(rows)==9 and all(v['passed']is True for v in rows)
 prefixes={v['index']:v['prefixes']for v in rows if v['case']=='actual_Deep_allprefix_decoder_and_actual_public_parser'};assert prefixes=={0:97,1:343}
 actual=[v for v in rows if v['case']=='actual_private_factory_selector_public_parser'];assert len(actual)==4
 ids={0:'681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df',1:'d956ecfc84f1c221a7768772b868a237d02fb1e6df2b682291b9a58c9ce2e287'}
 for index in [0,1]:
  pair={v['mode']:v for v in actual if v['original_index']==index};assert set(pair)=={'P0','P1'}
  for mode,v in pair.items():assert v['actual_prompt_IDs_SHA256']==ids[index]and v['actual_prompt_tokens']==8489+index and v['final_semantics_match_original_wire']and v['first_before_owned_finish']==(mode=='P1')
  for key in ['final_content_SHA256','final_reasoning_SHA256']:assert pair['P0'][key]==pair['P1'][key]
 for name in ['actual_new_P1_selector_timeout_abort','Deep_actual_tokenizer_DSML_Unicode_complete_typed_tool','invalid_mode_rejected_by_actual_prepared_validate_before_queue']:assert len([v for v in rows if v['case']==name])==1
 assert q['original_Model_sequences']==2 and q['actual_checkpoint_tokenizer']and q['actual_public_parser']and q['actual_private_prepared_factory']and not q['real_HTTP_or_Native_Model']
 return dict(functional_case_scope_passed=True,functional_cases=9,original_decoder_prefixes=440,original_sequences=2,actual_prompt_tokens=[8489,8490],P0_P1_final_reasoning_content_equal=True,P1_first_before_recorded_owned_finish=True,producer_actual_CPU='910407.863',producer_actual_state='FAILED1/12s',producer_passed_flag='UNRELIABLE_PREMATURE_TRUE; original error/Slurm FAIL retained',original_receipt_SHA256=r['SHA256'],terminal_metadata_scope='UNKNOWN until separate actual import/maps probe',retroactive_producer_PASS=False,GPU_or_Model_qualified=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();q=verify(a.archive);raw=(json.dumps(q,indent=2)+'\n').encode()
 if a.output:a.output.write_bytes(raw)
 print(raw.decode(),end='')
