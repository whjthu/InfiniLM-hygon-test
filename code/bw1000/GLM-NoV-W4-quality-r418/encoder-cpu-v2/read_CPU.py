"""Reconstruct original844 report/privateprompt bytes and verify all64 bindings."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,math
digest=lambda b:hashlib.sha256(b).hexdigest()
def read(path):
 outer=json.loads(Path(path).read_bytes());a=outer['archive'];assert digest((json.dumps(a,ensure_ascii=False,indent=2)+'\n').encode())==outer['archive_SHA256']
 assert a['actual_CPU']=='910407.844'and a['file_count']==len(a['files'])==65 and a['new_jobs']==a['GPU']==a['Native']==a['Model']==0
 for p,f in a['files'].items():
  raw=f['rawtext'].encode(f['encoding']);assert p==f['path']and len(raw)==f['bytes']and digest(raw)==f['SHA256']
 r=a['files'][a['receipt_path']];assert r['SHA256']==a['receipt_SHA256']=='2051a3a74e5ce471b53c38ddffd0f744e1a0669bdc2312590f6bd6d2a5013c12';q=json.loads(r['rawtext'])
 assert q['completed']and q['passed']and q['actual_checkpoint_CPU_processor']and not q['Torch_CUDA_initialized']and not q['unexpected_Native_library_maps']
 assert q['actual_CPU']=='910407.844'and len(q['cases'])==64 and q['unique_actual_encodings']==q['matching_body_pairs']==64 and q['planned_logical_requests']==128
 assert q['template_SHA256']=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'and q['source_manifest_SHA256']=='e9570b0e6d797bb2399e7fac58fd34c53093bbc8b92b670dc124f2bedc6bc149'
 base=Path(__file__).resolve().parent.parent;fixture=json.loads((base/'fixtures-r1/gsm8k-new64.json').read_bytes());plan=json.loads((base/'fixtures-r1/requests-and-pairs.json').read_bytes());assert digest((base/'fixtures-r1/requests-and-pairs.json').read_bytes())==q['requests_file_SHA256']=='b536c18715a60aebd593d181436503b65ab10f70258f7abc791f1c9426797183'
 assert digest((base/'fixtures-r1/gsm8k-new64.json').read_bytes())=='ec1d98eb5dc6a45c7ab06c54df707715854afaecc046b3b657d62a1b028de523';cases={r['id']:r for r in fixture['cases']}
 pbody={r['case_id']:r for r in plan['requests']};assert len(cases)==len(pbody)==64
 small=[]
 for ordinal,c in enumerate(q['cases']):
  assert c['case_id']in cases and c['source_index']==cases[c['case_id']]['source_index']
  body=pbody[c['case_id']]['body'];assert body['messages'][1]['content']==cases[c['case_id']]['question']and body['max_tokens']==4096 and body['temperature']==0
  assert digest(json.dumps(body,ensure_ascii=False,separators=(',',':')).encode())==c['request_body_SHA256']==pbody[c['case_id']]['compact_body_SHA256']
  fp='/public/home/aclsylqidf/hygon-inference-refactor-20261014/'+str(PurePosixPath(c['private_path']).relative_to('/refactor'));f=a['files'][fp];assert f['SHA256']==c['private_SHA256'];p=json.loads(f['rawtext'])
  assert p['case_id']==c['case_id']and p['request_body_SHA256']==c['request_body_SHA256'];ids=p['prompt_token_ids'];prompt=p['prompt']
  assert len(ids)==c['prompt_tokens']and all(type(i)is int and 0<=i<154880 for i in ids)and digest(json.dumps(ids,separators=(',',':')).encode())==c['prompt_IDs_SHA256']
  assert len(prompt.encode())==c['prompt_bytes']and digest(prompt.encode())==c['prompt_SHA256']and cases[c['case_id']]['question']in prompt and body['messages'][0]['content']in prompt
  assert c['prefix']=='<|assistant|><think>'and prompt.endswith(c['prefix'])and digest(c['prefix'].encode())==c['prefix_SHA256']
  assert c['ordinary_budget']==4096 and c['input_plus_output']==len(ids)+4096<=32768
  assert c['body_unchanged']and c['reverse_consumer_prompt_and_IDs_equal']and c['CPU_queue_retired']and not c['Native_reset_or_generation_executed']
  assert c['history_render_audit']['qualified']and c['history_render_audit']['prompt_tokens']==len(ids)
  small.append({k:c[k]for k in ['case_id','source_index','request_body_SHA256','prompt_tokens','prompt_bytes','prompt_SHA256','prompt_IDs_SHA256','prefix_SHA256','input_plus_output','private_path','private_SHA256']})
 shards=[]
 for replica in range(2):
  selected=[c for i,c in enumerate(q['cases'])if(i//2)%2==replica];selected_ids={c['case_id']for c in selected};schedule=[v for v in plan['comparison_schedule']if v['case_id']in selected_ids];assert len(selected)==32 and len(schedule)==64
  first=[v['window']for v in schedule[::2]];assert first.count(2)==first.count(4)==16
  shards.append(dict(replica_index=replica,cases=32,HTTP=64,W2_first_pairs=16,W4_first_pairs=16,prompt_tokens_sum=sum(c['prompt_tokens']for c in selected),shard_formula='(selected_ordinal//2)%2',all_prefill_single512chunk=all(c['prompt_tokens']<=512 for c in selected)))
 return dict(passed=True,actual_CPU=q['actual_CPU'],receipt_path=a['receipt_path'],receipt_SHA256=a['receipt_SHA256'],source_manifest_SHA256=q['source_manifest_SHA256'],actual_template_SHA256=q['template_SHA256'],tokenizer_files_SHA256=q['tokenizer_files_SHA256'],checkpoint_metadata_SHA256=q['checkpoint_metadata_SHA256'],cases=small,shards=shards,actual_encodings=64,prompt_tokens_min=min(c['prompt_tokens']for c in small),prompt_tokens_max=max(c['prompt_tokens']for c in small),prompt_tokens_sum=sum(c['prompt_tokens']for c in small),all_input_plus4096_within32768=True,actual_reverse_consumer_all64_equal=q['all_actual_reverse_consumer_equal'],all_private_prompt_ID_bytes_SHA_verified=64,all_GSM_prefill_single512chunk=True,actual_prefix='<|assistant|><think>',requested_thinkingFalse_did_not_close_prefix=True,elapsed_s=q['elapsed_s'],successful_CPU673_guard_reuse=q['successful_CPU673_guard_reuse'],archive_local_path=str(Path(path).resolve()),archive_local_SHA256=digest(Path(path).read_bytes()),archive_remote_path=outer['archive_path'],archive_remote_SHA256=outer['archive_SHA256'],GPU=0,Native=0,Model=0,actual_service_executed=False,root_or_ABI_qualified=False,scope='RealcheckpointCPUprocessor encoding and64completebody/prompt/IDs bindings. NotModelquality/performance ornewprofileHTTP/device qualification;single512chunk cannotmeasureW4multichunkgain.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=read(a.archive);out=Path(a.output);assert not out.exists();out.write_bytes((json.dumps(q,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(dict(output=str(out),SHA256=digest(out.read_bytes()),passed=q['passed'],actual_CPU=q['actual_CPU'],cases=q['actual_encodings'])))
