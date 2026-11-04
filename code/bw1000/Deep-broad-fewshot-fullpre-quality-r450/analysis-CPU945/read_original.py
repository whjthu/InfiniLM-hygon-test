"""Stdlib independent input/plan/CPU-glue audit; no tokenizer or Native executes."""
import argparse,base64,gzip,hashlib,json,statistics,unicodedata
from pathlib import Path
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()

def questions(q):
 if isinstance(q,list):return [s for v in q for s in questions(v)]
 if not isinstance(q,dict):return []
 return ([q['question']]if isinstance(q.get('question'),str)else[])+[s for k,v in q.items()if k not in ['source','metadata','sources']for s in questions(v)]

def key(s):return H(' '.join(unicodedata.normalize('NFC',s).split()).encode())

def analyze(path):
 blob=Path(path).read_bytes();assert H(blob)=='87ec623677735978f2f3370f4377044492673bef34f4d1e2906000e4fc754155';raw=gzip.decompress(blob);assert H(raw)=='e6c8786d909a298d6d11e2f8531d3ce35a52448ddab112a00964ded8c2a7a872'
 a=json.loads(raw);files=a['files'];assert len(files)==277 and a['actual_Model_or_GPU']is False
 for name,f in files.items():
  b=base64.b64decode(f['base64']);assert len(b)==f['bytes']and H(b)==f['SHA256'],name
 def exact(ref):
  if ref.startswith('/refactor/'):ref='/public/home/aclsylqidf/hygon-inference-refactor-20261014/'+ref.removeprefix('/refactor/')
  return base64.b64decode(files[ref]['base64'])
 def suffix(tail):
  found=[n for n in files if n.endswith(tail)];assert len(found)==1,(tail,found);return exact(found[0])
 q=json.loads(suffix('/encoding.json'));assert H(suffix('/encoding.json'))=='04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a'
 assert q['completed']and q['passed']and q['actual_CPU']=='910407.945'and q['actual_encodings']==256 and q['source_manifest_SHA256']=='822f91e71088cf6c63828f442b956ab6d59744becdbf23433b8ddeef2769f4bb'
 assert q['actual_Deep_codec']and q['actual_public_protocol']and q['actual_prepared_FIFO']and q['matching_original_fewshot_messages']==256 and q['no_answer_or_length_reselection']
 assert not any(q[k]for k in ['GPU','Native_dlopen','weights_loaded','Model_READY','HTTP','quality_result_created','old_suites_replayed'])
 rows=a['accounting'];assert rows==[['910407.945','Deep450-broad-fewshot-CPU-v1','COMPLETED','0:0','17','a14r2n20']]
 for stage in [*q['source_observation'],q['final_modules']]:assert not stage['project_or_SDK_maps']and not stage['Torch_CUDA_initialized']and set(stage['hidden'].values())=={'-1'}
 assert q['final_modules']['Torch_imported']is True # observed import is not GPU initialization.
 pins=json.loads(suffix('/PINS.json'));cases=json.loads(exact(pins['inputs']['cases']['path']))['cases'];old=json.loads(exact(pins['inputs']['requests']['path']))['requests'];metadata=json.loads(exact(pins['inputs']['metadata']['path']))
 assert [c['id']for c in cases]==[r['case_id']for r in old]==[r['case_id']for r in q['cases']]and len(cases)==256
 assert sum(c['suite']=='gsm8k'for c in cases)==sum(c['suite']=='mmlu'for c in cases)==128
 for pin in pins['inputs'].values():assert H(exact(pin['path']))==pin['SHA256']
 assert metadata['files_SHA256']['cases.json']==pins['inputs']['cases']['SHA256']and metadata['files_SHA256']['requests-and-pairs.json']==pins['inputs']['requests']['SHA256']
 requests_raw=exact(q['requests_file']);assert H(requests_raw)==q['requests_file_SHA256']=='5b90259f33666c99e5c74f41733899aaaeaf545481faf57e259c1dd233bb420d';assets=json.loads(requests_raw);assert len(assets['requests'])==256 and len(assets['logical_plan'])==512
 logical=[];stats={};body_hashes=[];ID_hashes=[]
 for i,(case,original,e,r)in enumerate(zip(cases,old,q['cases'],assets['requests'])):
  assert case['id']==original['case_id']==e['case_id']==r['case_id']and case['shot_count']==(8 if case['suite']=='gsm8k'else 5)
  assert e['selected_ordinal']==i and e['suite']==case['suite']and e['source_index']==case['source_index']and e['source_case_SHA256']==H(C(case))
  before=original['body'];assert H(C(before))==original['compact_body_SHA256']==e['original_fewshot_body_SHA256']
  expected=dict(before,model='hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8',stream=True,text_delivery='P1',stream_options={'include_usage':True},user='Deep-r450-'+case['id'],chat_template_kwargs={'enable_thinking':False})
  assert r['body']==expected and C(r['body']['messages'])==C(before['messages'])and case['question']in r['body']['messages'][1]['content']
  private_raw=exact(e['private_path']);assert H(private_raw)==e['private_SHA256'];private=json.loads(private_raw);assert private['case_id']==case['id']and private['body']==r['body']
  ids=private['prompt_token_ids'];assert ids and all(type(v)is int and 0<=v<129280 for v in ids)and len(ids)==e['prompt_tokens']and len(ids)+4096<=32768
  assert H(C(ids))==e['prompt_IDs_SHA256']and H(private['prompt'].encode())==e['prompt_SHA256']and H(C(r['body']))==r['request_body_SHA256']==e['request_body_SHA256']
  chunks=[min(1024,len(ids)-s)for s in range(0,len(ids),1024)];assert chunks==e['prefill_chunks']and e['T1_prefill_calls']==sum(v==1 for v in chunks)and e['prefill_chunk']==1024
  assert e['output_budget']==4096 and e['input_plus_output']==len(ids)+4096 and e['text_delivery']=='P1'and e['actual_thinking']is False and e['CPU_queue_retired']and not e['Native_generation_or_reset']
  h=e['history_render_audit'];assert h['qualified']and h['prequeue_bound']and h['same_prepared_encoder_reproduces_prompt_and_IDs']and h['pending_chat_prefix_closed']and not h['pending_thinking_prefix_open']and not h['markers_sent_to_Native']and h['Native_prompt_token_ids_SHA256']==H(C(ids))and h['output_budget']==4096
  assert h['content_one_contribution_each_and_ordered']and h['all_content_contribution_slices_equal']
  for mode in ([0,1]if i%2==0 else[1,0]):logical.append(dict(logical_index=len(logical),case_id=case['id'],case_ordinal=i,mode=mode,request_body_SHA256=e['request_body_SHA256'],prompt_IDs_SHA256=e['prompt_IDs_SHA256'],prompt_tokens=len(ids),output_budget=4096))
  body_hashes.append(e['request_body_SHA256']);ID_hashes.append(e['prompt_IDs_SHA256'])
 assert logical==assets['logical_plan']and sum(r['mode']==0 for r in logical)==sum(r['mode']==1 for r in logical)==256
 overlaps=[]
 for pin,observed in zip(pins['known_prior_question_files'],q['known_prior_overlap']):
  prior=exact(pin['path']);assert H(prior)==pin['SHA256'];known={key(s)for s in questions(json.loads(prior))};matched=[c['id']for c in cases if key(c['question'])in known]
  assert observed==dict(SHA256=pin['SHA256'],aliases=pin['aliases'],known_unique_questions=len(known),matching_case_ids=matched,overlap=len(matched));overlaps.append(len(matched))
 assert len(overlaps)==5 and sum(len(v['aliases'])for v in pins['known_prior_question_files'])==44
 consumer=q['changed_finite_plan_policy_consumer'];assert consumer['passed']and consumer['recording_only']and not consumer['actual_Native_GPU_or_quality']and not consumer['old938_cases_replayed']and consumer['recording_Native_calls']==6
 for k in ['actual_new_wide_policy','actual_original_generate_and_reset','actual_encoded_first_case_two_arms','finite512_plan_checked','zero_T1_early_EOS_kept','one_token_prefill_counted','extra_request_and_wrong_owner_rejected','actual_new_client_main_CLI_and_real_body_encoding_plan_file_handoff','actual_SSE_reconstruction','wrong_parse_continues_both_arms','cap_matching_answer_rejected','global_denominator256_perarm_kept']:assert consumer[k]is True,k
 fake=json.loads(suffix('/changed-client-recording/result.json'));assert fake['completed']and fake['execution_passed']and fake['completed_HTTP']==2 and fake['global_planned_HTTP']==512
 assert fake['rows'][0]['correct']and not fake['rows'][1]['correct']and fake['rows'][1]['parse_failure']and all(fake['summary']['pooled']['arms'][str(mode)]['planned']==256 for mode in [0,1])
 for suite in ['gsm8k','mmlu']:
  selected=[r for r in q['cases']if r['suite']==suite];counts={}
  for r in selected:counts[str(len(r['prefill_chunks']))]=counts.get(str(len(r['prefill_chunks'])),0)+1
  stats[suite]=dict(count=len(selected),prompt_min=min(r['prompt_tokens']for r in selected),prompt_median=statistics.median(r['prompt_tokens']for r in selected),prompt_max=max(r['prompt_tokens']for r in selected),prefill_chunk_count_histogram=counts,T1_prefill_cases=sum(r['T1_prefill_calls']>0 for r in selected))
 return dict(passed=True,actual_CPU='910407.945',Slurm='COMPLETED0/17s',raw_receipt_SHA256=H(suffix('/encoding.json')),archive_SHA256=H(blob),verified_raw_files=277,source_manifest_SHA256=q['source_manifest_SHA256'],actual_Deep_encodings=256,finite_logical_requests=512,full_original_fewshot_messages_retained=256,per_suite=stats,known_prior_unique_files=5,known_prior_aliases=44,overlap_counts=overlaps,overlap_scope='Exact known prior inventory only; not universal unseen status',requests_private_SHA256=H(requests_raw),logical_plan_compact_SHA256=H(C(logical)),actual_maps_empty=True,Torch_import_observed=True,Torch_CUDA_initialized=False,hidden_all_minus1=True,changed_policy_client_consumer_recording_only=True,new_actual_Native_Runtime_GPU_Model_or_quality=False,GPU_GO=False,Model_GO=False,root_must_freeze_new_time_and_shards=True,CPU_no_quality_or_performance_claim=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output');a=p.parse_args();q=analyze(a.archive);text=json.dumps(q,sort_keys=True,indent=2)+'\n'
 if a.output:Path(a.output).write_bytes(text.encode())
 print(text,end='')
