"""Stdlib audit of CPU918 originals and exact public-source few-shot bodies.

No tokenizer, SDK, Native, dataset Python code or jobs execute here.
"""
from pathlib import Path,PurePosixPath
import argparse,base64,collections,csv,gzip,hashlib,io,json,random,re,statistics,tarfile,unicodedata
from fractions import Fraction
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014';B=R+'/candidates/GLM-W4-fewshot-quality-r444-CPU-v1';D=R+'/incoming/GLM-W4-fewshot-quality-r444'
ARCHIVE='71e514ed04967b1b67c19f6e19decc92a3c0d9d09450eff659f610efa3005133';MF='56254b9f9b703f61339f1db912dbe9650d93913ca83b298cf1c2aa6f19127f7f';ENC='1e083b8241422ed87511d83fa8585bfc19eb356e539d42acfa80b4c10cc8351c';RESULT='c9d5b70d658b63852c0c80bdd867c19d5ef36fd107175bdb183cc3e0b2647b2e'
sha=lambda b:hashlib.sha256(b).hexdigest()
def compact(v):return json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()
def key(v):return sha(' '.join(unicodedata.normalize('NFC',v).split()).encode())
def questions(v):
 out=[]
 if isinstance(v,dict):
  if isinstance(v.get('question'),str):out.append(v['question'])
  for k,q in v.items():
   if k not in ['source','metadata','sources']:out+=questions(q)
 elif isinstance(v,list):
  for q in v:out+=questions(q)
 return out

def audit(archive,sources,workspace):
 blob=Path(archive).read_bytes();assert sha(blob)==ARCHIVE;a=json.loads(gzip.decompress(blob));assert len(a['files'])==283 and a['new_Slurm']==a['GPU']==a['Model']==0 and not a['Native_dlopen']
 files={}
 for p,m in a['files'].items():
  raw=base64.b64decode(m['base64'],validate=True);assert len(raw)==m['bytes']and sha(raw)==m['SHA256'];files[p]=raw
 def host(p):assert p.startswith('/refactor/');return R+p[len('/refactor'):]
 assert '910407.918|GLM-r444-fewshot-checkpoint-CPU-v1|COMPLETED|0:0|00:00:23|a14r2n20'in a['accounting']
 assert sha(files[B+'/CPU-result.json'])==RESULT and sha(files[B+'/encoding.json'])==ENC
 q=json.loads(files[B+'/CPU-result.json']);e=json.loads(files[B+'/encoding.json']);assert q['passed']and e['passed']and q['actual_CPU']==e['actual_CPU']=='910407.918'and q['actual_unique_encodings']==256
 assert q['window_consumer']['case_count']==17 and q['score_consumer']['checks']==11 and not q['whole_quality_client_main_executed']and not q['project_and_SDK_library_map_observation']and not q['Torch_CUDA_initialized']
 assert e['actual_checkpoint_CPU_processor']and not e['Torch_CUDA_initialized']and not e['Native_constructed']and not e['weights_loaded']and not e['actual_service_executed']
 assert e['template_SHA256']=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
 assert sha(files[D+'/SOURCE_MANIFEST.json'])==MF;manifest=json.loads(files[D+'/SOURCE_MANIFEST.json']);assert len(manifest['files_SHA256'])==19
 for n,h in manifest['files_SHA256'].items():assert sha(files[D+'/'+n])==h,n
 fm=json.loads(files[D+'/fixtures-r1/MANIFEST.json']);assert sha(files[D+'/fixtures-r1/MANIFEST.json'])=='6fb1ec5ec4299e415575e7c4feb66d36fc25d05cda0275e0a131d1a5b77eee3c'
 for n,h in fm['files_SHA256'].items():assert sha(files[D+'/fixtures-r1/'+n])==h
 dataset=json.loads(files[D+'/fixtures-r1/cases.json']);plan=json.loads(files[D+'/fixtures-r1/requests-and-pairs.json']);cases=dataset['cases'];requests=plan['requests'];schedule=plan['comparison_schedule'];assert len(cases)==len(requests)==256 and len(schedule)==512 and plan['warmups']==0
 assert collections.Counter(c['suite']for c in cases)=={'gsm8k':128,'mmlu':128};assert[c['id']for c in cases]==[r['case_id']for r in requests]==[c['case_id']for c in e['cases']]
 # Rebuild actual canonical source rows using only stdlib; never import the freezer.
 spec_raw=Path(sources).read_bytes();assert sha(spec_raw)==fm['source_spec_SHA256'];spec=json.loads(spec_raw)
 def records(entry):
  raw=Path(entry['path']).read_bytes();assert sha(raw)==entry['SHA256'];return[json.loads(line)for line in raw.splitlines()if line.strip()]
 train=records(spec['gsm8k_train']);test=records(spec['gsm8k_test']);assert len(train)==7473 and len(test)==1319
 entry=spec['mmlu_csv_tar'];archivefile=Path(entry['path']);assert sha(archivefile.read_bytes())=='bec563ba4bac1d6aaf04141cd7d1605d7a5ca833e38f994051e818489592989b';dev={};mmlu=[]
 with tarfile.open(archivefile,'r:*')as tar:
  members=sorted(tar.getmembers(),key=lambda m:m.name)
  for m in members:
   parts=PurePosixPath(m.name).parts
   if len(parts)!=3 or parts[0]!='data'or parts[1]not in ['dev','test']or not parts[2].endswith('_'+parts[1]+'.csv'):continue
   assert m.isfile()and not m.issym()and not m.islnk()and not PurePosixPath(m.name).is_absolute()and '..'not in parts
   subject=parts[2].removesuffix('_'+parts[1]+'.csv');rows=[]
   for i,row in enumerate(csv.reader(io.StringIO(tar.extractfile(m).read().decode('utf8')))):
    assert len(row)==6 and row[5]in'ABCD';rows.append(dict(subject=subject,subject_row=i,question=row[0],choices=row[1:5],answer='ABCD'.index(row[5])))
   if parts[1]=='dev':assert subject not in dev;dev[subject]=rows
   else:mmlu+=rows
 assert len(dev)==57 and all(len(v)==5 for v in dev.values())and len(mmlu)==14042
 prior=json.loads(files[D+'/PRIOR_INPUTS.json']);excluded=set()
 for p in prior:
  raw=(Path(workspace)/p['workspace_relative_path']).read_bytes();assert sha(raw)==p['SHA256'];excluded.update(map(key,questions(json.loads(raw))))
 selected={}
 for suite,rows,seed in [('gsm8k',test,20261104),('mmlu',mmlu,20261105)]:
  order=list(range(len(rows)));random.Random(seed).shuffle(order);selected[suite]=[i for i in order if key(rows[i]['question'])not in excluded][:128]
  assert [c['source_index']for c in cases if c['suite']==suite]==selected[suite]
 def choice(r,answer):return r['question']+'\n'+'\n'.join(c+'. '+v for c,v in zip('ABCD',r['choices']))+'\nAnswer:'+(' '+'ABCD'[r['answer']]+'\n\n'if answer else'')
 by_suite={};ids_seen=set();exposure=[]
 for ordinal,(case,request,encoded)in enumerate(zip(cases,requests,e['cases'])):
  body=request['body'];assert body['max_tokens']==4096 and body['temperature']==0 and body['stream']is False and body['tool_choice']=='none'and len(body['messages'])==2
  if case['suite']=='gsm8k':
   row=test[case['source_index']];gold=re.findall(r'####\s*([-+]?[0-9][0-9,.]*(?:/[0-9]+)?)',row['answer']);assert len(gold)==1 and case['gold_answer']==row['answer']and Fraction(case['gold_numeric'])==Fraction(gold[0].replace(',',''));user='\n\n'.join('Question: '+r['question']+'\nAnswer: '+r['answer']for r in train[:8])+'\n\nQuestion: '+row['question']+'\nAnswer:';assert case['train_shot_indices']==list(range(8))and case['shot_count']==8
  else:
   row=mmlu[case['source_index']];assert case['choices']==row['choices']and case['gold_choice']=='ABCD'[row['answer']];user='The following are multiple choice questions (with answers) about '+row['subject'].replace('_',' ')+'.\n\n'+''.join(choice(r,True)for r in dev[row['subject']])+choice(row,False);assert case['shot_count']==5 and case['subject']==row['subject']
  assert body['messages'][1]['content']==user and case['question']==row['question']and key(row['question'])not in excluded
  assert request['compact_body_SHA256']==encoded['request_body_SHA256']==sha(compact(body))
  private_raw=files[host(encoded['private_path'])];assert sha(private_raw)==encoded['private_SHA256'];private=json.loads(private_raw);prompt=private['prompt'];ids=private['prompt_token_ids'];assert len(ids)==encoded['prompt_tokens']and all(type(i)is int and 0<=i<154880 for i in ids)
  assert user in prompt and sha(prompt.encode())==encoded['prompt_SHA256']and len(prompt.encode())==encoded['prompt_bytes']and sha(json.dumps(ids,separators=(',',':')).encode())==encoded['prompt_IDs_SHA256']
  assert len(ids)+4096<=32768 and encoded['body_unchanged']and encoded['reverse_consumer_prompt_and_IDs_equal']and encoded['CPU_queue_retired']and not encoded['Native_reset_or_generation_executed']
  chunks=[min(512,len(ids)-i)for i in range(0,len(ids),512)];assert encoded['prefill_chunks']==len(chunks)and encoded['prefill_chunk_tokens']==chunks and encoded['planned_pair_control_exposed']==(len(chunks)>2)and not encoded['actual_live_window_counter_verified']
  for w in[2,4]:assert encoded['expected_control_groups'][str(w)]==[min(w,len(chunks)-i)for i in range(0,len(chunks),w)]
  assert len({r['window']for r in schedule if r['case_id']==case['id']})==2 and sum(r['case_id']==case['id']for r in schedule)==2
  assert all(r['compact_body_SHA256']==encoded['request_body_SHA256']for r in schedule if r['case_id']==case['id'])
  ids_seen.add(case['id']);by_suite.setdefault(case['suite'],[]).append(encoded)
  if len(chunks)>2:exposure.append(case['id'])
 assert len(ids_seen)==256 and len(exposure)==160 and e['planned_multigroup_by_suite']=={'gsm8k':128,'mmlu':32}
 stats={}
 for suite,rows in by_suite.items():
  lengths=[r['prompt_tokens']for r in rows];stats[suite]=dict(questions=len(rows),min=min(lengths),median=statistics.median(lengths),mean=statistics.mean(lengths),max=max(lengths),sum=sum(lengths),chunk_hist=dict(collections.Counter(r['prefill_chunks']for r in rows)),control_exposed=sum(r['prefill_chunks']>2 for r in rows),full_four_chunk_group_possible=sum(r['prefill_chunks']>=4 for r in rows))
 return dict(passed=True,actual_CPU='910407.918 COMPLETED0/23s',archive_SHA256=sha(blob),raw_files=283,source_members=19,all256_actual_prompts_and_IDs_rehashed=True,authoritative_sources_and_selected_questions_reconstructed=True,all256_gold_verified_against_actual_source=True,prior_files_SHA_checked=len(prior),full_fewshot_bodies_reconstructed=True,actual_unique_encodings=256,paired_HTTP_planned=512,actual_GPU_HTTP=0,source_manifest_SHA256=MF,CPU_result_SHA256=RESULT,encoding_SHA256=ENC,template_SHA256=e['template_SHA256'],suite_stats=stats,planned_control_exposed=160,unexposed_MMLU_kept=96,qualified_live_W4_counter=False,whole_new_HTTP_main_qualified=False,Native=False,GPU=False,Model=False,noninferiority_proven=False,full_benchmark=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--sources',required=True);p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive,a.sources,a.workspace);Path(a.output).write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q,indent=2))
if __name__=='__main__':main()
