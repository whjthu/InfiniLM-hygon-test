"""Freeze a preselected disjoint few-shot quality set. No tokenizer or devices.

Raw sources/questions/prompts stay in the supplied private output directory.
This is an explicit custom chat few-shot protocol, not an lm-eval reproduction.
"""
import argparse,csv,hashlib,io,json,random,re,tarfile,unicodedata
from fractions import Fraction
from pathlib import Path,PurePosixPath
COUNT=128;SEED=20261104;LETTERS='ABCD';CHUNK=512;CONTEXT=32768;BUDGET=4096
sha=lambda b:hashlib.sha256(b).hexdigest()
def compact(q):return json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
def question_key(text):return sha(' '.join(unicodedata.normalize('NFC',text).split()).encode())
def item_key(row,suite):
 if suite=='gsm8k':return suite+':'+question_key(row['question'])
 return suite+':'+sha(compact(dict(question=' '.join(unicodedata.normalize('NFC',row['question']).split()),choices=[' '.join(unicodedata.normalize('NFC',v).split())for v in row['choices']])))
def checked_file(entry):
 p=Path(entry['path']);raw=p.read_bytes()
 if not re.fullmatch('[0-9a-f]{64}',entry['SHA256'])or sha(raw)!=entry['SHA256']:raise ValueError('Source bytes mismatch: '+str(p))
 if not entry.get('source_url')or not entry.get('revision'):raise ValueError('Exact authoritative origin/revision metadata required')
 return raw

def records(entry):
 raw=checked_file(entry)
 if entry['format']=='jsonl':rows=[json.loads(line)for line in raw.splitlines()if line.strip()]
 elif entry['format']=='json':rows=json.loads(raw)
 else:raise ValueError('Expected full official-source json/jsonl records')
 if type(rows)is not list:raise ValueError('Dataset must be a full record list')
 return rows

def mmlu_sources(spec):
 if 'mmlu_csv_tar'in spec:
  entry=spec['mmlu_csv_tar'];checked_file(entry);out={'dev':[],'test':[]}
  # Read selected CSV members in memory; never extract paths from an archive.
  with tarfile.open(entry['path'],'r:*')as archive:
   entries=archive.getmembers();members={}
   for member in entries:
    parts=PurePosixPath(member.name).parts
    if len(parts)!=3 or parts[0]!='data'or parts[1]not in('dev','test')or not parts[2].endswith('_'+parts[1]+'.csv'):continue
    if PurePosixPath(member.name).is_absolute()or '..'in parts or '\\'in member.name or not member.isfile()or member.issym()or member.islnk()or member.size>8*1024*1024:raise ValueError('Unsafe expected MMLU CSV member')
    if member.name in members:raise ValueError('Duplicate expected archive member')
    members[member.name]=member
   for split in out:
    names=sorted(n for n in members if n.endswith('_'+split+'.csv')and '/'+split+'/'in '/'+n)
    if len(names)!=57:raise ValueError('Require all57 subjects for '+split)
    for name in names:
     subject=Path(name).name.removesuffix('_'+split+'.csv');content=archive.extractfile(members[name]).read().decode('utf8')
     for index,row in enumerate(csv.reader(io.StringIO(content))):
      if len(row)!=6 or row[5]not in LETTERS:raise ValueError('Invalid original MMLU CSV row')
      out[split].append(dict(subject=subject,subject_row=index,question=row[0],choices=row[1:5],answer=LETTERS.index(row[5])))
  return out['dev'],out['test']
 return records(spec['mmlu_dev']),records(spec['mmlu_test'])

def collect_questions(value):
 found=[]
 if isinstance(value,dict):
  if isinstance(value.get('question'),str):found.append(value['question'])
  for k,v in value.items():
   if k not in ['source','metadata','sources']:found.extend(collect_questions(v))
 elif isinstance(value,list):
  for v in value:found.extend(collect_questions(v))
 return found

def body_for(user,suite):
 system=('Solve the math problem accurately. Explain your work and end with #### followed by the final numeric answer.'if suite=='gsm8k'else'Answer the multiple-choice question. End with Answer: followed by exactly one letter A, B, C, or D.')
 return dict(model='GLM-5.3',messages=[dict(role='system',content=system),dict(role='user',content=user)],temperature=0,max_tokens=BUDGET,stream=False,tool_choice='none',parallel_tool_calls=False,chat_template_kwargs=dict(enable_thinking=False,clear_thinking=False))

def render_mmlu(row,answered):
 choices=row['choices'];assert len(choices)==4 and all(type(v)is str for v in choices)
 text=row['question']+'\n'+'\n'.join(c+'. '+v for c,v in zip(LETTERS,choices))+'\nAnswer:'
 return text+(' '+LETTERS[row['answer']]+'\n\n'if answered else'')

def freeze(spec_path,workspace,destination):
 D=Path(__file__).resolve().parent;spec_raw=Path(spec_path).read_bytes();spec=json.loads(spec_raw)
 train=records(spec['gsm8k_train']);test=records(spec['gsm8k_test']);dev,mmlu=mmlu_sources(spec)
 if len(train)!=7473 or len(test)!=1319 or len(dev)!=285 or len(mmlu)!=14042:raise ValueError('Need full7473/1319 GSM train/test and285/14042 MMLU dev/test; no subset padding')
 dev_by={}
 for row in dev:
  if type(row.get('answer'))is not int or row['answer']not in range(4):raise ValueError('MMLU dev answer must be zero-based0..3')
  dev_by.setdefault(row['subject'],[]).append(row)
 if len(dev_by)!=57 or any(len(rows)!=5 for rows in dev_by.values()):raise ValueError('Exactly5 official dev examples for every57subject required')
 for row in mmlu:
  if row['subject']not in dev_by:raise ValueError('MMLU test subject missing dev examples')
 prior=json.loads((D/'PRIOR_INPUTS.json').read_bytes());excluded=set();evidence=[]
 for item in prior:
  raw=(Path(workspace)/item['workspace_relative_path']).read_bytes()
  if sha(raw)!=item['SHA256']:raise ValueError('Prior input bytes mismatch')
  questions=collect_questions(json.loads(raw));excluded.update(map(question_key,questions));evidence.append(dict(**item,question_records=len(questions)))
 shots=train[:8] # Predeclared train indices0..7, never chosen from test/gold.
 if len(shots)!=8:raise ValueError('Eight actual train exemplars required')
 # Random selection is fixed before target grading or prompt-length computation; labels are never a selection/filter criterion.
 selected={}
 for suite,rows in [('gsm8k',test),('mmlu',mmlu)]:
  order=list(range(len(rows)));random.Random(SEED+(0 if suite=='gsm8k' else 1)).shuffle(order)
  selected[suite]=[i for i in order if question_key(rows[i]['question'])not in excluded][:COUNT]
  if len(selected[suite])!=COUNT:raise ValueError('Insufficient disjoint full-source questions')
 cases=[];requests=[];schedule=[];public=[]
 for suite,rows in [('gsm8k',test),('mmlu',mmlu)]:
  for index in selected[suite]:
   row=rows[index];ordinal=len(cases);case_id=suite+'-r444-'+str(index)
   if suite=='gsm8k':
    if question_key(row['question'])in{question_key(s['question'])for s in shots}:raise ValueError('Selected test equals frozen train exemplar; do not resample')
    user='\n\n'.join('Question: '+s['question']+'\nAnswer: '+s['answer']for s in shots)+'\n\nQuestion: '+row['question']+'\nAnswer:'
    body=body_for(user,suite)
    # Test gold is only read after the complete prompt body has been built.
    match=re.findall(r'####\s*([-+]?[0-9][0-9,.]*(?:/[0-9]+)?)',row['answer'])
    if len(match)!=1:raise ValueError('GSM8K target final numeric gold malformed')
    gold=match[0].replace(',','');Fraction(gold)
    case=dict(id=case_id,suite=suite,source_index=index,question=row['question'],gold_numeric=gold,gold_answer=row['answer'],shot_count=8,train_shot_indices=list(range(8)))
   else:
    exemplars=dev_by[row['subject']]
    if item_key(row,suite)in{item_key(s,suite)for s in exemplars}:raise ValueError('Selected test equals dev exemplar; no resampling')
    user='The following are multiple choice questions (with answers) about '+row['subject'].replace('_',' ')+'.\n\n'+''.join(render_mmlu(s,True)for s in exemplars)+render_mmlu(row,False)
    body=body_for(user,suite)
    if type(row.get('answer'))is not int or row['answer']not in range(4):raise ValueError('Selected fixed test gold malformed; no resampling')
    case=dict(id=case_id,suite=suite,source_index=index,subject=row['subject'],subject_row=row.get('subject_row'),question=row['question'],choices=row['choices'],gold_choice=LETTERS[row['answer']],shot_count=5)
   h=sha(compact(body));cases.append(case);requests.append(dict(case_id=case_id,suite=suite,selected_ordinal=ordinal,body=body,compact_body_SHA256=h))
   arms=[2,4]if ordinal%2==0 else[4,2]
   schedule.extend(dict(index=2*ordinal+j,case_id=case_id,suite=suite,window=w,compact_body_SHA256=h)for j,w in enumerate(arms))
   public.append(dict(case_id=case_id,suite=suite,source_index=index,subject=row.get('subject'),question_text_SHA256=question_key(row['question']),benchmark_item_SHA256=item_key(row,suite),shot_count=case['shot_count'],request_body_SHA256=h,replica_pair_unit=ordinal//2))
 if len(cases)!=256 or len(schedule)!=512 or len({item_key(c,c['suite'])for c in cases})!=256:raise ValueError('Frozen count or duplicate test question violation; do not silently replace')
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
 files={'cases.json':dict(cases=cases,full_benchmark=False,custom_chat_fewshot_protocol=True),'requests-and-pairs.json':dict(requests=requests,comparison_schedule=schedule,HTTP=512,Native=512,warmups=0,ordinary_output_budget=4096)}
 hashes={}
 for name,value in files.items():
  raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();(destination/name).write_bytes(raw);hashes[name]=sha(raw)
 metadata=dict(freezer_SHA256=sha(Path(__file__).read_bytes()),prior_inputs_SHA256=sha((D/'PRIOR_INPUTS.json').read_bytes()),source_spec_SHA256=sha(spec_raw),sources={k:{a:b for a,b in v.items()if a!='path'}for k,v in spec.items()},seed=SEED,selection='Random(seed+suiteoffset).shuffle alltestindices, first128 outside exact normalized prior-question hashes; no test answer/length filtering',exclusion_evidence=evidence,shot_policy='GSMfirst8officialtrain;MMLUall5devsameSubject;fullanswersfromtrain/devonly',cases=public,files_SHA256=hashes,raw_data_stays_R_L=True,actual_checkpoint_encoding=False,actual_window_exposure='UNKNOWN until full real encoding and live delegated counters',full_benchmark=False,noninferiority_proven=False,GPU_GO=False)
 (destination/'MANIFEST.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
 return metadata
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--workspace',required=True);p.add_argument('--output',required=True);a=p.parse_args();q=freeze(a.sources,a.workspace,a.output);print(json.dumps({k:q[k]for k in ['source_spec_SHA256','files_SHA256','GPU_GO']}))
