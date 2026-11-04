"""Stdlib, offline, exact original1417 six-cohort/JSONL audit. No device/imports."""
from pathlib import Path
import argparse,base64,collections,gzip,hashlib,json,math,re,struct
from fractions import Fraction
H=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014'
D=R+'/incoming/Deep-a314-fixed-cohort-C4-r458'
O=R+'/results/Deep458-fixed-cohort-private/fixed-c4-20261104-a'
LABEL='Deep-r458-fixed-C4-fixed-c4-20261104-a'
NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
MODES=['serial','cohort','serial','cohort','cohort','serial']
def close(a,b):assert math.isfinite(a)and math.isfinite(b)and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-7),(a,b)
def quantile(a,p):
 a=sorted(a);k=(len(a)-1)*p;i=int(k);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(k-i)
def numeric(s):
 m=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',s)
 try:return Fraction(m[-1].replace(',',''))if m else None
 except(ValueError,ZeroDivisionError):return None
def choice(s):
 m=re.findall(r'(?:^|\n)\s*(?:\*\*)?(?:Answer|Final answer)\s*:\s*[\[(]?([ABCD])[\])]?(?:\*\*)?(?:[.\s]|$)',s,re.I)
 if m:return m[-1].upper()
 m=re.fullmatch(r'\s*[\[(]?([ABCD])[\])]?\s*[.]?\s*',s,re.I);return m[1].upper()if m else None
def affinity(q):
 assert q['verified']and q['width']==4;owners=q['actual_owner_report']['owners'];assert len(owners)==8
 assert sorted(v['device']for v in owners)==list(range(8));allcpus=[]
 for v in owners:
  assert v['changes']==1;mask=list(map(int,v['verified_mask'].split(',')));assert len(set(mask))==len(mask)==4;allcpus+=mask
 assert len(set(allcpus))==32
 return[(v['device'],v['verified_mask'])for v in owners]
def audit(path):
 raw=Path(path).read_bytes();archive=json.loads(raw);files={}
 for v in archive['files']:
  b=gzip.decompress(base64.b64decode(v['gzip_base64']));assert len(b)==v['bytes']and H(b)==v['SHA256']
  assert v['path']not in files;files[v['path']]=b
 def load(p):return json.loads(files[p])
 for root,name,h in[(D,'SOURCE_MANIFEST.json','4c782ade4e6f28c71e210de3344bf1630533dfdb39481b12a144464dd9c4d474'),(D+'/deployment-v1','SOURCE_MANIFEST.json','148373a956f627478f989041f49fff54238cbf2ae862acdb70abdb0be2e57eec'),(D+'/deployment-v1','GPU_MANIFEST.json','110bcc510bc77459815ad4356ef39fa04613945c5e5ac0054b37a92897d0d335')]:
  assert H(files[root+'/'+name])==h
  for n,digest in load(root+'/'+name)['files_SHA256'].items():assert H(files[root+'/'+n])==digest,n
 w=load(R+'/results/'+LABEL+'-watch.json');r=load(O+'/entry-report.json');inputs=load(O+'/four-inputs-private.json')
 assert H(files[O+'/entry-report.json'])==w['entry_report_SHA256']and H(files[O+'/four-inputs-private.json'])==r['inputs_SHA256']
 assert w['completed']and w['passed']and not w['forced_cleanup']and w['actual_GPU']=='910374.1417'and w['CPU_keeper']=='910407.973'
 epoch=w['original_GPU_start_unix'];assert epoch==r['gpu_start_epoch']==1793770039.5510223
 close(w['GPU_deadline_unix']-epoch,5400);close(w['keeper_deadline_unix']-epoch,5700)
 assert r['completed']and r['passed']and r['actual_step']=='910374.1417'and r['HTTP']==0 and r['logical_requests']==24 and r['completed_cohorts']==6
 assert r['Native_SHA256']==NATIVE and r['actual_rows']==4 and r['capacity_per_row']==32768 and r['loaded_key_count']==67569
 assert all(r[k]for k in['normal_Native_shutdown','engine_worker_shutdown_completed','observed_all8_threads_exited','all_cohorts_completed_normal'])and r['infrastructure_error']is None
 pp=r['precision_policy'];assert pp['policy']==pp['status']==pp['FP32_expected_delta_per_owner']==0 and pp['provider']['SHA256']==NATIVE and not pp['F64_allowed']
 assert set(r['actual_providers_before_weights'])=={'create','load','finish','forward','reset','destroy'}and set(r['actual_providers_before_weights'].values())=={pp['provider']['path']}
 masks=affinity(r['owner_affinity_before_weight_load']);assert masks==affinity(r['owner_affinity_after_weight_load'])==affinity(r['owner_affinity_after_cohorts'])
 tids=r['owner_thread_observation'];assert tids['status']=='OBSERVED'and len({v['TID']for v in tids['owners']})==8
 blas=r['blas_policy_before_destroy'];assert blas['failures']==0 and all(v['created']==v['verified']for v in blas['owners'])
 assert len({v['tid']for v in blas['owners']if'libinfiniops.so'in v['caller']})==8
 for key in['GPU_memory_after_create','GPU_memory_after_load','memory_after_cohorts']:
  assert len(r[key])==8 and all(f>=16*1024**3 and t==68702699520 for f,t in r[key])
 for phase in['fresh_before','post']:
  q=w[phase];assert q['node']=='a02r4n05'and q['CPU_only']and not q['GPU_initialized']and q['KFD_available']
  assert not q['kfd_pids']and not q['matching_component_processes']and not q['expected_PIDs_still_exist']and q['ports_bindable']=={'43882':True}and len(set(q['physical_PCI']))==8
 assert w['post']['actual_step']=='910374.1418'
 accounting=archive['accounting'];assert accounting['rc']==0
 for step in['910374.1416','910374.1417','910374.1418','910407.973']:
  rows=[x.split('|')for x in accounting['rows']if x.startswith(step+'|')];assert len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0']
  if step=='910374.1417':assert int(rows[0][4])<5400
  if step=='910407.973':assert int(rows[0][4])<5700
 encpath=R+'/candidates/Deep-broad-fewshot-fullpre-quality-r450-CPU-v1/encoding.json';assert H(files[encpath])=='04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a';enc=load(encpath)
 goldpath=R+'/incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/cases.json';assert H(files[goldpath])=='c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38';gold={v['id']:v for v in load(goldpath)['cases']}
 assert len(inputs)==4
 for row,(record,ordinal,promptlen)in enumerate(zip(inputs,[0,1,128,129],[1242,1305,538,1050])):
  meta=enc['cases'][ordinal];private=load(R+meta['private_path'].removeprefix('/refactor'))
  assert H(files[R+meta['private_path'].removeprefix('/refactor')])==meta['private_SHA256']
  assert record['row']==row and record['ordinal']==ordinal and record['case_id']==meta['case_id']==private['case_id']
  assert record['body']==private['body']and record['gold']==gold[record['case_id']]and record['prompt_token_ids']==private['prompt_token_ids']
  assert len(record['prompt_token_ids'])==promptlen and H(compact(record['prompt_token_ids']))==meta['prompt_IDs_SHA256']==record['prompt_IDs_SHA256']
  assert H(compact(record['body']))==record['body_SHA256']==meta['request_body_SHA256']and record['max_new_tokens']==4096 and promptlen+4096<=32768
 journal=files[O+'/calls-and-events-private.jsonl'];assert H(journal)==r['journal']['SHA256']and len(journal)==r['journal']['bytes']
 events=[json.loads(x)for x in journal.splitlines()];assert len(events)==r['journal']['writes']
 grouped={};current=None;pending=None;returns=[];allcounts=[]
 for e in events:
  if e['event']=='cohort_begin':
   current=e['ordinal'];assert current not in grouped and e['mode']==MODES[current]and e['warm']==(current<2);grouped[current]=dict(attempts=[],returns=[],callers=[],tokens=[],boundaries=[])
  else:
   assert current is not None;g=grouped[current]
   if e['event']=='CAPI_attempt':assert pending is None;pending=e;g['attempts'].append(e)
   elif e['event']=='CAPI_return':
    assert pending is not None and all(e[k]==v for k,v in pending.items()if k!='event');pending=None;g['returns'].append(e);returns.append(e)
   elif e['event']=='caller_return':g['callers'].append(e)
   elif e['event']=='token_emitted':g['tokens'].append(e)
   elif e['event']=='cohort_boundary':g['boundaries'].append(e)
   else:raise AssertionError('Foreign journal event')
 assert pending is None and sorted(grouped)==list(range(6))and len(returns)==r['attempted_CAPI']==r['completed_CAPI']==1821
 previous=[dict(FP32=0,HC=[0]*3,c4_tile=[0]*2,layout=[0]*4,tail=[0]*5)for _ in range(8)];activecounts=collections.Counter();CAPIwall=0.
 for index,e in enumerate(returns):
  assert e['index']==index and e['rc']==e['BF16_nonfinite_count']==0 and e['input_readonly']and e['logits_dtype']=='BF16'and e['logits_shape']==[e['count'],129280]
  close(e['CAPI_seconds'],e['CAPI_return_monotonic']-e['CAPI_start_monotonic']);CAPIwall+=e['CAPI_seconds'];activecounts[e['count']]+=1
  assert len(e['sampled_IDs'])==e['count']and all(type(x)is int and 0<=x<129280 for x in e['sampled_IDs'])
  assert len(e['GPU_free_total_after_CAPI'])==8 and all(f>=16*1024**3 and t==68702699520 for f,t in e['GPU_free_total_after_CAPI'])
  assert len(e['all8_counters'])==8
  for d,c in enumerate(e['all8_counters']):
   assert c['device']==d;before=previous[d];kind=0 if e['tokens']==512 else 2;target=e['count']==1 and e['tokens']==512 and e['past']==[0]
   expected=dict(FP32=0,HC=[86 if k==kind else 0 for k in range(3)],c4_tile=[0,0],layout=[0,0,0,43],tail=[0,0,0,43*int(target),43*int(not target)])
   for key,value in expected.items():
    delta=c[key]-before[key]if type(value)is int else[x-y for x,y in zip(c[key],before[key])];assert delta==value,(index,d,key,delta,value)
   previous[d]=c
 totals=[]
 for ordinal in range(6):
  tpath=O+'/cohort-%d-private.json'%ordinal;trial=load(tpath);descriptor=r['fixed_cohort_trials'][ordinal]
  assert H(files[tpath])==descriptor['SHA256']and trial['ordinal']==ordinal and trial['mode']==MODES[ordinal]and trial['warm']==(ordinal<2)
  assert trial['completed']and not trial['failed']and trial['normal']and trial['Native_reset_before']and trial['Native_reset_after']
  close(trial['whole_cohort_seconds'],trial['completion_monotonic']-trial['all_arrival_monotonic']);close(descriptor['whole_cohort_seconds'],trial['whole_cohort_seconds'])
  g=grouped[ordinal];assert len(g['returns'])==len(g['attempts'])==len(g['callers'])==len(trial['calls'])
  assert len(g['boundaries'])==1 and g['boundaries'][0]['phase']=='terminal'and g['boundaries'][0]['normal']and g['boundaries'][0]['reset_after']and not g['boundaries'][0]['failed']
  past=[0]*4;generated=[[]for _ in range(4)];tokenlines=[[]for _ in range(4)];finishes=[False]*4;seen_prefill=[];counts=collections.Counter()
  for call,ret,caller in zip(trial['calls'],g['returns'],g['callers']):
   assert all(caller[k]==v for k,v in call.items()if k!='finite')and call['rc']==0 and call['finite']
   rows=call['rows'];widths=call['widths'];assert rows==ret['rows']and call['past']==ret['past']==[past[i]for i in rows]and len(rows)==len(set(rows))==ret['count']
   assert all(not finishes[i]for i in rows)and call['offsets']==ret['offsets']and call['output_rows']==len(rows)
   close(call['seconds'],call['return_monotonic']-call['start_monotonic']);assert call['start_monotonic']<=ret['CAPI_start_monotonic']<=ret['CAPI_return_monotonic']<=call['return_monotonic']
   seqs=[];positions=[]
   for i,width in zip(rows,widths):
    assert width>0 and past[i]+width<=32768
    seq=inputs[i]['prompt_token_ids'][past[i]:past[i]+width]if call['kind']=='prefill'else[generated[i][-1]]
    assert len(seq)==width;seqs+=seq;positions+=list(range(past[i],past[i]+width))
   if call['kind']=='prefill':assert len(rows)==1 and max(widths)<=1024;seen_prefill.append(rows[0])
   else:
    assert widths==[1]*len(rows)and all(generated[i]for i in rows)
    if MODES[ordinal]=='cohort':assert rows==[i for i in range(4)if not finishes[i]]and all(generated)
    else:assert len(rows)==1
   if MODES[ordinal]=='serial':assert rows==[next(i for i in range(4)if not finishes[i])]
   assert H(struct.pack('<'+'q'*len(seqs),*seqs))==ret['IDs_SHA256']and H(struct.pack('<'+'q'*len(positions),*positions))==ret['positions_SHA256']
   assert ret['tokens']==len(seqs)and ret['total']==[p+w for p,w in zip(ret['past'],widths)]
   assert ret['position_endpoints']==[positions[0],positions[-1]]
   if call['kind']=='decode':assert ret['decode_input_IDs']==seqs;counts[len(rows)]+=1
   for j,i in enumerate(rows):
    past[i]+=widths[j]
    if call['kind']=='decode'or past[i]==len(inputs[i]['prompt_token_ids']):
     tok=ret['sampled_IDs'][j];generated[i].append(tok);tokenlines[i].append(dict(token=tok,monotonic=call['return_monotonic'],visible=tok!=1));finishes[i]=tok==1 or len(generated[i])==4096
  assert seen_prefill==[0,0,1,1,2,3,3]and all(finishes)
  quality=[];metrics=[]
  for i,(state,q)in enumerate(zip(trial['states'],trial['quality'])):
   assert state['row']==i and state['case_id']==inputs[i]['case_id']==q['case_id']and state['past']==past[i]and state['generated_ids']==generated[i]and state['token_events']==tokenlines[i]
   actualevents=[{k:e[k]for k in['token','monotonic','visible']}for e in g['tokens']if e['row']==i];assert actualevents==tokenlines[i]
   assert state['finish']=='EOS'and generated[i][-1]==1 and 1 not in generated[i][:-1]and len(generated[i])<=4096
   times=[e['monotonic']for e in tokenlines[i]if e['visible']];assert all(b>a for a,b in zip(times,times[1:]));itl=[b-a for a,b in zip(times,times[1:])]
   m=q['metrics'];assert m['visible_tokens']==len(times)and m['tokens_including_EOS']==len(generated[i])and m['natural_EOS']and not m['cap']
   for k,value in [('first_visible_monotonic',times[0]),('last_visible_monotonic',times[-1]),('TTFT_s',times[0]-trial['all_arrival_monotonic']),('decode_tok_s',(len(times)-1)/(times[-1]-times[0])),('ITL_P95',quantile(itl,.95)),('ITL_P99',quantile(itl,.99))]:close(m[k],value)
   assert m['ITL']==itl
   # All actual outputs on this fixed run are plain text: parser content is
   # exactly the retained rawdecoded text, no hidden reasoning/tool strip.
   assert q['parsed']==dict(content=q['raw_decoded'],reasoning_content='',tool_calls=[])
   gold=inputs[i]['gold'];ans=numeric(q['raw_decoded'])if gold['suite']=='gsm8k'else choice(q['raw_decoded']);correct=ans is not None and(ans==Fraction(gold['gold_numeric'])if gold['suite']=='gsm8k'else ans==gold['gold_choice'])
   s=q['score'];assert s['parsed_answer']==(str(ans)if ans is not None else None)and s['parse_failure']==(ans is None)and s['answer_matches_gold']==s['correct']==correct and s['natural_EOS']and not s['cap_hit']and not s['unexpected_tool_intent']
   quality.append(dict(case_id=q['case_id'],suite=gold['suite'],correct=correct,parsed_answer=s['parsed_answer'],natural_EOS=True,cap=False,parse_failure=ans is None))
   metrics.append(dict(row=i,TTFT_s=m['TTFT_s'],decode_tok_s=m['decode_tok_s'],ITL_P95=m['ITL_P95'],ITL_P99=m['ITL_P99'],visible_tokens=len(times),tokens_including_EOS=len(generated[i])))
  useful=sum(len(v)-1 for v in generated);close(trial['engine_emitter_tok_s'],useful/trial['whole_cohort_seconds']);close(trial['engine_emitter_tok_s_per_card'],useful/trial['whole_cohort_seconds']/8)
  assert useful==trial['useful_tokens']and len(trial['quality'])==len(trial['states'])==4
  totals.append(dict(ordinal=ordinal,mode=MODES[ordinal],warm=ordinal<2,whole_cohort_seconds=trial['whole_cohort_seconds'],useful_tokens=useful,tokens_including_EOS=useful+4,engine_emitter_tok_s=useful/trial['whole_cohort_seconds'],tok_s_per_card=useful/trial['whole_cohort_seconds']/8,correct=sum(v['correct']for v in quality),quality=quality,metrics=metrics,CAPI_calls=len(g['returns']),decode_active_counts=dict(counts)))
 assert r['HC_dispatch_checked_forwards']==r['layout_dispatch_checked_forwards']==sum(r['dispatch_checks_r4842'].values())==1821
 pairs=[]
 for a,b in[(2,3),(5,4)]:
  off,on=totals[a],totals[b];pairs.append(dict(serial=a,cohort=b,whole_cohort_speedup=off['whole_cohort_seconds']/on['whole_cohort_seconds'],useful_throughput_speedup=on['engine_emitter_tok_s']/off['engine_emitter_tok_s'],serial_visible=off['useful_tokens'],cohort_visible=on['useful_tokens'],same_output_length=off['useful_tokens']==on['useful_tokens']))
 drift=abs(totals[5]['whole_cohort_seconds']/totals[2]['whole_cohort_seconds']-1)
 decodecounts=collections.Counter()
 for t in totals:decodecounts.update(t['decode_active_counts'])
 return dict(passed=True,archive_SHA256=H(raw),files=len(files),decoded_bytes=sum(map(len,files.values())),actual_GPU='910374.1417',CPU_keeper='910407.973',post='910374.1418',normal8_destroy_and_thread_exit=True,post_clean=True,source_policy0_all8_FP32_delta0=True,HC0_all86_per_forward_all8_verified=True,logical_requests=24,CAPI_forwards=1821,CAPI_wall_seconds=CAPIwall,CAPI_output_row_counts_actual=dict(activecounts),decode_active_rows_actual=dict(decodecounts),cohorts=totals,formal_pairs=pairs,serial_endpoint_wall_change=drift,both_wholewall_pairs_exceed_serial_endpoint_change=all(p['whole_cohort_speedup']-1>drift for p in pairs),formal_quality_correct=sum(t['correct']for t in totals[2:]),formal_quality_planned=16,formal_natural_EOS=16,loading_seconds=r['loading_seconds'],min_free_after_call_bytes=min(min(f for f,t in e['GPU_free_total_after_CAPI'])for e in returns),GPU_memory_after_create=r['GPU_memory_after_create'],GPU_memory_after_load=r['GPU_memory_after_load'],sampled_min_free_scope='Driver snapshots after every full CAPI, not inside-kernel peak.',clock_scope='Same-process engine-emitter complete cohort includes sequentialprefill/reset/copies/audit/drain; no HTTP or online SLO.',limits=['Four exact cases repeated, not broadquality or Modelnoninferiority.','Same rows4 engine in both arms; not standardpublic rows1 baseline.','Actual token lengths differ; wholewall and useful throughput both shown.','Retained generatedIDs/logit SHA and plain decoded content validated; independent tokenizer decoding not performed.','Recorded scalar journal/host memory snapshots add common driver overhead; no hypothetical kernel-only extrapolation.','No PD/export/import/online scheduler was tested.'])
def main():
 p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',required=True);a=p.parse_args();q=audit(a.archive);Path(a.output).write_text(json.dumps(q,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in q.items()if k not in['cohorts','GPU_memory_after_create','GPU_memory_after_load']}))
if __name__=='__main__':main()
