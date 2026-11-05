"""Stdlib offline audit of original public64 fixed-C4 run; never imports runtime."""
from pathlib import Path
import argparse,base64,collections,gzip,hashlib,json,math,re,statistics
from fractions import Fraction
SEAL='408f8b93c5e4ffbe868ff3b954b327900598c5a5ed02cd014a426349a225ed7f'
R='/public/home/aclsylqidf/hygon-inference-refactor-20261014/'
LABEL='Deep-r465-public-fixed-cohort-public-fixed-c4-20261104-a'
O=R+'results/Deep-public-fixed-cohort-r465/public-fixed-c4-20261104-a/'
C=R+'results/'+LABEL+'-client/'
NATIVE='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'
H=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda q:json.dumps(q,ensure_ascii=False,separators=(',',':')).encode()
def summary(v):
 v=[x for x in v if x is not None]
 if not v:return None
 s=sorted(v);n=len(s)
 def p(q):
  x=(n-1)*q;i=int(x);return s[i]+(s[min(i+1,n-1)]-s[i])*(x-i)
 assert all(math.isfinite(x)for x in v)
 return dict(n=n,sum=sum(v),min=min(v),median=statistics.median(v),mean=statistics.mean(v),P95=p(.95),P99=p(.99),max=max(v))
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return[max(0,m-h),min(1,m+h)]
def grade(g,text,natural,cap,tools):
 if g['suite']=='gsm8k':
  v=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',text)
  try:parsed=Fraction(v[-1].replace(',',''))if v else None
  except(ValueError,ZeroDivisionError):parsed=None
  correct=parsed is not None and parsed==Fraction(g['gold_numeric'])
 else:
  assert g['suite']=='mmlu';v=re.findall(r'(?:^|\n)\s*(?:\*\*)?(?:Answer|Final answer)\s*:\s*[\[(]?([ABCD])[\])]?(?:\*\*)?(?:[.\s]|$)',text,re.I)
  m=re.fullmatch(r'\s*[\[(]?([ABCD])[\])]?\s*[.]?\s*',text,re.I);parsed=v[-1].upper()if v else m[1].upper()if m else None;correct=parsed==g['gold_choice']
 return dict(parsed_answer=str(parsed)if parsed is not None else None,parse_failure=parsed is None,answer_matches_gold=bool(correct),correct=bool(correct and natural and not cap and not tools),natural_EOS=natural,cap_hit=cap,unexpected_tool_intent=bool(tools))
def unpack(path):
 raw=Path(path).read_bytes();assert H(raw)==SEAL;q=json.loads(gzip.decompress(raw));files={}
 for f in q['files']:
  assert f['path']not in files;b=gzip.decompress(base64.b64decode(f['gzip_base64']));assert len(b)==f['bytes']and H(b)==f['SHA256'];files[f['path']]=b
 assert len(files)==len(q['files'])
 return q,files

def audit(path,b):
 global SEAL,LABEL,O,C
 SEAL=b["archive_SHA256"];LABEL=b["prefix"];O=R+"results/Deep-public-fixed-cohort-r465/remaining-c4-20261105-b-s"+str(b["shard"])+"/";C=R+"results/"+LABEL+"-client/"
 archive,files=unpack(path);J=lambda p:json.loads(files[p]);validate_sources(files,b)
 watch=J(R+'results/'+LABEL+'-watch.json');client=J(C+'result.json');owner=J(O+'owner.json');ready=J(O+'MODEL_READY.json');closed=J(O+'closed-engine.json');front=J(O+'FRONTEND.json')
 assert watch['completed']and watch['passed']and not watch['forced_cleanup']and watch['execution_passed'];assert(watch['CPU_keeper'],watch['actual_GPU'],watch['actual_client'])==(b['keeper'],b['GPU'],b['client'])
 for step in [b['GPU'],b['client'],b['post'],b['keeper'],watch['fresh_before'][0]['actual_step']]:assert any(s.startswith(step+'|')and'|COMPLETED|0:0|'in s for s in archive['accounting']['rows'])
 assert all(q['identity']==owner['identity']for q in[ready,closed,front]);identity=owner['identity'];assert identity['actual_step']==b['GPU']and identity['gpu_start_epoch']==b['epoch']
 assert identity['source_manifest_SHA256']=='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec'and identity['provider_config_SHA256']==b['config_SHA256']
 assert ready['Model_READY']and ready['rows']==4 and ready['capacity_per_row']==32768 and not ready['standard_public_C1'];assert owner['completed']and owner['normal_Native_shutdown']and owner['engine_worker_shutdown_completed']and owner['observed_all8_threads_exited']and owner['loaded_key_count']==67569
 assert closed['normal']and closed['engine_destroy_returned_zero']and closed['observed_all8_threads_exited'];assert owner['Native_SHA256']==NATIVE and(owner['TP'],owner['PP'],owner['HC_mode'],owner['cache_rows'],owner['cache_capacity'])==(8,1,0,4,32768)
 assert owner['precision_policy']['policy']==0 and owner['precision_policy']['status']==0 and not owner['precision_policy']['F64_allowed'];assert set(owner['actual_providers_before_weights'])=={'create','load','finish','forward','reset','destroy'}and all(v=='/refactor/candidates/Deep-HC-fullpre-Native-r377-cpu-v1/libDeep_HC_fullpre_optin_r377.so'for v in owner['actual_providers_before_weights'].values())
 old=None
 for key in['owner_affinity_before_weight_load','owner_affinity_after_weight_load']:
  a=owner[key]['actual_owner_report']['owners'];assert len(a)==8 and{v['device']for v in a}==set(range(8));masks=[]
  for v in a:assert v['changes']==1;m=[int(x)for x in v['verified_mask'].split(',')];assert len(m)==4 and len(set(m))==4;masks+=m
  assert len(set(masks))==32;signature=[v['verified_mask']for v in a];assert old is None or old==signature;old=signature
 assert owner['owner_thread_observation']['status']=='OBSERVED'and len({r['TID']for r in owner['owner_thread_observation']['owners']})==8
 assert all(f>=16*1024**3 for f,t in owner['GPU_memory_after_load']);assert owner['blas_policy_before_destroy']['failures']==0 and all(v['created']==v['verified']for v in owner['blas_policy_before_destroy']['owners'])
 post=watch['post'];assert len(post)==1;post=post[0];assert post['actual_step']==b['post']and post['node']==b['node']and post['KFD_available']and not post['kfd_pids']and not post['matching_component_processes']and not post['expected_PIDs_still_exist']and all(post['ports_bindable'].values())and len(set(post['physical_PCI']))==8
 pilotpath=R+'candidates/Deep-public-C4-remaining-quality-CPU-v2/shard'+str(b['shard'])+'-private.json';assert H(files[pilotpath])==b['plan_SHA256'];pilot=J(pilotpath);records=pilot['records'];assert len(records)==32 and[r['ordinal']for r in records]==b['ordinals'];case={v['case_id']:v for v in records};assert len(case)==32
 encpath=R+'candidates/Deep-broad-fewshot-fullpre-quality-r450-CPU-v1/encoding.json';assert H(files[encpath])=='04c1ffa27726cc89a177814d2e8ed4838fed51052fee715a33851b9b65c8dd5a';enc=J(encpath)
 goldpath=R+'incoming/GLM-W4-fewshot-quality-r444/fixtures-r1/cases.json';assert H(files[goldpath])=='c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38';gold={v['id']:v for v in J(goldpath)['cases']}
 for v in records:
  e=enc['cases'][v['ordinal']];p=R+e['private_path'].removeprefix('/refactor/');assert H(files[p])==e['private_SHA256'];original=J(p)
  assert original['body']==v['body']and original['prompt_token_ids']==v['prompt_token_ids']and v['gold']==gold[v['case_id']]
  assert H(compact(v['body']))==v['body_SHA256']==e['request_body_SHA256']and H(compact(v['prompt_token_ids']))==v['prompt_IDs_SHA256']==e['prompt_IDs_SHA256'];assert v['body']['max_tokens']==4096 and not v['body']['chat_template_kwargs']['enable_thinking']and len(v['prompt_token_ids'])+4096<=32768
 assert client['completed']and client['execution_passed']and len(client['rows'])==64 and len(client['cohorts'])==16 and len(owner['requests'])==64 and len(owner['cohorts'])==16
 requests={r['request_id']:r for r in owner['requests']};assert len(requests)==64;trialstates={};cohorts={};dispatch=collections.Counter();calls=0;active=collections.Counter();trialmetrics=[]
 for k,c in enumerate(owner['cohorts']):
  group=k//2;mode=(['serial','cohort']if group%2==0 else['cohort','serial'])[k%2];assert c['mode']==mode and c['completed']and not c['failed'];t=c['trial'];assert t['normal']and t['Native_reset_before']and t['Native_reset_after']and t['completed']and not t['failed'];assert t['rows']==4 and t['capacity_per_row']==32768 and len(t['states'])==4
  chosen=records[group*4:group*4+4];past=[0]*4;finished=set();cursor=[0]*4
  for call in t['calls']:
   assert call['rc']==0 and call['finite']and call['return_monotonic']>=call['start_monotonic']and math.isclose(call['seconds'],call['return_monotonic']-call['start_monotonic']);rows=call['rows'];assert len(rows)==len(set(rows))==call['output_rows']and not(finished&set(rows));assert call['past']==[past[i]for i in rows];widths=call['widths'];assert call['offsets']==[0]+[sum(widths[:i+1])for i in range(len(widths))]
   if call['kind']=='prefill':
    assert len(rows)==1 and widths==[min(1024,len(chosen[rows[0]]['prompt_token_ids'])-past[rows[0]])]
    assert past[rows[0]]<len(chosen[rows[0]]['prompt_token_ids'])
   else:
    assert call['kind']=='decode'and widths==[1]*len(rows)and all(past[i]>=len(chosen[i]['prompt_token_ids'])for i in rows);active[len(rows)]+=1
    if mode=='cohort':assert all(past[i]>=len(chosen[i]['prompt_token_ids'])for i in range(4))
    assert rows==([i for i in range(4)if i not in finished]if mode=='cohort'else rows)
   if mode=='serial':assert len(rows)==1 and rows==[next(i for i in range(4)if i not in finished)]
   for i,w in zip(rows,widths):
    past[i]+=w;assert past[i]<=32768
    if call['kind']=='decode'or past[i]==len(chosen[i]['prompt_token_ids']):
     ev=t['states'][i]['token_events'][cursor[i]];assert ev['monotonic']==call['return_monotonic'];cursor[i]+=1
     if ev['token']==1 or cursor[i]==4096:finished.add(i)
   dispatch[f"{call['kind']}-rows{len(rows)}-tile0-c40-active0"]+=1;calls+=1
  assert all(f>=16*1024**3 for key in ['memory_before','memory_after']for f,total in t[key]);assert math.isclose(t['whole_cohort_seconds'],t['completion_monotonic']-t['all_arrival_monotonic'])
  for i,s in enumerate(t['states']):
   assert s['row']==i and s['case_id']==c['requests'][i]and s['past']==past[i]and cursor[i]==len(s['generated_ids']);ids=s['generated_ids'];events=s['token_events'];assert ids==[e['token']for e in events]and s['finish']in['EOS','cap']and all(type(x)is int and 0<=x<129280 for x in ids)
   assert(ids[-1]==1 and ids.count(1)==1)if s['finish']=='EOS'else(len(ids)==4096 and 1 not in ids)
   assert s['past']==len(chosen[i]['prompt_token_ids'])+len(ids)-1
   assert all(e['visible']==(e['token']!=1)for e in events)and all(b['monotonic']>=a['monotonic']for a,b in zip(events,events[1:]));trialstates[s['case_id']]=(s,t,c)
  cohorts[c['id']]=c
  trialmetrics.append(dict(group=group,mode=mode,owner_whole_s=t['whole_cohort_seconds'],Native_caller_sum_s=sum(v['seconds']for v in t['calls']),calls=len(t['calls'])))
 assert calls==owner['HC_dispatch_checked_forwards']==owner['layout_dispatch_checked_forwards']==sum(owner['dispatch_checks_r4842'].values());assert dict(dispatch)==owner['dispatch_checks_r4842'];assert watch['Native_CAPI_calls']==calls and active[4]==watch['true_C4_decode_calls']
 rows=[];quality={};user=[]
 for r in client['rows']:
  v=case[r['case_id']];mode=r['mode'];g=r['group'];i=r['row'];assert v['ordinal']==r['ordinal']and records[g*4+i]['case_id']==v['case_id'];stem=f'g{g}-{mode}-r{i}';raw=files[C+stem+'.request.json'];body=json.loads(raw);ticket=body.pop('native_fixed_cohort');assert body==v['body']and ticket==dict(id=r['cohort'],mode=mode,row=i,size=4)and H(raw)==r['request_SHA256']and r['original_body_SHA256']==v['body_SHA256'];wire=files[C+stem+'.wire'];assert H(wire)==r['wire_SHA256']
  if body.get('stream'):
   content='';tools=[];finish=None;usage=None;rid=None
   for line in wire.splitlines():
    if not line.startswith(b'data: ')or line[6:].strip()==b'[DONE]':continue
    event=json.loads(line[6:]);rid=event.get('id',rid);usage=event.get('usage')or usage
    for ch in event.get('choices',[]):d=ch.get('delta',{});content+=d.get('content')or'';tools.extend(d.get('tool_calls')or[]);finish=ch.get('finish_reason')or finish
  else:q=json.loads(wire);rid=q['id'];usage=q['usage'];ch=q['choices'][0];content=ch['message'].get('content')or'';tools=ch['message'].get('tool_calls')or[];finish=ch['finish_reason']
  assert rid==r['request_id']and usage==r['usage'];s,t,c=trialstates[rid];o=requests[rid];m=o['final_metadata'];ids=s['generated_ids'];assert ids==o['generated_token_ids']==m['Native_generated_IDs'];assert len(ids)==usage['completion_tokens']==r['Native_generated_tokens']and usage['prompt_tokens']==len(v['prompt_token_ids']);assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'];assert o['normal_reset_after']and m['cohort_normal_reset_after']and m['cohort_row']==i and m['fixed_cohort_id']==r['cohort']and m['fixed_cohort_mode']==mode
  assert m['Native_prompt_token_ids_SHA256']==m['history_render_audit']['Native_prompt_token_ids_SHA256']==v['prompt_IDs_SHA256'];a=m['history_render_audit'];assert a['prompt_tokens']==len(v['prompt_token_ids'])and a['output_budget']==4096 and a['official_reencode_equals_actual_prompt_and_IDs']and a['prequeue_bound'];assert m['raw_budget_contract']['selected']==4096 and not m['raw_budget_contract']['clamp']and m['real_cache_rows']==4 and m['real_cache_capacity']==32768
  assert (finish=='stop')==(s['finish']=='EOS')and(finish=='length')==(s['finish']=='cap');score=grade(v['gold'],content,finish=='stop',finish=='length',tools)
  for k,value in score.items():assert r['score'][k]==value,(r['case_id'],mode,k)
  clk=r['client_clock'];assert clk['sent']<=clk['headers']<=clk['completed']and math.isclose(clk['wall_s'],clk['completed']-clk['sent']);ticks=m['token_monotonic'];assert len(ticks)==len(ids)and all(y>=x for x,y in zip(ticks,ticks[1:]));visible=[e['monotonic']for e in s['token_events']if e['visible']];itls=[y-x for x,y in zip(visible,visible[1:])]
  u=dict(case_id=v['case_id'],suite=v['gold']['suite'],ordinal=v['ordinal'],mode=mode,group=g,row=i,prompt_tokens=len(v['prompt_token_ids']),generated_including_EOS=len(ids),visible_tokens=len(visible),**score,HTTP_stream=bool(body.get('stream')),HTTP_whole_s=clk['wall_s'],HTTP_headers_s=clk['headers']-clk['sent'],HTTP_first_semantic_chunk_s=clk['semantic_chunk_ticks'][0]-clk['sent']if clk['semantic_chunk_ticks']else None,owner_TTFT_from_accept_s=ticks[0]-m['accepted_monotonic'],Native_return_TTFT_from_cohort_arrival_s=visible[0]-t['all_arrival_monotonic']if visible else None,Native_return_visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 else None,Native_return_ITL_seconds=summary(itls),emitter_visible_decode_tps=None)
  emitted=[tick for tick,token in zip(ticks,ids)if token!=1];assert all(y>x for x,y in zip(emitted,emitted[1:]));u['emitter_visible_decode_tps']=(len(emitted)-1)/(emitted[-1]-emitted[0])if len(emitted)>1 else None
  rows.append(u)
 assert len({(r['case_id'],r['mode'])for r in rows})==64
 for suite in['gsm8k','mmlu','pooled']:
  selected=[v for v in rows if suite=='pooled'or v['suite']==suite];n=32 if suite=='pooled'else 16;arms={}
  for mode in['serial','cohort']:
   rr=[v for v in selected if v['mode']==mode];assert len(rr)==n;k=sum(v['correct']for v in rr);arms[mode]=dict(planned=n,completed=len(rr),correct=k,accuracy=k/n,Wilson95=wilson(k,n),natural_EOS=sum(v['natural_EOS']for v in rr),caps=sum(v['cap_hit']for v in rr),parse_failures=sum(v['parse_failure']for v in rr))
  ix={(v['case_id'],v['mode']):v for v in selected};pairs=[(ix[(i,'serial')],ix[(i,'cohort')])for i in sorted({v['case_id']for v in selected})];gain=sum(not a['correct']and b['correct']for a,b in pairs);loss=sum(a['correct']and not b['correct']for a,b in pairs);discord=gain+loss;delta=(gain-loss)/n;radius=math.sqrt(2*math.log(40)/n)
  quality[suite]=dict(arms=arms,paired=dict(n=n,cohort_only_correct=gain,serial_only_correct=loss,both_correct=sum(a['correct']and b['correct']for a,b in pairs),both_wrong=sum(not a['correct']and not b['correct']for a,b in pairs),cohort_minus_serial_accuracy=delta,paired_Hoeffding95=[max(-1,delta-radius),min(1,delta+radius)],exact_McNemar_two_sided_p=min(1,2*sum(math.comb(discord,k)for k in range(min(gain,loss)+1))/2**discord)if discord else 1))
 metrics={}
 for mode in['serial','cohort']:
  rr=[v for v in rows if v['mode']==mode];cc=[v for v in client['cohorts']if v['mode']==mode];wall=sum(v['whole_client_seconds']for v in cc);visible=sum(v['visible_tokens']for v in rr)
  metrics[mode]=dict(logical_requests=32,HTTP_cohort_wall_sum_s=wall,owner_whole_cohort_sum_s=sum(v['owner_whole_s']for v in trialmetrics if v['mode']==mode),visible_tokens=visible,EOS_inclusive_tokens=sum(v['generated_including_EOS']for v in rr),whole_8card_visible_tps=visible/wall,per_card_visible_tps=visible/wall/8,correct_EOS_visible_goodput_8cards=sum(v['visible_tokens']for v in rr if v['correct'])/wall,correct_EOS_requests_per_s=sum(v['correct']for v in rr)/wall,per_user_decode_tps=summary([v['Native_return_visible_decode_tps']for v in rr]),owner_TTFT_from_accept_s=summary([v['owner_TTFT_from_accept_s']for v in rr]),HTTP_whole_s=summary([v['HTTP_whole_s']for v in rr]))
 assert all(t['completion_monotonic']-t['all_arrival_monotonic']>0 for c in owner['cohorts']for t in[c['trial']]);assert watch['service_deadline_unix']-watch['original_GPU_start_unix']==13800 and watch['GPU_deadline_unix']-watch['original_GPU_start_unix']==15000 and not watch['retry']and not watch['reload']
 return dict(passed=True,scope='32 fixed paired cases / 64 real public HTTP, same rows4 engine serialC1 versus activeC4; not standard rows1 C1',seal_SHA256=SEAL,files_verified=len(files),decoded_file_bytes=sum(map(len,files.values())),source_manifests_verified=True,actual_GPU=b['GPU'],CPU_keeper=b['keeper'],actual_client=b['client'],post=b['post'],Native_SHA256=NATIVE,profile=dict(HC=0,r366=0,fullpre=0,precision_policy=0,rows=4,capacity_per_row=32768),execution=dict(logical_requests=64,cohorts=16,Native_CAPI_calls=calls,decode_active_rows=dict(active),actual_all8_dispatch_counter_checked_calls=calls,natural_EOS=sum(v['natural_EOS']for v in rows),caps=sum(v['cap_hit']for v in rows),parse_failures=sum(v['parse_failure']for v in rows)),quality=quality,metrics=metrics,per_user=rows,cohorts=trialmetrics,normal_lifecycle=dict(Native_destroy_returned_zero=True,internal8owner_TIDs_exited=True,affinity_changes1_masks4_disjoint_stable=True,post_KFD_PID_ports_clean=True,CPU_GPU_client_post_accounting_all_completed0=True,forced_cleanup=False),limitations=['No broad noninferiority or automatic promotion; fixedremaining shard32 exactordinals, no output-based selection.','Different generated lengths between modes retained; throughput is observed useful output per complete cohort wall, not fixed-token causal kernel speed.','CAPI percall raw logits and raw perowner counter deltas were not archived: finite/all8 dispatch are successful production assertions plus aggregate counters/source, not independent logits replay.','Input945 IDs and public metadata hashes match; no tokenizer or Native re-execution. Raw activations/expert route IDs are absent.','Native return token clocks differ from bridge emitter clocks; both reported, HTTP chunks are not tokens. Nonstream HTTP first-token TTFT UNKNOWN; full response/header wall retained.','Nested caller Native wall is not exclusive GPUbusy; no device/CCL event profile in this run. SDK maps were asserted during production providers(), but raw maps were not persisted in owner.json.'],new_jobs=0,new_HTTP=0,Model_promotion=False,online_SLO=False)
def validate_sources(files,b):
 J=lambda p:json.loads(files[p]);P=R+'incoming/Deep-public-fixed-cohort-profile-r465/source-v1/';D=R+'incoming/Deep-public-C4-remaining-quality/deployment-v3/';F=R+'incoming/Deep-public-C4-remaining-quality/final-claim-v2/'
 assert H(files[P+'SOURCE_MANIFEST.json'])=='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec';pm=J(P+'SOURCE_MANIFEST.json')
 for n,h in pm['files_SHA256'].items():assert H(files[P+n])==h,n
 for p,h in pm['parents_manifest_SHA256'].items():assert H(files[R+p.removeprefix('/refactor/')+'/SOURCE_MANIFEST.json'])==h
 assert H(files[D+'SOURCE_MANIFEST.json'])=='95af7817339c78db12142501e536515d79199c3288ba1baf1d8492693e1462bf';cpu=J(D+'SOURCE_MANIFEST.json')
 assert H(files[F+'GPU_RESOURCE_CLAIM.json'])=='cc6f3f7d8ceb0e1f8c3dcd5232427a882b96a1b9f771b2a4b0626753cc1e6d79';claim=J(F+'GPU_RESOURCE_CLAIM.json');assert claim['CPU_receipt_source_MF']=='f46027ec3b796529f5a417efe35476dc8825068ed5e8b05bfda7683b89ea4c5e'
 for n,h in cpu['files_SHA256'].items():
  if n.endswith('/SOURCE_MANIFEST.json'):
   j=int(n.split('/')[0][5:]);assert H(files[F+'original-false-shard%d.json'%j])==h;assert H(files[D+n])==claim['shards'][j]['final_SHA256']
  else:assert H(files[D+n])==h,n
 for n,h in J(F+'SOURCE_MANIFEST.json')['files_SHA256'].items():assert H(files[F+n])==h
 actual=claim['shards'][b['shard']];assert actual['private_plan']['SHA256']==b['plan_SHA256'];assert actual['final_SHA256']==b['execution_manifest_SHA256']
 c=J(D+'shard%d/config.json'%b['shard']);assert H(files[D+'shard%d/config.json'%b['shard']])==b['config_SHA256'];assert c['node']==b['node']and(c['rows'],c['HC_mode'],c['r366'],c['fullpre'])==(4,0,0,0)
 ref=claim['CPU_receipt'];p=R+ref['path'].removeprefix('/refactor/');assert H(files[p])==ref['SHA256'];q=J(p);assert q['passed']and q['actual_CPU']=='910407.1034'and q['source_manifest_SHA256']==claim['CPU_receipt_source_MF']and len(q['shards'])==7
 ref=claim['actual_HOST_preflight'];assert H(files[ref['path']])==ref['SHA256'];q=J(ref['path']);assert q['passed']and q['source_MF']=='95af7817339c78db12142501e536515d79199c3288ba1baf1d8492693e1462bf'
 w=J(R+'results/'+b['prefix']+'-watch.json');assert w['publication_commit']=='f222a13801ddde061372c6b9e21ca4bdf4a86865'and w['deployment_source_SHA256']==b['execution_manifest_SHA256']and w['provider_source_SHA256']=='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec'
 ident=J(R+'results/'+b['prefix']+'-client-identity.json');assert ident['node']==b['node']and ident['actual_CPU']==b['client']and ident['GPU']==ident['Native']==0 and type(ident['PID'])is int and ident['PID']>0
 path=R+ident['client'].removeprefix('/refactor/')if ident['client'].startswith('/refactor/')else ident['client'];assert path.startswith(R);assert H(files[path])==ident['client_SHA256'];args=ident['argv']
 def arg(k):assert args.count(k)==1;return args[args.index(k)+1]
 assert arg('--expected-step')==b['GPU']and arg('--run')==b['prefix'].removeprefix('Deep-public-C4-remaining-quality-')and arg('--source-sha')=='cb0e092859a2804611a20673910b9289a9bf64e16e3d93abfe82d3a3d56e03ec'and arg('--config-sha')==b['config_SHA256']
 account=b['accounting'];expected={b['GPU']:(b['prefix'],b['node'],15000),b['client']:(b['prefix']+'-client',b['node'],13800),b['keeper']:(b['prefix']+'-keeper','a14r2n20',15600),b['post']:(b['prefix']+'-post',b['node'],180)}
 for step,(label,node,limit)in expected.items():
  r=[x.split('|')for x in account if x.split('|')[0]==step];assert len(r)==1 and r[0][1]==label and r[0][2:4]==['COMPLETED','0:0']and r[0][5]==node and int(r[0][4])<=limit
