from pathlib import Path
import json,hashlib,ast,sys
r=Path.cwd();o=r/'Deep-eos-compaction-private-r9130';o.mkdir(exist_ok=False);src=r/'Deep-quality-private-r9101';old=(src/'Deep_quality_protocol_r9101.py').read_text(encoding='utf8');head=old.split(' report.update(',1)[0].replace('from quality_bench_r9100 import','from Deep_eos_quality_metrics_r9130 import')
body=r'''
 report.update(scope='One resident TP8PP1 8DCU, realGSM8K32 custom3shot. Same packed prefill micro4 in both arms; decode fixed rows vs retire EOS rows. B8/16/32 are closedcohorts, not HTTP or continuous serving.',quality_trials=[],grid_complete=False,performance_qualified=False,quality_noninferiority_proven=False,goal_complete=False,no_historical_token_identity_gate=True,quality_dataset=prepared['template_provenance'])
 deadline=g['packet_started']+3300
 def generate(items,compact,label):
  select(1);check(lib.modern_native_engine_reset(ctx));began=time.perf_counter();B=len(items);assert 1<=B<=32
  prompts=[v['input_ids']for v in items];caps=[min(512,int(v.get('max_new_tokens',512)))for v in items];generated=[[]for _ in items];done=[False]*B;reasons=[None]*B;windows=[];events=[[]for _ in items];budget=False
  def forward(rows,seq,past,prefill):
   nonlocal budget
   if time.perf_counter()>deadline-120:budget=True;return None
   A=len(rows);sizes=list(map(len,seq));assert A and len(set(rows))==A and all(0<=x<B for x in rows);assert all(p+n<=1024 for p,n in zip(past,sizes))
   off=[0]+np.cumsum(sizes).tolist();pos=sum([list(range(p,p+n))for p,n in zip(past,sizes)],[])
   arrays=[np.ascontiguousarray(v,dtype=np.int64 if j>=4 else np.int32)for j,v in enumerate([rows,past,[p+n for p,n in zip(past,sizes)],off,sum(seq,[]),pos])];inputs=[v.copy()for v in arrays]
   outback=np.full(A*cfg['vocab_size']+64,0x7fc0,np.uint16);outback[-64:]=0x1234;out=outback[:-64].reshape(A,cfg['vocab_size']);ib=np.full(A+8,-999,np.int64);ids=ib[:A]
   before=counters('deep_reference_packed_batch_counts_r8996b',7)if prefill else None
   tick=time.perf_counter();rc=lib.modern_native_engine_forward(ctx,sum(sizes),A,*[ctypes.c_void_p(v.ctypes.data)for v in arrays],ctypes.c_void_p(out.ctypes.data),ctypes.c_void_p(ids.ctypes.data));ended=time.perf_counter();check(rc)
   assert np.all(outback[-64:]==0x1234)and np.all(ib[A:]==-999)and all(np.array_equal(a,b)for a,b in zip(arrays,inputs))
   logits=(out.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(logits).all()and np.array_equal(logits.argmax(-1),ids);assert np.all((ids>=0)&(ids<cfg['vocab_size']))
   delta=None
   if prefill:
    after=counters('deep_reference_packed_batch_counts_r8996b',7);delta=[[y-x for x,y in zip(a,b)]for a,b in zip(before,after)];assert 64<sum(sizes)<=2048 and all(v[0]+v[1]==43 and v[0]>0 and v[3]>0 and v[4]>0 for v in delta)
   useful=0
   for j,i in enumerate(rows):
    if done[i]:continue
    token=int(ids[j]);generated[i].append(token);eos=token==EOS;events[i].append(dict(time=ended,kind='EOS'if eos else'committed',visible=not eos,token_id=token));useful+=int(not eos)
    if eos:done[i]=True;reasons[i]='EOS'
    elif len(generated[i])>=caps[i]:done[i]=True;reasons[i]='max_new_tokens'
   windows.append(dict(prefill=prefill,rows=rows.copy(),physical_rows=A,active_useful_tokens=useful,seconds=ended-tick,return_monotonic=ended,packed_delta=delta));return True
  # Stage all requests with a proven <=2048-token prefill. Same prefill schedule in both arms.
  for start in range(0,B,4):
   rows=list(range(start,min(B,start+4)))
   if forward(rows,[prompts[i]for i in rows],[0]*len(rows),True)is None:break
   report['live_generation']={'label':label,'phase':'prefill','admitted_rows':min(B,start+4)};save()
  for step in range(1,max(caps)):
   if budget or all(done):break
   rows=[i for i in range(B)if not done[i]]if compact else list(range(B));seq=[[generated[i][-1]if not done[i]else EOS]for i in rows];past=[len(prompts[i])+step-1 for i in rows]
   if forward(rows,seq,past,False)is None:break
   if step%32==0:
    assert min(g['torch'].cuda.mem_get_info(d)[0]for d in range(8))>=512*1024**2;report['live_generation']={'label':label,'step':step,'active':sum(not x for x in done),'last_physical_rows':len(rows),'generated_lengths':[len(v)for v in generated]};save()
  ended=time.perf_counter();records=[];requests=[]
  for i,item in enumerate(items):
   reason=reasons[i]or'wall_budget';text=tokenizer.decode(generated[i],skip_special_tokens=True).strip();request={'id':item['id'],'arrival':began,'status':'unfinished'if reason=='wall_budget'else'completed','stop_reason':reason,'events':events[i]};requests.append(request);records.append(dict(id=item['id'],generated_ids=generated[i],text=text,stop_reason=reason,input_tokens=len(prompts[i]),score=score_case(gold[item['id']],text,reason),logical_row=i))
  pre=sum(w['seconds']for w in windows if w['prefill']);dec=sum(w['seconds']for w in windows if not w['prefill']);useful=sum(w['active_useful_tokens']for w in windows if not w['prefill']);physical=sum(w['physical_rows']for w in windows if not w['prefill']);outs=sum(w['active_useful_tokens']for w in windows)
  return dict(label=label,B=B,compact_EOS_rows=compact,records=records,requests=requests,windows=windows,native_prefill_seconds=pre,native_decode_seconds=dec,native_prefill_input_tok_s=sum(map(len,prompts))/pre if pre else None,native_decode_useful_tok_s=useful/dec if dec else None,native_decode_useful_tokens=useful,physical_decode_row_steps=physical,driver_wall_seconds=ended-began,driver_output_tok_s=outs/(ended-began),serving_metrics=serving_metrics(requests,began,ended,8),budget_stop=budget,finite_readonly_canary_all_calls=True,all_prefill_packed_hit=True,clock_scope='Single native CAPI return emitter; same closedcohort arrival before reset, includes allprefill staging inTTFT, notHTTP or continuous service')
 order=sorted(cases,key=lambda v:(-v['input_tokens'],v['id']));grid={};report['matrix']=[8,16,32];report['prefill_microbatch']=4;save()
 for B in [8,16,32]:
  score={False:[],True:[]};trials={False:[],True:[]}
  for group in range(0,32,B):
   items=order[group:group+B]
   for compact in ([False,True]if(group//B)%2==0 else[True,False]):
    if time.perf_counter()>deadline-180:report['budget_stop_before']={'B':B,'group':group,'compact':compact};break
    label='B%d-group%d-%s'%(B,group//B,'compact'if compact else'fixed');entry=generate(items,compact,label);report['quality_trials'].append(entry);trials[compact].append(entry);score[compact].extend(z['score']for z in entry['records']);save()
    if entry['budget_stop']:break
   if report.get('budget_stop_before')or any(t['budget_stop']for ts in trials.values()for t in ts):break
  arms={}
  for compact in [False,True]:
   ts=trials[compact];pre=sum(t['native_prefill_seconds']for t in ts);dec=sum(t['native_decode_seconds']for t in ts);wall=sum(t['driver_wall_seconds']for t in ts);useful=sum(t['native_decode_useful_tokens']for t in ts)
   arms['compact'if compact else'fixed']={'groups':len(ts),'cases':len(score[compact]),'native_prefill_seconds':pre,'native_decode_seconds':dec,'native_decode_useful_tok_s':useful/dec if dec else None,'native_decode_useful_tokens':useful,'driver_wall_seconds':wall,'physical_decode_row_steps':sum(t['physical_decode_row_steps']for t in ts),'task_correct':sum(x['correct']for x in score[compact])}
  common={x['id']for x in score[False]}&{x['id']for x in score[True]};paired=paired_quality([x for x in score[False]if x['id']in common],[x for x in score[True]if x['id']in common])if common else None
  grid[str(B)]={'arms':arms,'paired_quality':paired,'complete':len(score[False])==len(score[True])==32 and not any(t['budget_stop']for ts in trials.values()for t in ts)};report['grid']=grid;save()
  if not grid[str(B)]['complete']:break
 report['grid_complete']=len(grid)==3 and all(x['complete']for x in grid.values());report['performance_qualified']=report['grid_complete'];report['quality_noninferiority_proven']=False;report['outcome']='GRID_COMPLETE'if report['grid_complete']else'PARTIAL_BUDGET';save()
'''
(o/'Deep_eos_compaction_r9130.py').write_text(head+body,encoding='utf8');(o/'Deep_eos_quality_metrics_r9130.py').write_bytes((r/'quality_bench_r9100.py').read_bytes())
model=(src/'modern_Deep_quality_Model_r9101.py').read_text(encoding='utf8').replace('deep-quality-bench-r9101-','deep-eos-compaction-r9130-').replace('Deep-quality-Model-manifest-r9101.json','Deep-eos-compaction-manifest-r9130.json').replace('modern_Deep_quality_Model_r9101.py','modern_Deep_eos_compaction_r9130.py').replace('Deep_quality_protocol_r9101.py','Deep_eos_compaction_r9130.py').replace('from Deep_quality_protocol_r9101 import run','from Deep_eos_compaction_r9130 import run').replace("Explicit human instruction: final full token IDs exact, logits bitwise differences diagnostic; finite, CPUargmax, EOS, ownership and cleanup retained.","Current user allows different tokens; heldout task quality and finite/currentargmax/lifecycle are checked. No historical sequence identity gate.")
(o/'modern_Deep_eos_compaction_r9130.py').write_text(model,encoding='utf8')
inside=(src/'Deep_quality_Model_r9101_inside.sh').read_text(encoding='utf8').replace('2600 python','3500 python').replace('modern_Deep_quality_Model_r9101.py','modern_Deep_eos_compaction_r9130.py');(o/'Deep_eos_compaction_r9130_inside.sh').write_text(inside,encoding='utf8')
verify="""import json,pathlib,sys,hashlib
p=pathlib.Path(sys.argv[1]);q=json.loads(p.read_bytes());assert q['completed']and q['engine_worker_shutdown_completed']and q['collection_finished'];assert q['grid_complete']and set(q['grid'])=={'8','16','32'}
for B in ['8','16','32']:
 g=q['grid'][B];assert g['complete']and g['paired_quality']['n']==32
 for arm in ['fixed','compact']:assert g['arms'][arm]['cases']==32
for t in q['quality_trials']:
 assert t['finite_readonly_canary_all_calls']and t['all_prefill_packed_hit']and not t['budget_stop'];assert len(t['records'])==t['B']
 assert len({z['logical_row']for z in t['records']})==t['B']
print(json.dumps({'passed':True,'step':q['step'],'grid':[8,16,32],'paired_real_quality_cases_each':32,'native_not_HTTP':True,'quality_noninferiority_proven':False,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}))
""";(o/'verify_Deep_eos_compaction_r9130.py').write_text(verify,encoding='utf8')
launch=(src/'launch_Deep_quality_Model_r9101.py').read_text(encoding='utf8').replace('deep-quality-bench-r9101','deep-eos-compaction-r9130').replace('Deep-quality-Model-manifest-r9101.json','Deep-eos-compaction-manifest-r9130.json').replace('verify_Deep_quality_Model_r9101.py','verify_Deep_eos_compaction_r9130.py').replace('Deep_quality_Model_r9101_inside.sh','Deep_eos_compaction_r9130_inside.sh').replace('00:45:00','01:00:00').replace('p.wait(timeout=2800)','p.wait(timeout=3700)')
(o/'launch_Deep_eos_compaction_r9130.py').write_text(launch,encoding='utf8')
m=json.loads((src/'Deep-quality-Model-manifest-r9101.json').read_bytes());m['incoming_source_SHA256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in o.iterdir()if p.suffix in ['.py','.sh']};m['resource'].update(hard_seconds=3600,inner_seconds=3500);m['plan']='One existingDSO resident load; B8/16/32 x fixedEOSpadding/compactfinishedrows, 32matchedGSM8K cases perarm, same packedPFmicro4; actualEOS/max512; noHTTP claim';m['actual_final_argv']=['timeout','--kill-after=20s','3500','python','-B','-u','/refactor/incoming/modern_Deep_eos_compaction_r9130.py'];m['status']='SOURCE_AST_ONLY_CPU_PREFLIGHT_PENDING';m['existing_incoming_dependency_SHA256'].update({k:v for k,v in json.loads((src/'Deep-quality-Model-manifest-r9101.json').read_bytes())['incoming_source_SHA256'].items()if k in ['quality-bench-gsm8k-r9100.json','quality-bench-gsm8k-train-shots-r9100.json','deepseek_v4_reference_template_r18_r2.py']});(o/'Deep-eos-compaction-manifest-r9130.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf8')
for p in o.glob('*.py'):ast.parse(p.read_bytes(),str(p))
print(json.dumps({'AST_pass':True,'files':len(list(o.iterdir())),'GPU_launched':False,'no_new_DSO':True}))
