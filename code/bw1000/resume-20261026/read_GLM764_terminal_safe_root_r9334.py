import pathlib,json,hashlib,statistics,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();wp=r/'GLM-M15-Model-r129-watch.json';w=json.loads(wp.read_bytes());assert w['qualified']and w['post_passed']and w['terminal'][0:4]==['910376.764','GLM-M15-Model-r129','COMPLETED','0:0']
launch=json.loads((r/'GLM-M15-Model-r129-launch.json').read_bytes());run=launch['run'];out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'level':'ACTUAL_SAME_LOAD_FINITE_ARRIVAL_MODEL','cards':16,'handle':'910376.764','watch_SHA256':sha(wp),'source_publication_commit':launch['publication_commit'],'run':run,'owners':[],'episodes':[],'lifecycle':{k:w.get(k)for k in ['qualified','terminal','audit_returncode','audit_SHA256','audit_qualified','post_KFD','post_passed','logs']}}
def desc(v):
 x=sorted(v)
 def pct(p):
  if not x:return None
  at=(len(x)-1)*p;i=int(at);j=min(i+1,len(x)-1);return x[i]+(x[j]-x[i])*(at-i)
 return {'n':len(x),'min':min(x)if x else None,'mean':statistics.mean(x)if x else None,'p50':pct(.5),'p95':pct(.95),'p99':pct(.99),'max':max(x)if x else None}
for stage in [0,1]:
 for rank in range(8):
  p=r/'results'/('glm-arrival-r129-'+run+'-stage'+str(stage)+'-rank'+str(rank)+'.json');q=json.loads(p.read_bytes());assert q['completed']and q['worker_shutdown_completed']and q['normal_reset_close']and q['benchmark_complete']
  out['owners'].append({'stage':stage,'rank':rank,'path':str(p),'SHA256':sha(p),'library_SHA256':q['library_sha256'],'minimum_driver_free_bytes':q['minimum_driver_free_bytes'],'M15_after_destroy':q['M15_after_destroy'],'episodes':[{'label':a['label'],'on':a['m15_oproj'],'delta17':a['m15_provider']['delta'],'expected_eligible':a['m15_provider']['expected_eligible_projection_calls']}for a in q['episodes']]})
  if stage!=1 or rank!=0:continue
  out['paired_quality']=q['paired_quality'];out['native_load_seconds']=q['actual_owner_weight_load_wall_seconds']
  for a in q['episodes']:
   s=a['serving'];users=[];gaps=[];visible=decode=0;quality=a['task_scores'];calls={x['op']:[]for x in a['native_calls']}
   for c in a['native_calls']:calls[c['op']].append(q['native_events'][c['event_index']])
   for u in s['requests']:
    ev=[e for e in u['events']if e.get('visible')and not any(e.get(k,False)for k in ['EOS','pad','draft','replay'])];stamps=[e['time']for e in ev];n=len(stamps);visible+=n;decode+=max(0,n-1);gap=[b-a for a,b in zip(stamps,stamps[1:])];gaps+=gap
    users.append({'id':u['id'],'arrival':u['arrival'],'N':n,'first':stamps[0]if n else None,'last':stamps[-1]if n else None,'user_tok_s':(n-1)/(stamps[-1]-stamps[0])if n>1 and stamps[-1]>stamps[0]else None,'TTFT':u['TTFT_seconds'],'stop_reason':u.get('stop_reason'),'score':u.get('score')})
   ds=sum(e['native_seconds']for e in calls.get('DECODE',[]));pf=[e for op in ['PREFIX','ANCHOR']for e in calls.get(op,[])];ps=sum(e['native_seconds']for e in pf);pt=sum(e['input_tokens']for e in pf)
   out['episodes'].append({'label':a['label'],'on':a['m15_oproj'],'complete':a['complete'],'elapsed_seconds':s['elapsed_seconds'],'visible_output_tokens':visible,'task_output_tok_s':visible/s['elapsed_seconds'],'task_output_tok_s_per_card':visible/s['elapsed_seconds']/16,'visible_decode_tokens_excluding_first_EOS':decode,'decode_HOST_CAPI_seconds':ds,'decode_tok_s_per_HOST_CAPI':decode/ds,'prefill_inputs':pt,'prefill_HOST_CAPI_seconds':ps,'prefill_input_tok_s_per_HOST_CAPI':pt/ps,'user_decode':desc([u['user_tok_s']for u in users if u['user_tok_s']is not None]),'TTFT':desc([u['TTFT']for u in users if u['TTFT']is not None]),'ITL':desc(gaps),'user_above15':sum(u['user_tok_s']is not None and u['user_tok_s']>15 for u in users),'correct':sum(s['flexible_correct']for s in quality),'natural_EOS':sum(u['stop_reason']=='EOS'for u in users),'caps':sum(u['stop_reason']=='max_new_tokens'for u in users),'quality':quality,'users':users,'original_metrics':a['metrics'],'M15_provider':a['m15_provider']})
assert len(out['owners'])==16 and len(out['episodes'])==4
out['all_owners_measured_M15_activated']=all(all(a['expected_eligible']>0 and a['delta17'][1]>0 and (a['delta17'][2]>0 if a['on']else a['delta17'][3]>0)for a in o['episodes'])for o in out['owners'])
out['limits']=['Nativefinitearrivalemulation notHTTP;same32questionsfourarmsnot128unique','CAPIdecoderatedenominatorexcludesseparateprefill/control/emission;userfirstlast retainsallwait','NoModelgainfromcomponentratio;noexacttokenqualitygate;15/usernotimplied']
print(json.dumps(out))
