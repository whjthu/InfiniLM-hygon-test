"""Read-only Model observer + authorized fixed data queue producer; never Slurm launch/cancel."""
import pathlib,json,hashlib,datetime,time,sys,subprocess,os
from transaction_evidence import extract as transaction_evidence
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');inc=r/'incoming/Kimi-state-operator-r9116d';record=r/'Kimi-state-operator-r9116d.json';name='Kimi-state-TP16PP2-r9114b';run='state-r9114b-7a2cec9379e624aa';expected_handle='910375.714';base=r/'results/Kimi-state-operator-r9116d';queue=r/'results/Kimi-state-queue-r9114b';plan=json.loads((inc/'pilot32-plan-r9116d.json').read_text());counter_names=['begin_epochs','captured_layers','replayed_layers','Conv_calls','KDA_calls','retained_rows_sum','captured_operand_bytes_cumulative']
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
report_cache={};report_freshness={}
def reports():
 global report_freshness
 report_freshness={};result=[]
 for node in range(4):
  p=r/'results'/('kimi-state-r9114b-'+run+'-node'+str(node)+'.json');probe={'fresh':False,'UTC':stamp()}
  try:
   d=json.loads(p.read_text());assert d['node']==node
   old=report_cache.get(node);n=len(d.get('speculative_quality_cases',[]));prev=len(old.get('speculative_quality_cases',[]))if old else 0
   if old and(n<prev or len(d.get('phase_calls',[]))<len(old.get('phase_calls',[]))):probe['reason']='transient nonmonotonic report; keep last known'
   else:report_cache[node]=d;probe.update(fresh=True,case_count=n)
  except (OSError,ValueError,KeyError)as e:probe['reason']=repr(e)
  if node in report_cache:result.append(report_cache[node]);probe['last_known_cases']=len(report_cache[node].get('speculative_quality_cases',[]))
  report_freshness[node]=probe
 return result
def mini_case(c):
 z={k:c.get(k)for k in ['label','strategy','K','input_tokens','generated_tokens','useful_decode_tokens','decode_user_wall_s','decode_user_tok_s','prefill_local_wall_s','EOS','truncated','mean_useful_outputs_per_cycle','proposal_acceptance','adaptive_fallback_cycle','state_owner_stats8x7','quality']};z['state_only_replay_cycles']=sum(x.get('state_only_replay',False)for x in c['cycles']);z['cost_s']={k:sum(x.get(k,0)for x in c['cycles'])for k in ['verify_s','replay_s','draft_RPC_s','draft_query_s','append_H2D_s','commit_local_s','cycle_user_wall_s']};return z
if len(sys.argv)==1:
 assert not record.exists();base.mkdir(exist_ok=False);launch=json.loads((r/(name+'-launch.json')).read_text());assert launch['run']==run and launch['draft_handle']=='910374.1048';w=json.loads((r/(name+'-watch.json')).read_text());assert w['actual_handle']==expected_handle and not w.get('terminal');record.write_text(json.dumps({'run':run,'target_handle':expected_handle,'status':'intent','UTC':stamp()}));f=(r/'logs/Kimi-state-operator-r9116d.log').open('x');p=subprocess.Popen([sys.executable,__file__,'watch'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'operator_pid':p.pid,'run':run,'target_handle':expected_handle,'new_Slurm_jobs':0}));raise SystemExit
assert sys.argv[1]=='watch';elapsedrows=subprocess.check_output(['sacct','-j',expected_handle,'-n','-P','-o','JobID,ElapsedRaw'],text=True,timeout=30).splitlines();elapsed=[int(x.split('|')[1])for x in elapsedrows if x.split('|')[0]==expected_handle];assert len(elapsed)==1;budget_tick=time.monotonic();budget_remaining=5000-elapsed[0]
state={'run':run,'target_handle':expected_handle,'status':'observing_same_run','UTC':stamp(),'pid':os.getpid(),'events':[],'quality_enqueued':False,'quality_admitted_pairs':0,'new_Slurm_jobs':0,'resumed_from':'Kimi-state-operator-r9116c.json; original observer error preserved','counter_fields':counter_names};last=None;deadline=time.monotonic()+5400
try:
 while time.monotonic()<deadline:
  w=json.loads((r/(name+'-watch.json')).read_text());assert w['run']==run and w['actual_handle']==expected_handle;ds=reports();em=next((d for d in ds if d['node']==2),{});cases=em.get('speculative_quality_cases',[]);quality=[c for c in cases if c['label'].startswith('pilot32-')];source_errors=[{'node':d['node'],'error':d.get('error'),'shutdown_error':d.get('shutdown_error')}for d in ds if d.get('error')or d.get('shutdown_error')];snapshot={'UTC':stamp(),'probe_freshness':report_freshness,'watch':{k:v for k,v in w.items()if k not in ['audit_stdout','audit_stderr']},'owners':[{k:d.get(k)for k in ['node','phase','active_command','loaded_keys','parameter_keys','loading_elapsed_seconds','loading_seconds','completed','error','shutdown_error']}for d in ds],'cases':[mini_case(c)for c in cases],'pilot_counts':{'finished_cases':len(quality),'AR':sum(c['strategy']=='ar'for c in quality),'state':sum(c['strategy']=='state'for c in quality),'correct':sum(c.get('quality',{}).get('correct',False)for c in quality),'EOS':sum(c['EOS']for c in quality),'truncated':sum(c['truncated']for c in quality),'parsefail':sum(c.get('quality',{}).get('extracted_answer')is None for c in quality)}}
  snapshot['transaction_evidence']=transaction_evidence(em)if cases else None
  signature=json.dumps([snapshot['owners'],[(c['label'],c.get('quality',{}).get('correct'))for c in cases],w.get('terminal')],sort_keys=True)
  state['latest_probe_freshness']=report_freshness
  if not em:
   state['observation_hold']='no emitter report; last trusted snapshot retained';write(record,state)
   if w.get('terminal'):state.update(status='same_run_terminal_without_current_emitter',terminal=w['terminal']);write(record,state);break
   time.sleep(5);continue
  if signature!=last:
   path=base/('event-%04d.json'%len(state['events']));write(path,snapshot);state['events'].append({'file':str(path),'UTC':snapshot['UTC'],'case_count':len(cases),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest()});last=signature;state['last_snapshot']=str(path);state['case_count']=len(cases);write(record,state)
  if source_errors:state['source_errors']=source_errors
  # Enqueue only after all16 core cases complete on every node and both minimal
  # state-only math answers are correct. Never reset/reload/native-hotpatch.
  core=[c for c in cases if not c['label'].startswith('pilot32-')]
  ready=len(ds)==4 and all(len(d.get('speculative_quality_cases',[]))>=16 for d in ds)and all(v['fresh']for v in report_freshness.values())
  if ready and not state.get('core_qualified')and not source_errors and not w.get('terminal'):
   smoke=[c for c in core if c['strategy']=='state'and c.get('quality')];assert len(smoke)==2
   if all(c['quality']['correct']and c['EOS']and not c['truncated']for c in smoke)and all(c.get('state_owner_stats8x7')and sum(c['state_owner_stats8x7'][0::7])>0 and sum(c['state_owner_stats8x7'][1::7])>0 for c in smoke):
    state['core_qualified']=True;state['core_qualified_UTC']=stamp();state['core_actual_partial_replay_exercised']=any(c.get('state_owner_stats8x7')and sum(c['state_owner_stats8x7'][2::7])>0 for c in smoke)
   else:state['quality_enqueue_hold']='core state math smoke or actual owner capture evidence not qualified';write(record,state)
  if state.get('core_qualified')and not source_errors and not w.get('terminal')and len(cases)>=16 and all(v['fresh']for v in report_freshness.values())and state['quality_admitted_pairs']<32:
   done={c['label']for c in cases};done.update(x['id']for x in em.get('capacity_skipped_commands',[]));existing={}
   for path in queue.glob('q*.json'):
    cmd=json.loads(path.read_text());existing[cmd.get('id')]=cmd
   known_ids=done|set(existing);prefix_pairs=0
   for i in range(32):
    pairids=[x['id']for x in plan['planned_commands'][2*i:2*i+2]]
    if all(x in known_ids for x in pairids):prefix_pairs+=1
    else:break
   state['quality_admitted_pairs']=max(state['quality_admitted_pairs'],prefix_pairs)
   if state['quality_admitted_pairs']>=32:state['quality_enqueued']=True;write(record,state);time.sleep(5);continue
   pending=[v for k,v in existing.items()if k not in done];remaining=budget_remaining-(time.monotonic()-budget_tick);state['budget_remaining_conservative_s']=remaining
   # Keep at most one paired case ahead. Predict max-output cost from the
   # slowest observed same-run AR/state rates; reserve150s before next case.
   rates={mode:min(c['decode_user_tok_s']for c in cases if c['strategy']==mode and c['decode_user_tok_s']>0)for mode in ['ar','state']}
   prefill=max(10.,max(c['prefill_local_wall_s']for c in cases));estimate=lambda cmd:cmd['cap']/rates[cmd['mode']]+prefill
   pairfiles=plan['planned_commands'][2*state['quality_admitted_pairs']:2*state['quality_admitted_pairs']+2];pair=[]
   for file in pairfiles:
    src=inc/'pending-pilot32-cap512'/file['file'];raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()==file['SHA256'];cmd=json.loads(raw);assert cmd['cap']==512 and cmd['mode']in ['ar','state']and cmd['id']==file['id'];assert not any(k in cmd for k in ['prompt','code','script','path']);pair.append((file,cmd,raw))
   required=sum(estimate(cmd)for cmd in pending)+sum(estimate(cmd)for _,cmd,_ in pair)+150;state['next_pair_required_conservative_s']=required
   if len(pending)<=2 and remaining>required:
    staged=[]
    for j,(file,cmd,raw)in enumerate(pair):
     if cmd['id']in done or cmd['id']in existing:continue
     p=queue/('q1000-pilot-%04d.json'%(2*state['quality_admitted_pairs']+j));assert not p.exists();tmp=p.with_suffix('.tmp');assert not tmp.exists();tmp.write_bytes(raw);tmp.replace(p);staged.append({'id':cmd['id'],'path':str(p),'SHA256':hashlib.sha256(raw).hexdigest(),'UTC':stamp()})
    state.setdefault('quality_staged',[]).extend(staged);state['quality_admitted_pairs']+=1;state['quality_enqueued']=True;state['quality_enqueue_UTC']=stamp();state['quality_scope']='fixedrandom32 paired AR/state cap512, one pair ahead bounded by observed cost and150s reserve; pending is not benchmark completion';write(record,state)
   elif remaining<=required:
    state['quality_budget_hold']=True;write(record,state)
  if w.get('terminal'):
   state['terminal']=w['terminal'];state['status']='same_run_terminal';state['quality_finished_cases']=len(quality);state['pending_quality_ids']=[c['id']for c in plan['planned_commands']if c['id']not in {x['label']for x in cases}];write(base/'terminal-snapshot.json',snapshot);write(record,state);break
  time.sleep(5)
 else:state['status']='observer_timeout_no_target_signal';write(record,state)
except BaseException as e:state['observer_error']=repr(e);state['status']='observer_error_no_target_signal';write(record,state);raise
