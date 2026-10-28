"""Real new scheduler/719 helper with fake CPU forward; not model/HTTP proof."""
import importlib.util,sys,pathlib,json,ast,hashlib
D=pathlib.Path(__file__).resolve().parent;V=pathlib.Path('/refactor/candidates/GLM-arrival-defer-r153');sys.path.insert(0,str(D));import arrival_runner_r9258 as r
requests=[{'id':'CPU-'+str(i),'input_ids':[i+1]*10,'max_new_tokens':8}for i in range(32)];checks=[]
for arrivals in [[0.]*32,[4.*(i//8)for i in range(32)]]:
 outcomes=[]
 for enabled in [False,True]:
  now=[0.];events=[];last=[None];pastledger=[0]*32
  def clock():return now[0]
  def wait(s):now[0]+=s
  def exchange(cmd,seq):last[0]=r.decode_command(r.encode_command(cmd),seq);return last[0]
  def forward(rows,values,pasts,kind,prefill):
   assert len(rows)<=(16 if not prefill else 8)and len(rows)==len(set(rows));assert pasts==[pastledger[i]for i in rows]
   for i,p,t in zip(rows,pasts,values):pastledger[i]=p+len(t)
   now[0]+=2.;events.append(('forward',kind,list(rows)));out=[999 if pastledger[i]>=12 else 17+i for i in rows];return out,2.,now[0]-2.,now[0]
  def tokens(cmd,out):assert len(cmd['items'])<=16;events.append(('feedback',cmd['op'],[x[0]for x in cmd['items']]));return out
  result,trace=r.run_episode(requests,arrivals,False,{999},True,exchange,tokens,forward,clock=clock,wait=wait,budget_seconds=120,deferred_feedback=enabled)
  assert result['all_requests_completed']and result['max_admitted_live']==32 and result['peak_admitted_live']==32 and result['global_reset_allowed'];assert all(a['stop_reason']=='EOS'and len(a['generated_ids'])==3 and a['service_start']>=a['admitted']>=a['arrival']for a in result['requests']);assert pastledger==[12]*32
  outcomes.append([a['generated_ids']for a in result['requests']]);decode=[e for e in events if e[1]in ['arrival.DECODE','DECODE']]
  if enabled:assert any(a[0]==b[0]=='forward'for a,b in zip(decode,decode[1:]))
  else:assert all(not(a[0]==b[0]=='forward')for a,b in zip(decode,decode[1:]))
  checks.append({'arrivals':arrivals[:9],'enabled':enabled,'passed':True,'actual_commands':len(trace),'fake_forward_only':True})
 assert outcomes[0]==outcomes[1]
# CPU verifies the actual private probe source accepts split subcommands and no hidden kernel build.
for p in D.glob('*.py'):ast.parse(p.read_text())
assert hashlib.sha256((D/'decode_wave_r9225.py').read_bytes()).hexdigest()=='20482dc35b2f45bb0de1dad7a97b68a69e66638b0d3ab47ee1fcd5bf966ba911'

tree=ast.parse((D/'glm_arrival_probe_r153.py').read_text());outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');episode=next(n for n in outer.body if isinstance(n,ast.FunctionDef)and n.name=='episode');fn=next(n for n in episode.body if isinstance(n,ast.FunctionDef)and n.name=='command_exchange')
elapsed=[0.];ticks=[];last={};env={'cm':lambda cmd,seq:dict(cmd,seq=seq),'last':last,'warm':False,'label':'CPU-counterexample','progress':lambda *args:ticks.append(elapsed[0])};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_progress_callback','exec'),env)
durations=[12.821089,.2]*4+[1.02]*8;maxgap=0.
for seq,seconds in enumerate(durations,1):
 env['command_exchange']({'op':'PREFIX'if seq<=8 else'DECODE'},seq);elapsed[0]+=seconds;maxgap=max(maxgap,elapsed[0]-ticks[-1])
assert len(ticks)==16 and maxgap==max(durations)and maxgap<60 and sum(durations)>60
checks.append({'actual_progress_callback_AST':True,'every_command_tick':True,'max_gap':maxgap,'sixty_second_guard_unchanged':True,'no_background_keepalive':True,'not_GPU_performance_or_actual778_replay':True})
out={'passed':True,'CPU_only':True,'GPU':False,'model':False,'real_719_helper_byte_exact':True,'checks':checks,'source_SHA256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest()for n in ['arrival_runner_r9258.py','glm_arrival_probe_r153.py','decode_wave_r9225.py']}}
(V/'CPU-GLM-arrival-defer-r153.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
