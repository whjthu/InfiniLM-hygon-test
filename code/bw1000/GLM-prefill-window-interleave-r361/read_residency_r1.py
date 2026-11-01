"""Only follow the original361 labels/handles; metadata without raw IDs/body."""
from pathlib import Path
import json,subprocess,datetime
R=Path.home()/'hygon-inference-refactor-20261014';run='GLM-C2M1-r361-a';prefix='GLM-r361-'+run
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'Slurm':{},'logs':{},'tail':None}
for parent in ['910371','910374','910407']:
 raw=subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,Elapsed,NodeList%120'],text=True,timeout=25)
 q['Slurm'][parent]='\n'.join(x for x in raw.splitlines()if prefix in x)
watch=R/'results'/('GLM-r361-watch-'+run+'.json')
if watch.exists():
 try:q['watch']=json.loads(watch.read_bytes())
 except json.JSONDecodeError:q['watch_partial_JSON']=True
for p in(R/'logs').glob(prefix+'*.log'):q['logs'][str(p)]=p.read_text(errors='replace')[-2200:]
ready=R/'results'/('GLM-r361-ready-'+run+'.json')
if ready.exists():
 try:q['ready']=json.loads(ready.read_bytes())
 except json.JSONDecodeError:q['ready_partial_JSON']=True
p=R/'results'/('GLM-r361-'+run+'-stage3-rank0.json')
if p.exists():
 try:
  data=json.loads(p.read_bytes());tail={k:data[k]for k in ['completed','actual_step','progress','constructor_rows','completed_pairs','completed_logical_requests','normal_reset_close','normal_shutdown','normal_HTTP_shutdown','worker_shutdown_completed','service_traceback']if k in data};tail['pairs']=[]
  for pair in data.get('pairs',[]):
   item={k:pair[k]for k in ['index','enabled','warmup','complete_schedule_wall_s','decode_schedule_wall_s','decode_schedule_scope','prefill_consumption','decode_defer_fixed_both_arms']if k in pair};item['requests']=[]
   for r in pair['requests']:
    ids=r['generated_ids'];ticks=r.get('emitter_token_monotonic')or[];visible=[t for token,t in zip(ids,ticks)if token not in[154820,154827,154829]];gaps=sorted(y-x for x,y in zip(visible,visible[1:]))
    def pct(p):
     if not gaps:return None
     x=(len(gaps)-1)*p;i=int(x);return gaps[i]+(gaps[min(i+1,len(gaps)-1)]-gaps[i])*(x-i)
    admission=r.get('emitter_admission_monotonic');duration=visible[-1]-visible[0]if visible else None
    item['requests'].append({**{k:r[k]for k in ['row','fixture_id','prompt_tokens','prompt_IDs_SHA256','tokens','generated_IDs_SHA256','natural_EOS']},'backend_TTFT_s':visible[0]-admission if visible else None,'full_first_to_last_s':duration,'backend_decode_tok_s':(len(visible)-1)/duration if duration and len(visible)>1 else None,'max_ITL_s':max(gaps)if gaps else None,'P95_ITL_s':pct(.95),'P99_ITL_s':pct(.99)})
   tail['pairs'].append(item)
  q['tail']=tail
 except json.JSONDecodeError:q['tail_partial_JSON']=True
print(json.dumps(q))
