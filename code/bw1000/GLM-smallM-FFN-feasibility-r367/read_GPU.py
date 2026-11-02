"""Read the original367 handle/watch only; no load, replay or probe."""
from pathlib import Path
import json,subprocess,hashlib,datetime
R=Path.home()/'hygon-inference-refactor-20261014';q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
raw=subprocess.check_output(['sacct','-S','2026-11-01','-j','910371','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
q['accounting']=[line.split('|')for line in raw.splitlines()if line.strip()and line.split('|')[1].startswith('GLM-r367-M1-gate-layer')]
p=R/'results/GLM367-watch-r1.json'
if p.exists():
 try:q['watch']=json.loads(p.read_bytes());q['watch_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
 except json.JSONDecodeError:q['watch_partial_JSON']=True
q['log_tails']={str(p):p.read_text(errors='replace')[-2200:]for p in (R/'logs').glob('GLM*r367*.log')}
q['controller_log_tail']=(R/'logs/GLM367-controller-r1.log').read_text(errors='replace')[-2000:]
if q.get('watch',{}).get('actual_step'):
 step=q['watch']['actual_step'];q['owner_progress']=[]
 for p in(R/'results').glob('GLM367-'+step+'-*.json'):
  try:
   a=json.loads(p.read_bytes());q['owner_progress'].append(dict(path=str(p),**{k:a[k]for k in ['rank','PID','parent_PID','completed','normal','error','traceback','create_seconds','weight_load_seconds','free_after_load','measurement_calls','warm_calls']if k in a},saved_warm=len(a.get('warm',[])),saved_trials=len(a.get('trials',[]))))
  except json.JSONDecodeError:pass
print(json.dumps(q))
