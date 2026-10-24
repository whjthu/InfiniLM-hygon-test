"""Bound only this tool's trace directory and process group; never parent hold."""
import json,os,pathlib,signal,subprocess,sys,time
d=pathlib.Path(sys.argv[1]);assert d.name.startswith('glm-spawn-trace-r8692-')and d.parent==pathlib.Path('/local')
p=subprocess.Popen(sys.argv[2:],start_new_session=True);deadline=time.monotonic()+100;peak=0;reason=None
try:
 while p.poll()is None:
  total=sum(f.stat().st_size for f in d.rglob('*')if f.is_file());peak=max(peak,total)
  if total>256*1024**2:reason='trace_directory_budget_exceeded';break
  if time.monotonic()>deadline:reason='own_tool_100s_timeout';break
  time.sleep(.2)
finally:
 if p.poll()is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
q={'own_tool_pid':p.pid,'exit':p.returncode,'trace_peak_observed_bytes':peak,'stop_reason':reason,'global_RLIMIT_FSIZE_added':False}
(d/'directory-budget.json').write_text(json.dumps(q));print(json.dumps(q),flush=True)
assert reason is None and p.returncode==0,q
root=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
for rank in range(8):
 f=root/'results'/f'GLM-spawn-trace-r8692-{tag}-rank{rank}.json';owner=json.loads(f.read_text());assert owner['completed']and owner['start_code']==owner['stop_code']==0 and owner['sum']==120
