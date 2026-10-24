import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-PMC-tool-r8669f';out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
p=r/(label+'-launch.json')
if p.exists():
 d=json.loads(p.read_text());out['controller']={k:d[k]for k in ['state','label','parent','node','scope','fresh_before_KFD','pid','watcher_pid','submitted_utc','returned_utc','srun_returncode','error']if k in d}
p=r/(label+'-watch.json')
if p.exists():out['watch']=json.loads(p.read_text())
out['actual_steps']=[v for v in subprocess.check_output(['sacct','-S','2026-10-24','-j','910374','-n','-P','-o','JobID,JobName%110,State%40,ExitCode,Elapsed'],text=True,timeout=30).splitlines()if label in v]
print(json.dumps(out))
