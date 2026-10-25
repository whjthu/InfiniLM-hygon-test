import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-eos-compaction-r9148';o={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name in [label+'-supervisor.json',label+'-launch.json',label+'-watch.json']:
 p=r/name
 if p.exists():
  d=json.loads(p.read_bytes());o[name]={k:v for k,v in d.items()if k not in ['probe_stdout','probe_stderr','launcher_stdout','launcher_stderr','error_warning_lines','CPU_result_audit_stdout','CPU_result_audit_stderr']}
rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910374,910406','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30).splitlines()];o['jobs']=[x for x in rows if x[1]in [label,'Deep-eos-supervisor-r9148','Deep-eos-affinity-r9148',label+'-before',label+'-post']]
print(json.dumps(o))
