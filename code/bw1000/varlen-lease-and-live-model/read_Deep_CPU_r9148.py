import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-25','-j','910406,910407','-n','-P','-o','JobID,JobName%100,State%30,ExitCode,Elapsed,NodeList%80'],text=True,timeout=30).splitlines()];jobs=[x for x in rows if x[1]=='Deep-eos-compaction-CPU-r9148'];out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,'logs':[]}
for j in jobs:
 p=r/'logs'/('Deep-eos-compaction-CPU-r9148-'+j[0]+'.log')
 if p.exists():out['logs'].append({'path':str(p),'tail':p.read_text(errors='replace')[-4500:]})
print(json.dumps(out))
