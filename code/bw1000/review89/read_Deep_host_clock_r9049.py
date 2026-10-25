import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-host-clock-r9049'
def load(p):return json.loads(p.read_bytes())if p.exists()else None
watch=load(r/(label+'-watch.json'));record=load(r/(label+'-launch.json'));p=None
if watch and watch.get('actual_handle'):p=r/'results'/(label+'-'+watch['actual_handle']+'.json')
a=subprocess.run(['sacct','-S','2026-10-25','-j','910374','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList'],capture_output=True,text=True,timeout=30)
rows=[s for s in a.stdout.splitlines()if label in s]
report=load(p)if p else None
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'watch':watch,'record':record,'sacct':rows,'report':report,'report_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()if p and p.exists()else None},ensure_ascii=False))
