import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';q={}
q['sacct']=subprocess.run(['sacct','-S','2026-10-24','-j','910371,910374','-n','-P','-o','JobID,JobName%70,State,ExitCode,NodeList'],capture_output=True,text=True).stdout
q['sacct']='\n'.join(s for s in q['sacct'].splitlines() if 'GLM341' in s)
q['results']=[]
for f in (r/'results').glob('GLM341-*-rank*.json'):
 z=json.loads(f.read_text());q['results'].append({'path':str(f),'SHA256':hashlib.sha256(f.read_bytes()).hexdigest(),**{k:z.get(k) for k in ['actual_step','node','fusion_variant','completed','normal','error','performance','performance_admission_passed']}})
q['logs']={str(f):f.read_text(errors='replace')[-2500:] for f in (r/'logs').glob('GLM341-GPU*.log') if any(k in f.read_text(errors='replace') for k in ['Traceback','Error','error','NORMAL'])}
print(json.dumps(q))
