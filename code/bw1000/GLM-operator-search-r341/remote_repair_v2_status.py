import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';q={}
q['sacct']=subprocess.check_output(['sacct','-S','2026-10-24','-j','910371,910374','-n','-P','-o','JobID,JobName%80,State,ExitCode,NodeList'],text=True)
q['sacct']='\n'.join(s for s in q['sacct'].splitlines() if 'GLM341c' in s)
q['probes']=[json.loads(p.read_text()) for p in (r/'results').glob('GLM341c-map-*-rank*.json')]
q['results']=[]
for p in (r/'results').glob('GLM341-*-rank*.json'):
 if any(s in p.name for s in ['910371.613','910374.1231']):continue
 z=json.loads(p.read_text());q['results'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),**{k:z.get(k) for k in ['actual_step','node','fusion_variant','completed','normal','error','performance']}})
q['logs']={str(p):p.read_text(errors='replace')[-2500:] for p in (r/'logs').glob('GLM341c-*.log') if any(s in p.read_text(errors='replace') for s in ['Traceback','Error','error','NORMAL'])}
print(json.dumps(q))

