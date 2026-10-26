import pathlib,json,subprocess,hashlib,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
for rel in ['results/Deep-inner-keeper-launch-r9193b.json','results/Deep-owner-prefill-control-r9193b.json']:
 p=r/rel
 if p.exists():q['files'][rel]={'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_bytes())}
q['Slurm']=[]
for job in ['910407','910374']:
 q['Slurm'] +=[l for l in subprocess.check_output(['sacct','-S','2026-10-25','-j',job,'-n','-P','-o','JobIDRaw,JobName%100,State,ExitCode,Elapsed,NodeList%80'],text=True).splitlines()if'r9193b'in l]
q['logs']={p.name:p.read_text(errors='replace')[-3500:]for p in(r/'logs').glob('*r9193b*log')if'CPU-r9193b'not in p.name};print(json.dumps(q))
