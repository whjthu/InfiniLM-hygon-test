import pathlib,json,hashlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-guard-r9090';q={}
for n in [label+'-launch.json',label+'-watch.json','results/'+label+'-910374.1030.json']:
 p=r/n
 if p.exists():q[n]={'data':json.loads(p.read_bytes()),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
q['sacct']=subprocess.check_output(['sacct','-j','910374.1030,910374.1031','-n','-P','-o','JobID,JobName%90,State,ExitCode,Elapsed,NodeList%80'],text=True)
print(json.dumps(q))
