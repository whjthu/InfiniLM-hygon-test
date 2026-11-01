import pathlib,json,subprocess,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';p=r/'results/GLM341c-probe-launch.json';q=json.loads(p.read_text());assert q['GPU_submissions']==[]
rows=subprocess.check_output(['sacct','-S','2026-10-24','-j','910371,910374','-n','-P','-o','JobID,JobName%80,State,ExitCode,NodeList'],text=True);rows='\n'.join(s for s in rows.splitlines() if 'GLM341c' in s);assert 'GLM341c-probe' not in rows
print(json.dumps({'observed_error':"TypeError: unsupported operand type(s) for +: 'PosixPath' and 'str'",'boundary':'first srun argv construction before Popen','GPU_submitted':False,'record':q,'record_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'sacct':rows,'performance_round':False}))
