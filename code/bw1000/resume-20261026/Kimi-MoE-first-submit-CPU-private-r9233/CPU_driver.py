import pathlib,subprocess,json,os
D=pathlib.Path(__file__).parent
for p in D.iterdir():
 if p.suffix in ['.py','.cpp','.hpp']:assert bytes([13])not in p.read_bytes()
z=subprocess.run(['c++','-std=c++17','-pthread',str(D/'CPU_contract.cpp'),'-o',str(D/'CPU_contract')],text=True,capture_output=True,timeout=45);assert z.returncode==0,z.stderr
r=subprocess.run([str(D/'CPU_contract')],text=True,capture_output=True,timeout=20);assert r.returncode==0,(r.returncode,r.stdout,r.stderr)
print(json.dumps({'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':os.uname().nodename,'contract':json.loads(r.stdout),'GPU':False}),flush=True)
