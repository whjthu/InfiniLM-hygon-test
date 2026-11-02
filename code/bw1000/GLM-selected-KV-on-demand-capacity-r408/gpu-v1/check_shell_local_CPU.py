"""Actual Bash required-env prefix; only hostname is a recording leaf."""
from pathlib import Path
import os,subprocess,json,shutil
def run():
 D=Path(__file__).resolve().parent;bash=Path('C:/Program Files/Git/bin/bash.exe')if os.name=='nt'else Path(shutil.which('bash'))
 for p in D.glob('*.sh'):assert b'\r'not in p.read_bytes();subprocess.run([str(bash),'-n',str(p)],capture_output=True,check=True)
 prefix=(D/'GPU_inside.sh').read_text().split('source /opt/dtk/env.sh',1)[0];program='hostname(){ printf a02r3n16; };\n'+prefix+'\nprintf REQUIRED_ENV_OK\n';results=[]
 for missing in [None,'GLM408_GPU_SOURCE_SHA256','GLM408_GPU_START']:
  env=dict(os.environ,SLURM_JOB_ID='910371',GLM408_GPU_SOURCE_SHA256='a'*64,GLM408_GPU_START='10000.0')
  if missing:env.pop(missing,None)
  p=subprocess.run([str(bash),'-s'],input=program,capture_output=True,text=True,env=env);assert(p.returncode==0)==(missing is None)
  results.append(dict(missing=missing,rejected=p.returncode!=0))
 return dict(passed=True,actual_Bash_required_env_prefix=results,shell_syntax=True,CR_bytes=0,Native_dlopen=False,GPU=0,Slurm=0)
if __name__=='__main__':print(json.dumps(run()))
