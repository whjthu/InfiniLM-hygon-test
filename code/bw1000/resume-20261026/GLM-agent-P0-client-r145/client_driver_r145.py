"""Root CPU-only client; never starts/stops Native or requests a GPU."""
import datetime,hashlib,json,os,pathlib,socket,subprocess,sys
R=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname()=='a02r4n13'and os.environ['SLURM_JOB_ID']=='910376'
assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
expected=os.environ['GLM_P0_EXPECTED_HANDLE_R145'];watch=json.loads((R/'GLM-agent-service-r143b-watch.json').read_bytes());assert watch['actual_handle']==expected and watch['GPU_submitted']
run=watch['run'];assert run.startswith('GLM-agent-service-r143b-')and run.replace('-','').isalnum()
ready=R/'results'/('GLM-agent-service-r143b-ready-'+run+'.json');identity=json.loads(ready.read_bytes());assert identity['ready']and identity['actual_step']==expected and identity['context']==640 and identity['cards']==16 and(identity['TP'],identity['PP'])==(8,2)and identity['stage']==1 and identity['rank']==0 and identity['port']==43843 and identity['HTTP_bind']=='127.0.0.1'and identity['model_path']=='/old/models/ZhipuAI-GLM-5.3'and identity['Native_DSO_SHA256']=='ce830c4d77fe193e4a6e82597aa4fb5b50019c82edce542eb6d61217a238bf68'
client=pathlib.Path(os.environ.get('GLM_P0_CLIENT_PATH_R145',str(R/'incoming/shared-agent-p0-client-r143/client_p0_r143.py')));assert sha(client)=='feb872ddadfb6bdc9d1c5281608a3294b78cc949707ca86804a5100eb30c7d8a'
out=R/'results'/('GLM-agent-P0-r145-'+run);assert not out.exists();record=R/'results'/('GLM-agent-P0-client-r145-'+run+'.json');assert not record.exists()
args=[sys.executable,'-B','-u',str(client),'--base-url','http://127.0.0.1:43843/v1','--model','GLM-5.3','--max-tokens','96','--timeout','180','--identity',str(ready),'--output-dir',str(out)]
q={'CPU_only':True,'Native_start_stop':False,'GPU_handle':expected,'CPU_client_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'ready_SHA256':sha(ready),'client_SHA256':sha(client),'args':args,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'model_P0_acceptance':'REQUIRES_RAW_WIRE_AND_NATIVE_BACKEND_AUDIT'}
with record.open('x')as f:json.dump(q,f,indent=2)
try:
 p=subprocess.run(args,timeout=1800,check=False);q.update(completed=True,client_exit_code=p.returncode,result_path=str(out/'result.json'))
 if(out/'result.json').exists():q['result_SHA256']=sha(out/'result.json')
except subprocess.TimeoutExpired:q.update(client_bound_exceeded=True,error='1800s CPU client bound exceeded; do not restart Native automatically')
finally:record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
