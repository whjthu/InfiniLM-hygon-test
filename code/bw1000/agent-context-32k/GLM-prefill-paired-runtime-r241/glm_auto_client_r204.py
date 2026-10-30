"""CPU-only resident supervisor: frozen regular client, no Native/Slurm/restart."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time
from service_budget_r204 import client_deadline,phase_timeout,REGULAR_PHASE_MAX,LONG_PHASE_MAX
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--handle',required=True);x=a.parse_args()
 assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 assert x.run.startswith('GLM-prefill-paired-r241-')and x.run.replace('-','').isalnum()
 ready=R/'results'/('GLM-prefill-paired-r241-ready-'+x.run+'.json');identity=json.loads(ready.read_bytes())
 assert identity['actual_step']==x.handle and(identity['TP'],identity['PP'],identity['cards'],identity['context'],identity['rows'])==(8,4,32,32768,1)
 assert identity['Native_DSO_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
 deadline=client_deadline(time.monotonic(),identity['service_deadline_monotonic'])
 stop=R/'control'/('GLM-prefill-paired-r241-stop-'+x.run+'.json');record=R/'results'/('GLM-regular-supervisor-r204-'+x.run+'.json');assert not record.exists()and not stop.exists()
 result={'completed':False,'passed':False,'CPU_only':True,'GPU_handle':x.handle,'regular_client_SHA256':sha(D/'client_regular_agent.py'),'WireClient_SHA256':sha(D/'client_GLM_regular_wire_r178a.py'),'ready_SHA256':sha(ready),'normal_agent_PASS':False,'long_parameter_extension':'NOT_CONFIGURED'}
 def save():
  p=record.with_suffix('.tmp');p.write_text(json.dumps(result,indent=2));p.replace(record)
 try:
  assert result['regular_client_SHA256']=='51218ed03190ea87b63cb06cf6c5883aae5d327a7db878109fd34c5ce4bde194'
  prep=pathlib.Path(os.environ['GLM_REGULAR_FIXTURE_RECEIPT_R180']);assert str(prep)=='/refactor/results/GLM-regular-fixture-r178a-910407.539/CPU-sizing.json'and sha(prep)==os.environ['GLM_REGULAR_FIXTURE_RECEIPT_SHA256_R180']=='ed2de5e3a0035655ae89f38c65359acf567821b5d254279eebc60847d5f89cbf';p=json.loads(prep.read_bytes());assert p['completed']and p['CPU_only']and not p['normal_agent_PASS']
  profiles=pathlib.Path(p['profiles_json']);assert sha(profiles)==p['profiles_SHA256'];result['profiles_SHA256']=sha(profiles);save()
  output=R/'results'/('GLM-regular-live-r204-'+x.run)
  result['phase_in_progress']='regular';result['shared_client_deadline_monotonic']=deadline;save()
  args=[sys.executable,'-B','-u',str(D/'client_regular_agent.py'),'--p0-client',str(D/'client_GLM_regular_wire_r178a.py'),'--p0-sha256',result['WireClient_SHA256'],'--base-url','http://127.0.0.1:43843/v1','--model','GLM-5.3','--output-dir',str(output),'--identity',str(ready),'--profiles',str(profiles),'--timeout','7200']
  z=subprocess.run(args,check=False,timeout=phase_timeout(time.monotonic(),deadline,REGULAR_PHASE_MAX));actual=output/'result.json';q=json.loads(actual.read_bytes())if actual.exists()else{}
  result.update(completed=True,regular_exit_code=z.returncode,regular_result_path=str(actual),regular_result_SHA256=sha(actual)if actual.exists()else None,passed=z.returncode==0 and q.get('passed')is True,actual_HTTP_requests=q.get('actual_HTTP_requests'))
  queue=R/'results'/('GLM-prefill-paired-r241-queue-'+x.run+'.json')
  if queue.exists():
   j=json.loads(queue.read_bytes());result['queue_journal_SHA256']=sha(queue);result['C2_entry_overlap']=j['max_distinct_P1_sessions_unfinished']>=2;result['passed']=result['passed']and result['C2_entry_overlap']
  else:result['passed']=False;result['C2_entry_overlap']=False
  regular_passed=result['passed'];longclient=D/'client_long_output.py';adapter=D/'tokenizer_adapter_r204.py'
  assert sha(longclient)=='20b9bbbc506072890df78d591fc0f22e654173f3ce4744ebf09d35ec8ce3daa9'
  longoutput=R/'results'/('GLM-long-parameter-live-r204-'+x.run)
  result['phase_in_progress']='long-output';save()
  longargs=[sys.executable,'-B','-u',str(longclient),'--p0-client',str(D/'client_GLM_regular_wire_r178a.py'),'--p0-sha256',result['WireClient_SHA256'],'--tokenizer-adapter',str(adapter),'--tokenizer-sha256',sha(adapter),'--base-url','http://127.0.0.1:43843/v1','--model','GLM-5.3','--output-dir',str(longoutput),'--identity',str(ready),'--timeout','7200']
  lz=subprocess.run(longargs,check=False,timeout=phase_timeout(time.monotonic(),deadline,LONG_PHASE_MAX));lp=longoutput/'result.json';lq=json.loads(lp.read_bytes())if lp.exists()else{}
  longpassed=lz.returncode==0 and lq.get('passed')is True
  result.update(regular_client_passed=regular_passed,long_parameter_exit_code=lz.returncode,long_parameter_client_SHA256=sha(longclient),long_parameter_tokenizer_SHA256=sha(adapter),long_parameter_result_path=str(lp),long_parameter_result_SHA256=sha(lp)if lp.exists()else None,long_parameter_passed=longpassed,long_parameter_extension='ROOT_FROZEN_EXECUTED_AFTER_REGULAR',passed=regular_passed and longpassed,client_acceptance_passed=regular_passed and longpassed,normal_agent_PASS=False)
  # Root must additionally audit all32 native normal exits, CCL and postKFD.

 except BaseException as error:
  result.update(error_class=type(error).__name__,error=str(error)[:512],passed=False,normal_agent_PASS=False)
  if isinstance(error,(TimeoutError,subprocess.TimeoutExpired)):result['deadline_exhaustion_is_failure']=True;result['long_parameter_extension']='NOT_COMPLETED_DEADLINE'if result.get('phase_in_progress')=='long-output'else'NOT_RUN_DEADLINE'
 finally:
  save()
  with stop.open('x')as f:json.dump({'normal_stop':True,'reason':'regular_CPU_client_finished_or_failed','GPU_handle':x.handle,'record_SHA256':sha(record),'regular_passed':result['passed']},f)
 print(json.dumps({k:result[k]for k in ['completed','passed','normal_agent_PASS']}));return 0 if result['passed']else 1
if __name__=='__main__':raise SystemExit(main())
