"""CPU HTTP supervision for exactly one complete GLM residency, never retries."""
import argparse,json,os,pathlib,subprocess,sys,time,hashlib
from source_pin_r290 import verify,profiles,D,sha
from client_owner_r290 import bind_identity
R=pathlib.Path('/refactor')
PHASES=[('regular','client_regular_agent.py',29),('long-output','client_long_output.py',32),('history-retention','client_history_r290.py',34),('multicall','client_multicall_r290.py',38),('quality','client_quality_r290.py',102),('paired','client_paired_r290.py',120)]

def main():
 p=argparse.ArgumentParser()
 for n in ['run','handle','identity','service-report','manifest-sha']:p.add_argument('--'+n,required=True)
 p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();record=R/'results'/('GLM-latest-full-owner-r290-supervisor-'+a.run+'.json');stop=R/'control'/('GLM-latest-full-owner-r290-stop-'+a.run+'.json');assert not record.exists()and not stop.exists();out=R/'results'/('GLM-latest-full-owner-r290-client-'+a.run);out.mkdir(exist_ok=False)
 q={'completed':False,'passed':False,'actual_GPU_handle':a.handle,'CPU_only':True,'Model':0,'no_retry_reload':True,'planned_HTTP':121,'planned_Native':120,'phases':{}}
 def save():
  t=record.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2)+chr(10));t.replace(record)
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']);mf=verify(a.manifest_sha);profiles(a.manifest_sha);ident=bind_identity(a.identity,a.manifest_sha,a.service_report);assert ident['actual_step']==a.handle and ident['run']==a.run
  for phase,filename,count in PHASES:
   left=a.deadline_unix-time.time()
   if left<=180:raise TimeoutError('Full plan deadline exhausted; do not skip, reset or reload')
   where=out/phase;args=[sys.executable,'-B',str(D/filename),'--base-url','http://127.0.0.1:43843/v1','--model','GLM-5.3','--output-dir',str(where),'--identity',a.identity,'--timeout',str(min(7200,left-120))]
   if phase in ['multicall','quality','paired']:args+=['--service-report',a.service_report,'--manifest-sha',a.manifest_sha,'--expected-step',a.handle,'--deadline-unix',str(a.deadline_unix)]
   else:
    args+=['--p0-client',str(D/'wire_client_r290.py'),'--p0-sha256',mf['files_SHA256']['wire_client_r290.py']]
    if phase=='regular':args+=['--profiles',json.loads((D/'INPUT_BINDING.json').read_bytes())['profile_manifest_path']]
    elif phase=='long-output':args+=['--tokenizer-adapter',str(D/'tokenizer_adapter_r204.py'),'--tokenizer-sha256',mf['files_SHA256']['tokenizer_adapter_r204.py']]
    else:args+=['--regular-dir',str(out/'regular'),'--service-report',a.service_report]
   q['phase_in_progress']=phase;save()
   with(out/(phase+'.log')).open('x')as log:z=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=left-120,check=False)
   result=where/'result.json';phase_result=json.loads(result.read_bytes())if result.is_file()else{};q['phases'][phase]={'exit_code':z.returncode,'result_SHA256':sha(result)if result.is_file()else None,'result':phase_result};save()
   assert z.returncode==0 and phase_result.get('completed')and phase_result.get('passed'),'Phase failed without advancing/retry: '+phase
   owner=json.loads(pathlib.Path(a.service_report).read_bytes());assert len(owner['service_requests'])==count and owner['source_manifest_SHA256']==a.manifest_sha and owner['actual_step']==a.handle
  owner=json.loads(pathlib.Path(a.service_report).read_bytes());journal=owner['queue_journal'];assert journal['max_distinct_P1_sessions_unfinished']>=2
  n=sum(v['result']['actual_HTTP_requests']for v in q['phases'].values());assert n==121;q.update(passed=True,actual_HTTP=n,actual_Native=len(owner['service_requests']),client_incremental_metrics_qualified=False,full_Model_qualification_requires_all32_normal_exit_and_postcheck=True)
 except BaseException as e:q['error']=repr(e)
 finally:
  q['completed']=True;save()
  with stop.open('x')as f:json.dump({'normal_stop':True,'passed':q['passed'],'actual_GPU_handle':a.handle,'supervisor_record_SHA256':sha(record)},f)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
