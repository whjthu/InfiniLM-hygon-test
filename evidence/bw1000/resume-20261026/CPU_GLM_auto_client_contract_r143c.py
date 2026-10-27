"""CPU orchestration contract using fake child results, not HTTP/model evidence."""
import importlib.util,json,pathlib,subprocess,tempfile,hashlib
V=pathlib.Path(__file__).resolve().parent;D=V/'glm-agent-service-p0-r143c'
s=importlib.util.spec_from_file_location('auto_client',D/'glm_auto_client_r143c.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
checks=[]
with tempfile.TemporaryDirectory()as raw:
 root=pathlib.Path(raw)
 for case in ['passed','failed','timeout','missing_result']:
  p=root/case;p.mkdir();ready=p/'ready.json';stop=p/'stop.json';record=p/'supervisor.json';out=p/'out'
  ready.write_text(json.dumps({'ready':True,'actual_step':'CPU-FAKE-NOT-GPU','context':640,'cards':16,'model_path':'/old/models/ZhipuAI-GLM-5.3','Native_DSO_SHA256':'ce830c4d77fe193e4a6e82597aa4fb5b50019c82edce542eb6d61217a238bf68'}))
  def runner(args,timeout,check):
   assert timeout==1800 and check is False and'--timeout'in args and args[args.index('--timeout')+1]=='180'
   if case=='timeout':raise subprocess.TimeoutExpired(args,timeout)
   if case!='missing_result':out.mkdir();(out/'result.json').write_text(json.dumps({'completed':True,'passed':case=='passed','actual_HTTP_requests':0,'CPU_fake_result_only':True}))
   return type('Exit',(),{'returncode':0 if case in ['passed','missing_result']else 1})()
  q=m.run_client(D/'client_p0_r143.py',ready,out,stop,record,'CPU-FAKE-NOT-GPU',runner=runner)
  assert stop.exists()and record.exists()and json.loads(stop.read_text())['normal_stop']is True
  assert q['passed']is(case=='passed')
  checks.append(case+'_always_stops_no_false_pass')
o={'passed':True,'CPU_fake_runner_only':True,'actual_HTTP':False,'Native':False,'GPU':False,'model':False,'checks':checks,'auto_client_SHA256':hashlib.sha256((D/'glm_auto_client_r143c.py').read_bytes()).hexdigest()}
(V/'CPU-GLM-auto-client-contract-r143c.json').write_text(json.dumps(o,indent=2));print(json.dumps(o))
