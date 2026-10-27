"""One-card native standard-server test. No mocked/generated model responses."""
import pathlib,json,hashlib,subprocess,os,sys,time,urllib.request,urllib.error,signal,datetime,traceback
R=pathlib.Path('/refactor');D=R/'incoming/toolcall-validation-GPU-r139';S=R/'incoming/toolcall-validation-CPU-r138';B=R/'candidates/toolcall-Qwen06-r138';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];result=R/'results'/('toolcall-Qwen06-r139-'+step+'.json');assert not result.exists()
m=json.loads((D/'GPU-source-manifest.json').read_bytes());go=json.loads((D/'root-GO-after-CPU.json').read_bytes());cp=B/'CPU-preflight.json';assert sha(cp)==go['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['CPU_only']and c['static_only']and c['static_overlay_AST_passed']and not c['standard_server_imported']and not c['GPU_initialized']and c['source_manifest_SHA256']==m['CPU_source_manifest_SHA256']
for p,h in c['baseline_files_SHA256'].items():assert sha(p)==h,p
for p,h in c['overlay_files_SHA256'].items():assert sha(p)==h,p
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_step':step,'completed':False,'passed':False,'model':'Qwen3-0.6B','cards':1,'TP':1,'PP':1,'native_backend':'standard AsyncLLMEngine','parser':'hermes','CPU_receipt_SHA256':sha(cp),'source_manifest_SHA256':sha(D/'GPU-source-manifest.json'),'protocol_commit':m['protocol_commit'],'publication_commit':go['publication_commit'],'three_large_models_qualified':False,'performance_benchmark':False,'service_persistent':False};server=None;client=None
def save():result.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
try:
 # Import/tokenizer/native checks require the allocated card. A separate
 # short-lived child exits before weights load, avoiding a retained parent CUDA
 # context and preserving fatal native import failures as real GPU evidence.
 smoke_path=R/'results'/('toolcall-Qwen06-r139-'+step+'-preweight.json')
 smoke_log=R/'logs'/('toolcall-Qwen06-r139-'+step+'-preweight.log')
 with smoke_log.open('x')as f:smoke=subprocess.run([sys.executable,'-B','-u',str(D/'GPU_preweight_import_r139.py'),str(smoke_path)],stdout=f,stderr=subprocess.STDOUT,timeout=180)
 q['preweight_smoke_exit']=smoke.returncode;q['preweight_smoke_path']=str(smoke_path);q['preweight_smoke_log']=str(smoke_log);save()
 assert smoke.returncode==0 and smoke_path.is_file(),'GPU preweight import failed; no model load started'
 sm=json.loads(smoke_path.read_bytes());assert sm['passed']and sm['actual_step']==step and sm['logical_visible_cards']==1 and not sm['model_constructed']and sm['native_modules']
 q['preweight_smoke_SHA256']=sha(smoke_path);q['hardware']=sm['hardware'];q['fresh_free_bytes']=sm['free_bytes'];q['runtime_import_qualified']=True;q['CPU_static_qualification_only']=True
 # The server's load, native inference, parser and client run unchanged in separate
 # processes. All responses below are received from the actual HTTP listener.
 base='http://127.0.0.1:43839/v1';q['node_local_service_url']=base;q['service_started_monotonic']=time.monotonic();identity=R/'results'/('toolcall-Qwen06-r139-'+step+'-identity.json');identity.write_text(json.dumps({'source_manifest_SHA256':q['source_manifest_SHA256'],'CPU_receipt_SHA256':sha(cp),'native_modules':sm['native_modules'],'checkpoint_metadata':c['checkpoint_metadata'],'model_directory':'/shared/Qwen3-0.6B','protocol_commit':m['protocol_commit'],'publication_commit':go['publication_commit'],'actual_step':step,'cards':1,'TP':1,'PP':1},indent=2))
 cmd=[sys.executable,'-B','-u','-m','infinilm.server.inference_server','--model','/shared/Qwen3-0.6B','--device','hygon','--dtype','bfloat16','--tp','1','--pp','1','--max-batch-size','1','--max-cache-len','8192','--max-new-tokens','512','--disable-prefix-caching','--tool-call-parser','hermes','--host','127.0.0.1','--port','43839']
 log=R/'logs'/('toolcall-Qwen06-r139-'+step+'-server.log');q['server_log']=str(log)
 with log.open('x')as f:server=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 q['owned_server_PID']=server.pid;save();deadline=time.monotonic()+180
 while True:
  assert server.poll()is None,('native server exited during startup',server.returncode)
  try:
   with urllib.request.urlopen('http://127.0.0.1:43839/health',timeout=2)as response:
    health=json.loads(response.read());assert response.status==200;q['health']=health;break
  except (urllib.error.URLError,TimeoutError,ConnectionError):pass
  assert time.monotonic()<deadline,'native server startup deadline';time.sleep(1)
 q['ready_after_s']=time.monotonic()-q['service_started_monotonic']
 # Record only native/code mappings of our own server. No raw environment/argv.
 maps=pathlib.Path('/proc')/str(server.pid)/'maps';paths=set()
 for line in maps.read_text().splitlines():
  p=line.split()[-1]
  if p.startswith('/refactor/')and '.so'in pathlib.Path(p).name:paths.add(p)
 q['actual_server_DSO_SHA256']={p:sha(p)for p in sorted(paths)}
 evidence=R/'results'/('toolcall-Qwen06-r139-'+step+'-client');q['client_evidence_directory']=str(evidence);save()
 client=subprocess.run([sys.executable,'-B','-u',str(D/'validate_tools_http_r139.py'),'--base-url',base,'--model','Qwen3-0.6B','--output-dir',str(evidence),'--identity',str(identity),'--timeout','100','--max-tokens','512'],capture_output=True,text=True,timeout=900)
 q['client_exit']=client.returncode;q['client_stdout']=client.stdout[-2000:];q['client_stderr']=client.stderr[-4000:];p=evidence/'result.json';q['client_result_SHA256']=sha(p)if p.exists()else None
 assert client.returncode==0,'real client validation failed; preserve actual request/response artifacts'
 cr=json.loads(p.read_bytes());assert cr['completed']and cr['passed']and len(cr['cases'])==6
 q['cases']=[{'choice_mode':x['choice_mode'],'stream':x['stream'],'passed':x['passed'],'tools_executed':len(x['calls'])}for x in cr['cases']];q['actual_real_tool_executions']=sum(len(x['calls'])for x in cr['cases'])
 # Reject invalid history before generation, then prove later ordinary requests
 # still work. A one-token truncation is never executed as a tool call.
 invalid={'model':'Qwen3-0.6B','messages':[{'role':'tool','tool_call_id':'missing-call','content':'unmatched'}],'tools':[],'max_tokens':1}
 try:urllib.request.urlopen(urllib.request.Request(base+'/chat/completions',data=json.dumps(invalid).encode(),headers={'Content-Type':'application/json'}),timeout=20)
 except urllib.error.HTTPError as e:q['invalid_history_HTTP']=e.code;q['invalid_history_response_SHA256']=hashlib.sha256(e.read()).hexdigest();assert e.code==400
 else:raise AssertionError('invalid tool history accepted')
 short={'model':'Qwen3-0.6B','messages':[{'role':'user','content':'Write ten numbers in order.'}],'tools':[],'tool_choice':'none','max_tokens':1,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
 with urllib.request.urlopen(urllib.request.Request(base+'/chat/completions',data=json.dumps(short).encode(),headers={'Content-Type':'application/json'}),timeout=100)as response:short_reply=json.loads(response.read())
 q['truncation_response']=short_reply;assert short_reply['choices'][0]['finish_reason']=='length'and not short_reply['choices'][0]['message'].get('tool_calls')
 short['max_tokens']=32;short['messages']=[{'role':'user','content':'Reply exactly OK.'}]
 with urllib.request.urlopen(urllib.request.Request(base+'/chat/completions',data=json.dumps(short).encode(),headers={'Content-Type':'application/json'}),timeout=100)as response:recovered=json.loads(response.read())
 q['recovery_response']=recovered;assert recovered['choices'][0]['finish_reason']=='stop'and 'OK'in recovered['choices'][0]['message'].get('content','')
 q.update(completed=True,passed=True,scope='Qwen3/hermes standard HTTP+SSE named/auto/required and none-final real sandbox closure; invalid history/truncation/recovery; does not qualify K3/DSV4/GLM or production service')
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if server is not None:
  if server.poll()is None:
   server.terminate()
   try:server.wait(timeout=40)
   except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10);q['server_forced_kill']=True
  q['server_exit']=server.returncode;q['normal_server_shutdown']=server.returncode==0 and not q.get('server_forced_kill',False)
  if not q['normal_server_shutdown']:q['passed']=False
 q['post_free_bytes']=None;q['post_memory_scope']='CPUkeeper verifies KFD after this GPU job exits';save()
print(json.dumps({k:q.get(k)for k in ['completed','passed','error','normal_server_shutdown','actual_real_tool_executions']}));assert q['passed']
