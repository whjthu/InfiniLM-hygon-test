"""One resident32K Native owner, real HTTP overlap with FIFO serial execution."""
import pathlib,json,hashlib,os,time,threading,subprocess,sys,queue,copy
from request_contract_r182 import raw_budget,check_sampling,install_pre_sse_guard
from native_binding_r182 import binding
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-agent-regular32K-r182';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def owner_threads(ns):
 try:
  owners=ns['verify_owner_affinity'](ns['lib'])['actual_owner_report']['owners'];rows=[];unknown=[]
  for owner in owners:
   mask={int(x)for x in owner['verified_mask'].split(',')};matches=[]
   for p in pathlib.Path('/proc/self/task').iterdir():
    try:
     if set(os.sched_getaffinity(int(p.name)))==mask:matches.append(int(p.name))
    except ProcessLookupError:pass
   if len(matches)==1:rows.append({'device':owner['device'],'tid':matches[0],'basis':'unique actual verified owner CPU mask'})
   else:unknown.append({'device':owner['device'],'matching_tasks':len(matches)})
  return {'status':'OBSERVED'if len(rows)==8 else'UNKNOWN','owners':rows,'ambiguous':unknown,'not_Native_per_owner_callback':True}
 except Exception as exc:return {'status':'UNKNOWN','error':repr(exc),'owners':[]}
def serve(ns):
 from native_engine_bridge_r143c import NativeEngineBridge,existing_server_app
 from native_decode_r182 import generate
 from chat_prepare_r173 import prepare
 from infinilm.llm import FinishReason
 import infinilm.server.inference_server as server_module
 import uvicorn
 b,c,n=binding();tokenizer=ns['tokenizer'];tag=ns['tag'];torch=ns['torch'];np=ns['np'];lib=ns['lib'];ctx=ns['ctx'];check=ns['check'];port=43882
 class Bridge(NativeEngineBridge):
  def validate_data(self,data,sampling_params):
   check_sampling(data,sampling_params);return super().validate_data(data,sampling_params)
  def add_chat_request(self,**kwargs):
   data=kwargs.get('request_data')or{};budget,proof=check_sampling(data,kwargs['sampling_params']);req=super().add_chat_request(**kwargs);assert type(req.max_new_tokens)is int and req.max_new_tokens==budget
   req.metadata.update(raw_budget_contract=proof,diagnostic_user=data.get('user'),pending_at_admission=len(self._requests),chat_template_kwargs=copy.deepcopy(data.get('chat_template_kwargs')or{}),real_cache_rows=1,real_cache_capacity=32768,no_output_clamp=True)
   return req
 def encode(messages,data,generation,kwargs):
  budget,_=raw_budget(data);return prepare(messages,data,generation,kwargs,tokenizer,32768,budget)
 engine=Bridge(tokenizer,32768,FinishReason,encode_chat=encode,eos_token_ids=(1,),default_max_tokens=4096,max_pending=8)
 # These are explicit bounded validation settings, separate from smoke600/1800.
 assert pathlib.Path(server_module.__file__).resolve().is_relative_to(B/'python')
 before={'stream':server_module.DEFAULT_STREAM_TIMEOUT,'request':server_module.DEFAULT_REQUEST_TIMEOUT};server_module.DEFAULT_STREAM_TIMEOUT=1800.0;server_module.DEFAULT_REQUEST_TIMEOUT=7200.0
 service,app=existing_server_app(engine,str(ns['model']),'deepseek_v4',host='127.0.0.1',port=port);budget_proof=install_pre_sse_guard(service)
 server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='info'));server.install_signal_handlers=lambda:None;thread=threading.Thread(target=server.run,name='Deep-regular32K-r182',daemon=False)
 root=R/'results'/('Deep-agent-regular32K-r182-'+tag);assert not root.exists();root.mkdir();result=R/'results'/('Deep-agent-regular32K-r182-'+tag+'.json');identity=root/'identity.json'
 report={'completed':False,'passed':False,'actual_step':tag,'Native_path':c['library'],'Native_SHA256':c['library_SHA256'],'new_Native_CPU_receipt':b,'model_directory':str(ns['model']),'TP':8,'PP':1,'cards':8,'rows':1,'capacity':32768,'prefill_chunk':1024,'default_output_budget':4096,'thinking_budget':8192,'raw_integer_budget_contract':budget_proof,'validation_timeouts':{'original':before,'stream':1800,'request':7200,'progress':'Only actual model tokens; absolute deadline unchanged'},'reasoning_policy':'Preserve complete messages/reasoning and explicit requested effort; otherwise native encoder default, no smoke low effort','HTTP_URL':'http://127.0.0.1:'+str(port)+'/v1','backend_scheduling':'FIFO serial Native worker; HTTP overlap is not Native batch','bootstrap':ns['__bootstrap_proof_r182__'],'provider_preweight':ns['__provider_proof_r182__'],'actual_preload_create':ns['__create_proof_r182__'],'requests':[],'normal_server_shutdown':False,'normal_Native_shutdown':False,'new_performance_claim':False}
 def save():
  temp=result.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(result)
 ns['__owner_exit_observation_r182__']=owner_threads(ns);identity.write_text(json.dumps({k:v for k,v in report.items()if k!='requests'},ensure_ascii=False,indent=2));save();thread.start();client=None;log=None;deadline=time.monotonic()+14400
 def start_phase(phase):
  where=root/phase;args=[sys.executable,'-B',str(D/('client_regular_agent.py'if phase=='regular'else'client_long_output.py')),'--p0-client',str(D/'wire_client_r182.py'),'--p0-sha256',sha(D/'wire_client_r182.py'),'--base-url',report['HTTP_URL'],'--model',pathlib.Path(ns['model']).name,'--output-dir',str(where),'--identity',str(identity),'--timeout','7200']
  if phase=='regular':args+=['--profiles',str(B/'profiles/profiles.json')]
  else:args+=['--tokenizer-adapter',str(D/'count_tokens_r182.py'),'--tokenizer-sha256',sha(D/'count_tokens_r182.py')]
  handle=(root/(phase+'.log')).open('x');return subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT),handle,where
 try:
  while not server.started:
   if not thread.is_alive()or time.monotonic()>deadline:raise RuntimeError('HTTP server failed to start')
   time.sleep(.05)
  phase='regular';client,log,where=start_phase(phase);outputs={'regular':where};report['server_ready']=True;save()
  while time.monotonic()<deadline:
   if client.poll()is not None:
    report.setdefault('client_returncodes',{})[phase]=client.returncode;log.close();save()
    if phase=='regular':phase='long-output';client,log,where=start_phase(phase);outputs[phase]=where;continue
    break
   if not thread.is_alive():raise RuntimeError('HTTP server stopped unexpectedly')
   try:req=engine.take(timeout=.1)
   except queue.Empty:continue
   if req is None:raise RuntimeError('Bridge stopped before clients completed')
   meta={'request_id':req.request_id,'phase':phase,'prompt_tokens':len(req.prompt_token_ids),'max_new_tokens':req.max_new_tokens,'admission':copy.deepcopy(req.metadata),'started_monotonic':time.monotonic(),'normal_reset_before':False,'normal_reset_after':False};report['requests'].append(meta);save();reason=FinishReason.LENGTH;failure=None
   def finite(logits):
    row=(logits[0].astype(np.uint32)<<16).view(np.float32)
    if not np.isfinite(row).all():raise RuntimeError('Nonfinite actual Native logits')
   try:
    check(lib.modern_native_engine_reset(ctx));meta['normal_reset_before']=True
    reason,trace=generate(req,ns['forward'],check,finite,FinishReason,context=32768,chunk_size=1024,eos=1,vocab=ns['cfg']['vocab_size']);meta['Native_continuation']=trace
   except BaseException as exc:failure=exc;meta['error']=repr(exc)
   finally:
    try:check(lib.modern_native_engine_reset(ctx));meta['normal_reset_after']=True
    except BaseException as exc:failure=exc;meta['reset_error']=repr(exc);engine._fatal=repr(exc)
    meta.update(ended_monotonic=time.monotonic(),native_finish_reason=getattr(reason,'value',str(reason)),actual_generated_tokens=len(req.generated_token_ids),generated_token_ids=list(req.generated_token_ids),GPU_free_all8=[list(torch.cuda.mem_get_info(i))for i in range(8)]);save()
   if failure is None:req.finish(reason)
   else:req.fail(failure)
   meta['transport_finish_reason']=getattr(req.finish_reason,'value',str(req.finish_reason));meta['final_metadata']=copy.deepcopy(req.metadata);save()
   if engine._fatal:raise RuntimeError('Normal Native reset failed; worker unhealthy')
  else:raise TimeoutError('Formal service14400s hard deadline; no automatic retry')
  report['phases']={}
  for name,where in outputs.items():
   p=where/'result.json';report['phases'][name]={'path':str(p),'SHA256':sha(p),'result':json.loads(p.read_bytes())}
  report.update(completed=all(v['result'].get('completed')is True for v in report['phases'].values()),passed=all(v['result'].get('passed')is True for v in report['phases'].values()));save()
 finally:
  if client is not None and client.poll()is None:client.terminate();client.wait(timeout=30)
  server.should_exit=True;thread.join(timeout=30);engine.stop();report['normal_server_shutdown']=not thread.is_alive()
  if log is not None and not log.closed:log.close()
  save()
 assert report['normal_server_shutdown'],'HTTP did not stop normally'
