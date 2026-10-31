"""One resident32K Native owner, real HTTP overlap with FIFO serial execution."""
import pathlib,json,hashlib,os,time,threading,subprocess,sys,queue,copy
from request_contract_r206b import raw_budget,check_sampling,install_pre_sse_guard
from native_binding_r206b import binding
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/refactor');B=R/'candidates/Deep-agent-regular32K-r206b';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def admission_snapshot(engine,req):
 # take() is outside this lock. No Native operation or queue wait is allowed here.
 with engine._admission_lock:
  if not getattr(req,'_admission_metadata_ready',False):raise RuntimeError('Admission metadata assembly failed; request must not enter Native')
  return copy.deepcopy(req.metadata)
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
 from native_decode_r206b import generate
 import infinilm.server.inference_server as server_module
 import uvicorn
 b,c,n=binding();prepared=ns['__latest_prepared_r277__'];tokenizer=prepared['tokenizer'];tag=ns['tag'];torch=ns['torch'];np=ns['np'];lib=ns['lib'];ctx=ns['ctx'];check=ns['check'];port=43882
 engine=prepared['engine'];FinishReason=prepared['FinishReason'];service=prepared['server'];app=prepared['app']
 assert pathlib.Path(server_module.__file__).resolve().is_relative_to(R/'candidates/latest-family-service-r270/Deep/python')
 assert prepared['identity']['upstream_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'and prepared['identity']['eos_token_ids']==[1]
 before={'stream':server_module.DEFAULT_STREAM_TIMEOUT,'request':server_module.DEFAULT_REQUEST_TIMEOUT};budget_proof={'CPU604_full_history_prequeue':True,'integer_budget_1_to8192_no_clamp':True,'source_manifest_SHA256':'81468549eec894ccc2eb1144826ce0cb27eb5d3ca5a57f703b7b31a715162a14'}
 server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='info'));server.install_signal_handlers=lambda:None;thread=threading.Thread(target=server.run,name='Deep-regular32K-r206b',daemon=False)
 root=R/'results'/('Deep-latest-full-owner-r281-'+tag);assert not root.exists();root.mkdir();result=R/'results'/('Deep-latest-full-owner-r281-'+tag+'.json');identity=root/'identity.json'
 report={'completed':False,'passed':False,'actual_step':tag,'Native_path':c['library'],'Native_SHA256':c['library_SHA256'],'new_Native_CPU_receipt':b,'model_directory':str(ns['model']),'TP':8,'PP':1,'cards':8,'rows':1,'capacity':32768,'prefill_chunk':1024,'default_output_budget':4096,'thinking_budget':8192,'raw_integer_budget_contract':budget_proof,'validation_timeouts':{'original':before,'stream':1800,'request':7200,'progress':'Only actual model tokens; absolute deadline unchanged'},'reasoning_policy':'Preserve complete messages/reasoning and explicit requested effort; otherwise native encoder default, no smoke low effort','HTTP_URL':'http://127.0.0.1:'+str(port)+'/v1','backend_scheduling':'FIFO serial Native worker; HTTP overlap is not Native batch','bootstrap':ns['__bootstrap_proof_r206b__'],'provider_preweight':ns['__provider_proof_r206b__'],'actual_preload_create':ns['__create_proof_r206b__'],'requests':[],'normal_server_shutdown':False,'normal_Native_shutdown':False,'new_performance_claim':False}
 report['history_policy_scope']='Latest99 processor with CPU604 complete RC/content contribution and real pending-prefix prequeue proof; actual Model qualification pending';report['prepared_entry_identity']=prepared['identity']
 report['fixed_frozen_regular_then_long_output']=True;report['additional_history_continuations']=2;report['source_manifest_SHA256']=os.environ['DEEP_SOURCE_MANIFEST_SHA'];report['plan']={'HTTP':103,'Native':102,'phase_Native_end':{'regular':29,'long-output':32,'history-retention':34,'multicall':38,'quality':102}};report['Model_qualification_scope']='Full normal plus actualmulticall and two fixed32coarse pilots only; pairedno-regression, fullbenchmark, incrementalclientSLO and physicalmaximum unproven'
 def save():
  temp=result.with_suffix('.tmp');temp.write_text(json.dumps(report,ensure_ascii=False,indent=2));temp.replace(result)
 ns['__owner_exit_observation_r206b__']=owner_threads(ns);identity.write_text(json.dumps({k:v for k,v in report.items()if k!='requests'},ensure_ascii=False,indent=2));save();thread.start();client=None;log=None;deadline=time.monotonic()+max(0,float(os.environ['DEEP_R277_GPUSTART_UNIX'])+27720-time.time());assert deadline>time.monotonic()+120
 phase_order=['regular','long-output','history-retention','multicall','quality']
 def start_phase(phase):
  where=root/phase;script={'regular':'client_regular_agent.py','long-output':'client_long_output.py','history-retention':'client_history_r206b.py','multicall':'client_multicall_r281.py','quality':'client_quality_r281.py'}[phase];left=deadline-time.monotonic();assert left>120;args=[sys.executable,'-B',str(D/script),'--base-url',report['HTTP_URL'],'--model',pathlib.Path(ns['model']).name,'--output-dir',str(where),'--identity',str(identity),'--timeout',str(min(7200,left-60))]
  if phase in ['multicall','quality']:args+=['--service-report',str(result),'--expected-step',str(tag),'--manifest-sha',os.environ['DEEP_SOURCE_MANIFEST_SHA'],'--deadline-unix',str(float(os.environ['DEEP_R277_GPUSTART_UNIX'])+27720)]
  else:
   source=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());args+=['--p0-client',str(D/'wire_client_r206b.py'),'--p0-sha256',source['files_SHA256']['wire_client_r206b.py']]
   if phase=='regular':args+=['--profiles',str(B/'profiles/profiles.json')]
   elif phase=='long-output':args+=['--tokenizer-adapter',str(D/'count_tokens_r206b.py'),'--tokenizer-sha256',source['files_SHA256']['count_tokens_r206b.py']]
   else:args+=['--regular-dir',str(root/'regular'),'--service-report',str(result)]
  handle=(root/(phase+'.log')).open('x');return subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT),handle,where
 try:
  while not server.started:
   if not thread.is_alive()or time.monotonic()>deadline:raise RuntimeError('HTTP server failed to start')
   time.sleep(.05)
  phase='regular';client,log,where=start_phase(phase);outputs={'regular':where};report['server_ready']=True;save()
  while time.monotonic()<deadline:
   if client.poll()is not None:
    report.setdefault('client_returncodes',{})[phase]=client.returncode;log.close();save();phase_result=json.loads((where/'result.json').read_bytes());assert client.returncode==0 and phase_result.get('completed')and phase_result.get('passed'),'Phase failed; stop without advancing or retry'
    assert len(report['requests'])==report['plan']['phase_Native_end'][phase],'Actual phase Native count differs from frozen full plan'
    next_phase=phase_order.index(phase)+1
    if next_phase<len(phase_order):phase=phase_order[next_phase];client,log,where=start_phase(phase);outputs[phase]=where;continue
    break
   if not thread.is_alive():raise RuntimeError('HTTP server stopped unexpectedly')
   try:req=engine.take(timeout=.1)
   except queue.Empty:continue
   if req is None:raise RuntimeError('Bridge stopped before clients completed')
   if time.monotonic()>=deadline:raise TimeoutError('Original residency deadline reached before Native admission')
   admission=admission_snapshot(engine,req)
   assert len(report['requests'])<102,'Frozen102Native plan exhausted; no extra request'
   meta={'request_id':req.request_id,'phase':phase,'admitted_index':len(report['requests']),'prompt_tokens':len(req.prompt_token_ids),'max_new_tokens':req.max_new_tokens,'admission':admission,'started_monotonic':time.monotonic(),'normal_reset_before':False,'normal_reset_after':False,'backend_token_monotonic':[],'Native_forward_host_summary':{'prefill_calls':0,'prefill_s':0.,'decode_calls':0,'decode_s':0.}};report['requests'].append(meta);save();reason=FinishReason.LENGTH;failure=None
   def finite(logits):
    row=(logits[0].astype(np.uint32)<<16).view(np.float32)
    if not np.isfinite(row).all():raise RuntimeError('Nonfinite actual Native logits')
   try:
    check(lib.modern_native_engine_reset(ctx));meta['normal_reset_before']=True
    actual_emit=req.emit_token
    def recorded_emit(token):
     tick=time.monotonic();actual_emit(token);meta['backend_token_monotonic'].append(tick)
    req.emit_token=recorded_emit
    def recorded_forward(*args):
     kind='decode'if meta['backend_token_monotonic']else'prefill';start=time.monotonic()
     try:return ns['forward'](*args)
     finally:meta['Native_forward_host_summary'][kind+'_calls']+=1;meta['Native_forward_host_summary'][kind+'_s']+=time.monotonic()-start
    reason,trace=generate(req,recorded_forward,check,finite,FinishReason,context=32768,chunk_size=1024,eos=1,vocab=ns['cfg']['vocab_size']);meta['Native_continuation']=trace
   except BaseException as exc:failure=exc;meta['error']=repr(exc)
   finally:
    try:check(lib.modern_native_engine_reset(ctx));meta['normal_reset_after']=True
    except BaseException as exc:failure=exc;meta['reset_error']=repr(exc);engine._fatal=repr(exc)
    meta.update(ended_monotonic=time.monotonic(),native_finish_reason=getattr(reason,'value',str(reason)),actual_generated_tokens=len(req.generated_token_ids),generated_token_ids=list(req.generated_token_ids),GPU_free_all8=[list(torch.cuda.mem_get_info(i))for i in range(8)]);save()
   if failure is None:req.finish(reason)
   else:req.fail(failure)
   meta['transport_finish_reason']=getattr(req.finish_reason,'value',str(req.finish_reason));meta['final_metadata']=copy.deepcopy(req.metadata);save()
   if failure is not None:raise RuntimeError('Owned Native request failed; stop without advancing or retry')from failure
   if engine._fatal:raise RuntimeError('Normal Native reset failed; worker unhealthy')
  else:raise TimeoutError('Original-start full service absolute deadline; no automatic retry')
  report['phases']={}
  for name,where in outputs.items():
   p=where/'result.json';report['phases'][name]={'path':str(p),'SHA256':sha(p),'result':json.loads(p.read_bytes())}
  actual_HTTP=len(list(root.rglob('request-*.wire')));assert actual_HTTP==103 and len(report['requests'])==102;report['actual_HTTP']=actual_HTTP;report['actual_Native']=len(report['requests']);report['client_incremental_metrics_qualified']=False
  audits=[r['admission']['history_render_audit']for r in report['requests']];qualified=sum(a.get('qualified')is True for a in audits);report['history_audit_summary']={'actual_Native_requests':len(audits),'qualified_standard_request_audits':qualified,'unqualified':len(audits)-qualified,'not_arbitrary_history_proof':True}
  report.update(completed=all(v['result'].get('completed')is True for v in report['phases'].values()),passed=all(v['result'].get('passed')is True for v in report['phases'].values())and qualified==len(audits));save()
 finally:
  if client is not None and client.poll()is None:client.terminate();client.wait(timeout=30)
  server.should_exit=True;thread.join(timeout=30);engine.stop();report['normal_server_shutdown']=not thread.is_alive()
  if log is not None and not log.closed:log.close()
  save()
 assert report['normal_server_shutdown'],'HTTP did not stop normally'
