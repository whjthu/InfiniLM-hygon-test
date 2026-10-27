"""Resident TP8 DeepSeek ac937 worker behind the shared unchanged HTTP routes."""
import pathlib,json,hashlib,os,time,threading,subprocess,sys,queue,copy,importlib.util
D=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def serve(ns):
 from native_engine_bridge_r143c import NativeEngineBridge,existing_server_app
 from infinilm.llm import FinishReason
 import uvicorn
 lib=ns['lib'];ctx=ns['ctx'];check=ns['check'];forward=ns['forward'];tokenizer=ns['tokenizer'];np=ns['np'];torch=ns['torch'];tag=ns['tag'];r=ns['r'];model=ns['model'];port=43863
 encoder_path=D/'encoding/encoding_dsv4.py';spec=importlib.util.spec_from_file_location('deep_official_encoder_r143',encoder_path);encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
 def encode_chat(messages,data,add_generation_prompt,kwargs):
  messages=copy.deepcopy(messages);tools=kwargs.get('tools')
  if tools:
   if not messages or messages[0].get('role')!='system':messages.insert(0,{'role':'system','content':''})
   messages[0]['tools']=tools
  for message in messages:
   if isinstance(message.get('content'),list):message['content']=''.join(p['text']for p in message['content'])
   for call in message.get('tool_calls')or[]:
    if isinstance(call['function'].get('arguments'),dict):call['function']['arguments']=json.dumps(call['function']['arguments'],ensure_ascii=False,allow_nan=False)
  mode=kwargs.get('thinking_mode','thinking'if kwargs.get('enable_thinking',True)else'chat')
  prompt=encoder.encode_messages(messages,thinking_mode=mode,reasoning_effort=kwargs.get('reasoning_effort'))
  if not add_generation_prompt:prompt=prompt.removesuffix('<｜Assistant｜>'+('<think>'if mode=='thinking'else'</think>'))
  return prompt,tokenizer.encode(prompt,add_special_tokens=False)
 engine=NativeEngineBridge(tokenizer,1024,FinishReason,encode_chat=encode_chat,eos_token_ids=(1,),default_max_tokens=96)
 service,app=existing_server_app(engine,str(model),'deepseek_v4',host='127.0.0.1',port=port)
 config=uvicorn.Config(app,host='127.0.0.1',port=port,log_level='info');server=uvicorn.Server(config);server.install_signal_handlers=lambda:None
 worker=threading.Thread(target=server.run,name='deep-http-r143',daemon=False)
 identity=r/'results'/('deep-agent-p0-r143c-'+tag+'-identity.json');report_path=r/'results'/('deep-agent-p0-r143c-'+tag+'.json')
 report={'completed':False,'passed':False,'actual_step':tag,'model_directory':str(model),'Native_SHA256':sha(ns['artifact']),'model_source':'5f91ad738583db8f5617aeae53b76523f85d1ec6','protocol_source':'7ceb60d656074376b911c2a232196a9457dfa6a6','official_encoder_SHA256':sha(encoder_path),'TP':8,'PP':1,'cards':8,'cache_rows':128,'cache_capacity':1024,'requests':[],'HTTP_URL':'http://127.0.0.1:%d/v1'%port,'P0_text_buffered':True,'new_performance_claim':False,'new_Native_compilation':False,'bootstrap':ns['__bootstrap_proof__'],'normal_server_shutdown':False}
 def save():
  tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(report_path)
 assert report['Native_SHA256']=='ac9370c8f188bfda8555718c52690b914883fc6b02b9267cbd279be3ecdd3835'
 report['actual_runtime_maps']=[x for x in pathlib.Path('/proc/self/maps').read_text().splitlines()if ('libinfinilm_runtime.so'in x or'_infinilm.cpython-'in x or'libdeep_c4_smallt_model_r9254.so'in x)]
 report['native_CAPI_keys']=len(ns['keys']);assert report['native_CAPI_keys']==67569
 identity.write_text(json.dumps({k:v for k,v in report.items()if k not in['requests']},ensure_ascii=False,indent=2));save();worker.start();client=None;log=None;deadline=time.monotonic()+6500
 try:
  while not server.started:
   if not worker.is_alive():raise RuntimeError('HTTP server stopped before ready')
   if time.monotonic()>deadline:raise TimeoutError('HTTP startup deadline')
   time.sleep(.05)
  report['server_ready']=True;report['server_ready_monotonic']=time.monotonic();save();print('DEEP_AGENT_P0_R143C_HTTP_READY',str(identity),flush=True)
  output=r/'results'/('deep-agent-p0-r143c-'+tag+'-client');log=(r/'logs'/('deep-agent-p0-r143c-'+tag+'-client.log')).open('x')
  args=[sys.executable,'-B',str(D/'client_p0_r143.py'),'--base-url',report['HTTP_URL'],'--model',pathlib.Path(model).name,'--output-dir',str(output),'--identity',str(identity),'--max-tokens','96','--timeout','1000','--thinking-case'];client=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
  while time.monotonic()<deadline:
   if client.poll()is not None:
    report['client_returncode']=client.returncode;break
   if not worker.is_alive():raise RuntimeError('HTTP server stopped during P0')
   try:req=engine.take(timeout=.1)
   except queue.Empty:continue
   if req is None:break
   meta={'request_id':req.request_id,'prompt_tokens':len(req.prompt_token_ids),'max_new_tokens':req.max_new_tokens,'generated_token_ids':[],'started_monotonic':time.monotonic(),'normal_reset_before':False,'normal_reset_after':False};report['requests'].append(meta);save();reason=FinishReason.LENGTH;failure=None
   try:
    check(lib.modern_native_engine_reset(ctx));meta['normal_reset_before']=True;ids=list(req.prompt_token_ids);past=0
    for _ in range(req.max_new_tokens):
     if req.cancel_event.is_set():reason=req.abort_reason or FinishReason.CANCELED;break
     assert ids and past+len(ids)<=1024
     rc,logits,emitted=forward(ids,list(range(past,past+len(ids))),[0],[past],[past+len(ids)],[0,len(ids)]);check(rc)
     row=(logits[0].astype(np.uint32)<<16).view(np.float32);assert np.isfinite(row).all()
     token=int(emitted[0]);assert 0<=token<ns['cfg']['vocab_size'];req.emit_token(token);meta['generated_token_ids'].append(token)
     if token==1:reason=FinishReason.EOS_TOKEN;break
     past+=len(ids);ids=[token]
   except BaseException as exc:failure=exc;meta['error']=repr(exc)
   finally:
    try:check(lib.modern_native_engine_reset(ctx));meta['normal_reset_after']=True
    except BaseException as exc:failure=exc;meta['reset_error']=repr(exc);engine._fatal=repr(exc)
    meta['ended_monotonic']=time.monotonic();meta['finish_reason']=getattr(reason,'value',str(reason));meta['GPU_free_after']=[list(torch.cuda.mem_get_info(k))for k in range(8)];save()
   if failure is None:req.finish(reason)
   else:req.fail(failure)
   if engine._fatal is not None:raise RuntimeError('Native reset failed; service unhealthy')
  else:raise TimeoutError('P0 service hard deadline')
  p=output/'result.json';q=json.loads(p.read_bytes());report.update(client_evidence_directory=str(output),client_result_SHA256=sha(p),P0_cases=q['cases'],P0_conditional_cases=q.get('conditional_cases',[]),passed=bool(q['passed']),completed=bool(q['completed']))
 except BaseException as exc:
  report['error']=repr(exc);save();raise
 finally:
  if client is not None and client.poll()is None:client.terminate();client.wait(timeout=20)
  server.should_exit=True;worker.join(timeout=30);report['normal_server_shutdown']=not worker.is_alive();engine.stop()
  if log is not None:log.close()
  save()
 assert report['normal_server_shutdown'],'HTTP failed to stop normally'
