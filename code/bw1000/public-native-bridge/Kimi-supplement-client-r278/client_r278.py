"""One complete latest99 residency: finite agent, fixed32 real quality, C1 events. No retry."""
import argparse,copy,hashlib,importlib.util,json,math,os,pathlib,re,subprocess,sys,time
from fractions import Fraction
R=pathlib.Path('/refactor');S=R/'incoming/Kimi-latest-service-r275';C=pathlib.Path(__file__).resolve().parent
NATIVE_SHA='ec239cf5d60fa4c00d6f2168beb2289b13151063f1f068a83e3cf64d8b9c99e4';MODEL_SHA='f72bbc3fcb0971e395121ba297eae08080d67eab16c7eb9aefb101eaa1057671';FIXED_SHA='9c488dcb625b72e9dfb3d0b001ec95878b4d0d1366785c1acb84e44f1ad0b2f9'
from transport_binding_r278 import load_transport
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p,n,h):
 assert sha(p)==h;spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def owner(path, ident, session, index, mode, fixture, remaining):
    end = time.monotonic() + 8
    while time.monotonic() < end:
        remaining()
        q = json.loads(path.read_bytes())
        rows = [v for v in q['agent_requests'] if v['request_id'] == ident]
        assert len(rows) <= 1
        if rows and rows[0].get('normal_reset') and ('finish' in rows[0]):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError('Exact same-run owner receipt unavailable')
    v = rows[0]
    assert v['client_session'] == session and v['admitted_index'] == index and (v['prefill_mode'] == mode)
    assert v['input_tokens'] == fixture['expected_prompt_tokens'] and v['output_budget'] == 4096 and v['normal_reset'] and (v['finish'] == 'eos') and (v['generated_ids'][-1] in [163585, 163586])
    a = v['history_render_audit']
    assert a['qualified'] and a['Native_prompt_token_ids_SHA256'] == fixture['expected_Native_prompt_IDs_SHA256'] and (not a['markers_sent_to_Native'])
    obs = v['long_prefill_observations']
    assert len(obs) == 1 and obs[0]['full_prompt_consumed'] and (obs[0]['input_tokens_committed'] == v['input_tokens'])
    ticks = v['backend_token_monotonic']
    assert len(ticks) == len(v['generated_ids']) and all((x <= y for x, y in zip(ticks, ticks[1:])))
    return {'mode': mode, 'admitted_index': index, 'request_id': ident, 'client_session': session, 'input_tokens': v['input_tokens'], 'output_budget': 4096, 'output_tokens': len(v['generated_ids']), 'output_ID_SHA256': hashlib.sha256(json.dumps(v['generated_ids'], separators=(',', ':')).encode()).hexdigest(), 'prompt_ID_SHA256': a['Native_prompt_token_ids_SHA256'], 'prefill_wall_s': obs[0]['prefill_wall_s'], 'prefill_CAPI_maxhost_sum_s': obs[0]['recorded_completed_forward_CAPI_maxhost_s_sum'], 'owner_wall_s': v['finished_monotonic'] - v['started_monotonic'], 'backend_first_token_wait_s': ticks[0] - v['started_monotonic'], 'natural_EOS': True, 'normal_reset': True}

def numeric_answer(text):
 values=re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)',text)
 if not values:return None
 try:return Fraction(values[-1].replace(',',''))
 except (ValueError,ZeroDivisionError):return None

def percentile(v,p):
 if not v:return None
 v=sorted(v);x=(len(v)-1)*p;i=int(x);j=min(i+1,len(v)-1);return v[i]+(v[j]-v[i])*(x-i)

def actual_owner(path,rid,index,session,budget,remaining):
 until=time.monotonic()+8
 while time.monotonic()<until:
  remaining();q=json.loads(path.read_bytes());rows=[v for v in q['agent_requests']if v['request_id']==rid];assert len(rows)<=1
  if rows and rows[0].get('normal_reset')and rows[0].get('finish'):break
  time.sleep(.05)
 else:raise RuntimeError('Same-run completed owner unavailable')
 v=rows[0];a=v['history_render_audit'];assert v['admitted_index']==index and v['client_session']==session and v['output_budget']==budget and v['prefill_mode']=='W512'and v['normal_reset'];assert v['finish']=='eos'and v['generated_ids'][-1]in[163585,163586];assert a['qualified']and a['prequeue_bound']and not a['markers_sent_to_Native']and v['long_prefill_progress']['input_tokens_committed']==v['input_tokens']==a['prompt_tokens'];assert v['input_tokens']+budget<=32768
 return v

def benchmark(p0,args,out,remaining,save,parent):
 suite=json.loads((S/'benchmark/gsm8k-fixed-random32.json').read_bytes());assert len(suite['cases'])==32;folder=out/'gsm32';folder.mkdir();rows=[];parent['GSM8K']={'scope':'Existing fixed random32 official-test seed20261026; custom0shot ordinary4096, reused inputs, screening only','suite_SHA256':sha(S/'benchmark/gsm8k-fixed-random32.json'),'source_revision':suite['source_revision'],'source_SHA256':suite['source_SHA256'],'seed':suite['seed'],'sample_count':32,'rows':rows,'paired_quality_delta':None,'no_regression_proven':False,'full_benchmark':False,'coarse_screen_min_correct':24};save()
 for offset,case in enumerate(suite['cases']):
  remaining();d=folder/('%02d'%offset);d.mkdir();client=p0.WireClient(args.base_url,args.model,d,4096,min(args.timeout,remaining()-60));session='P1-GSM32-r267-%02d'%offset;client.session=session
  messages=[{'role':'system','content':'Solve this math problem accurately. Explain your work and end with #### followed by your numeric answer.'},{'role':'user','content':case['question']}];message,finish,usage,event=client.request(copy.deepcopy(messages),'none',False,None,4096,False);assert event['HTTP_status']==200 and finish=='stop'and not message.get('tool_calls');wire=json.loads((d/'request-001.wire').read_bytes());body=json.loads((d/'request-001.request.json').read_bytes());assert body['messages']==messages and body['max_tokens']==4096;row=actual_owner(pathlib.Path(args.service_report),wire['id'],4+offset,session,4096,remaining);assert row['input_tokens']==usage['prompt_tokens'];value=numeric_answer(message.get('content')or'');gold=Fraction(case['gold_numeric'].replace(',',''));ticks=row['backend_token_monotonic'];ids=row['generated_ids'];assert len(ticks)==len(ids)and all(a<=b for a,b in zip(ticks,ticks[1:]));visible=[v for v,t in zip(ticks,ids)if t not in[163585,163586]];gaps=[b-a for a,b in zip(visible,visible[1:])];speed=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 and visible[-1]>visible[0]else None
  rows.append({'source_index':case['source_index'],'question_SHA256':hashlib.sha256(case['question'].encode()).hexdigest(),'correct':value==gold,'parse_failure':value is None,'natural_EOS':True,'cap_hit':False,'output_budget':4096,'prompt_tokens':row['input_tokens'],'generated_tokens':len(ids),'backend_visible_tokens':len(visible),'backend_node0_decode_tok_s':speed,'backend_node0_ITL_p95_s':percentile(gaps,.95),'backend_node0_ITL_p99_s':percentile(gaps,.99),'backend_first_wait_from_Native_start_s':visible[0]-row['started_monotonic']if visible else None,'client_HTTP_wall_s':event['elapsed_s'],'client_incremental_metrics_qualified':False,'request_SHA256':sha(d/'request-001.request.json'),'wire_SHA256':sha(d/'request-001.wire')});save()
 n=len(rows);k=sum(v['correct']for v in rows);p=k/n;z=1.959963984540054;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;speeds=[v['backend_node0_decode_tok_s']for v in rows if v['backend_node0_decode_tok_s']is not None];parent['GSM8K'].update(completed=True,correct=k,accuracy=p,Wilson95=[mid-half,mid+half],parse_failures=sum(v['parse_failure']for v in rows),EOS=n,caps=0,coverage=n/32,backend_node0_decode_p50=percentile(speeds,.5),backend_node0_decode_p95=percentile(speeds,.95),strict_client_gt15_qualified=False,coarse_screen_pass=k>=24);save();assert k>=24,'COARSE_GSM32_SCREEN_BELOW_PREDECLARED24_OF32_NOT_NO_REGRESSION_TEST'

def profile(p0,args,out,remaining,save,parent):
 fixed=R/'candidates/Kimi-latest-service-r267/fixed-real-request-r267.json';assert sha(fixed)==FIXED_SHA;f=json.loads(fixed.read_bytes());parent['C1_profile']={'warmup_indices':[36,37],'OFF_before':38,'ON':39,'OFF_after':40,'arms':[],'attribution_qualified':False};folder=out/'profile';folder.mkdir()
 for offset,label in enumerate(['warm0','warm1','OFF-before','ON','OFF-after']):
  remaining();d=folder/label;d.mkdir();client=p0.WireClient(args.base_url,args.model,d,4096,min(args.timeout,remaining()-60));session='P1-profile-r267-%02d'%offset;client.session=session;body=f['body'];message,finish,usage,event=client.request(copy.deepcopy(body['messages']),body['tool_choice'],False,copy.deepcopy(body['tools']),4096,False,body['parallel_tool_calls']);assert event['HTTP_status']==200 and finish=='tool_calls'and usage['prompt_tokens']==f['expected_prompt_tokens'];calls=message.get('tool_calls',[]);assert len(calls)==1 and calls[0]['function']['name']=='calculate'and json.loads(calls[0]['function']['arguments'])==f['expected_calculate_arguments'];wire=json.loads((d/'request-001.wire').read_bytes());row=owner(pathlib.Path(args.service_report),wire['id'],session,36+offset,'W512',f,remaining);row.update(label=label,client_HTTP_wall_s=event['elapsed_s']);parent['C1_profile']['arms'].append(row);save()
 arms=parent['C1_profile']['arms'];nodes=[];all_coverage=True
 for node in range(4):
  p=R/'results'/('Kimi-latest-service-r275-'+os.environ['GLOBAL_RUN_R2354']+'-node%d.json'%node);until=time.monotonic()+8
  while True:
   remaining();q=json.loads(p.read_bytes());requests=q['agent_requests']
   if len(requests)==41 and requests[-1].get('normal_reset')and requests[-1].get('finish'):break
   if time.monotonic()>=until:raise RuntimeError('Same-run final profile owner publication unavailable')
   time.sleep(.05)
  records={v['label']:v for v in q.get('sparse_model_profiles',[])};on=records.get('agent-'+requests[39]['request_id']);mapping=on.get('actual_forward_map',[])if on else [];raw=on.get('raw_decoder_owner_events',{})if on else {};coverage=bool(on and on.get('armed')and raw.get('all8_published')and len(raw.get('owners',{}))==8 and len(mapping)==1 and mapping[0]['C']==mapping[0]['T']==1 and mapping[0]['widths']==[1]and mapping[0]['actual_verify_index_zero_based']>=2);all_coverage=all_coverage and coverage
  selected=[];spans=[]
  for index in [38,39,40]:
   row=requests[index];rec=records.get('agent-'+row['request_id'],{});samples=[v['CAPI_maxhost_s']for v in rec.get('verify_host_samples',[])if v['verify_index_zero_based']==2];selected.append(samples[0]if len(samples)==1 else None);vis=[t for t,i in zip(row['backend_token_monotonic'],row['generated_ids'])if i not in[163585,163586]];spans.append(vis[-1]-vis[0]if len(vis)>1 else None)
  def bracket(values):
   if any(v is None or v<=0 for v in values):return {'qualified':False}
   baseline=(values[0]+values[2])/2;ratio=values[1]/baseline;spread=abs(values[0]-values[2])/baseline;return {'OFF_before':values[0],'ON':values[1],'OFF_after':values[2],'ON_over_OFF_bracket':ratio,'OFF_relative_spread':spread,'qualified':ratio<=1.03 and spread<=.03}
  nodes.append({'node':node,'all8_published_real_C1T1':coverage,'report_SHA256':sha(p),'on_record_SHA256':hashlib.sha256(json.dumps(on,sort_keys=True,separators=(',',':')).encode()).hexdigest()if on else None,'host_adapter_s':on.get('host_adapter_s')if on else None,'selected_actual_third_verify_CAPI':bracket(selected),'backend_visible_decode_span':bracket(spans),'decoder_summary':on.get('decoder_summary')if on else None})
 same_path=len({v['output_ID_SHA256']for v in arms})==1;perturbation=all(v['selected_actual_third_verify_CAPI']['qualified']and v['backend_visible_decode_span']['qualified']for v in nodes);parent['C1_profile'].update(completed=True,all32_owner_coverage=all_coverage,per_node=nodes,matched_actual_output_path=same_path,perturbation_within3pct=perturbation,attribution_qualified=all_coverage and same_path and perturbation,not_kernel_optimization=True,UNKNOWN_fine_scopes=['router/dequant/GEMM split','pure CCL or PP','GPU busy percentage']);save()

def main():
 p=argparse.ArgumentParser()
 for n in ['base-url','model','output-dir','identity','service-report']:p.add_argument('--'+n,required=True)
 p.add_argument('--timeout',type=float,default=11200);args=p.parse_args();out=pathlib.Path(args.output_dir);out.mkdir(exist_ok=False);identity=pathlib.Path(args.identity);ident=json.loads(identity.read_bytes());assert ident['native_SHA256']==NATIVE_SHA and ident['manifest_SHA256']==MODEL_SHA and ident['protocol_commit']=='99f5c90a40f9b2546bd0202cba3397ed1d55c5a0';end_unix=float(os.environ['KIMI_R205_CLIENT_DEADLINE_UNIX']);deadline=time.monotonic()+max(0,end_unix-time.time());q={'completed':False,'passed':False,'identity_SHA256':sha(identity),'max_HTTP_attempts':41,'max_Native_requests':41,'no_retries':True,'no_cropping':True,'parent_deadline_UTC_unix':end_unix,'deadline_resets_per_phase':False,'client_incremental_latency_qualified':False,'wide_benchmark_or_no_degradation_proven':False}
 def save():(out/'result.json').write_text(json.dumps(q,indent=2)+chr(10))
 def remaining():
  left=deadline-time.monotonic()
  if left<=120:raise TimeoutError('Fixed original client deadline reserve exhausted')
  return left
 save()
 try:
  q['normal34_inherited_from_GPU']='910380.513';q['normal34_repeated']=False
  multiout=out/'multicall';cmd=[sys.executable,'-B',str(C/'client_multicall_r278.py')];values={'base-url':args.base_url,'output-dir':str(multiout),'identity':str(identity),'service-report':args.service_report,'frozen-client-dir':str(S/'client'),'profiles':'/refactor/candidates/Kimi-long-agent32K-r179/regular-agent-profiles-r179/profiles.json','ready':str(R/'results'/('Kimi-latest-service-ready-r275-'+os.environ['GLOBAL_RUN_R2354']+'.json')),'expected-run':os.environ['GLOBAL_RUN_R2354'],'expected-gpu-handle':ident['actual_step'],'deadline-unix':str(min(end_unix,time.time()+8900)),'timeout':str(min(3600,remaining()-60))}
  for k,v in values.items():cmd+=['--'+k,v]
  q['multicall_returncode']=subprocess.run(cmd,timeout=remaining()-60).returncode;q['multicall']=json.loads((multiout/'result.json').read_bytes());save();assert q['multicall_returncode']==0 and q['multicall']['passed']
  p0=load_transport(S,'frozen_supplement_p0_r278');benchmark(p0,args,out,remaining,save,q);profile(p0,args,out,remaining,save,q);rows=json.loads(pathlib.Path(args.service_report).read_bytes())['agent_requests'];assert len(rows)==41 and [r['admitted_index']for r in rows]==list(range(41))and all(r['prefill_mode']=='W512'for r in rows);q['actual_HTTP_requests']=41;q['actual_Native_requests']=41;q['full_request_sequence_bound']=True;q['finite_latest_agent_and_coarse_screen_passed']=True;q['profile_diagnostic_is_not_a_quality_gate']=True;q['passed']=True
 except BaseException as error:q['error']=repr(error)
 finally:q['completed']=True;save()
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
