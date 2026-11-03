"""All256 fixed Kimi ordinary HTTP requests, one actual public residency.

No retry/reload/codec/Native import. Accuracy outcomes continue; actual transport,
identity/state/finite errors stop with the full denominator and pending attempt.
"""
import argparse,hashlib,importlib.util,json,math,signal,time,traceback,urllib.error,urllib.parse,urllib.request
from pathlib import Path
from handoff import NATIVE,ready_handoff,stop_requested
from listener import wait_listener

D=Path(__file__).resolve().parent
compact=lambda v:json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()
sha=lambda b:hashlib.sha256(b).hexdigest()
EOS={163585,163586}

def save(path,q):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_bytes((json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());tmp.replace(p)

def load_score():
    path=D/'scoring.py';assert sha(path.read_bytes())=='c9c39a5fc185ed6bfc5f48c2b547d4ef36a4cfed351626835793fcf0ab5b521c'
    spec=importlib.util.spec_from_file_location('Kimi448_original_score',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def summarize(cases,rows,score):
    assert len({r['case_id']for r in rows})==len(rows)
    out={}
    for suite in ['gsm8k','mmlu']:
        ids={c['id']for c in cases if c['suite']==suite};known=[r for r in rows if r['case_id']in ids]
        n=128;k=sum(r['correct']for r in known);missing=n-len(known)
        out[suite]=dict(planned=n,observed=len(known),correct=k,planned_completion_adjusted_accuracy=k/n,
                        Wilson95=score.wilson(k,n),missing_outcome_accuracy_bounds=[k/n,(k+missing)/n],
                        missing_or_unverified=missing,caps=sum(r['cap_hit']for r in known),parse_failures=sum(r['parse_failure']for r in known),
                        natural_EOS=sum(r['natural_EOS']for r in known),raw_answer_matches_gold=sum(r['answer_matches_gold']for r in known))
    return dict(per_suite=out,planned_denominator=256,single_baseline=True,noninferiority_proven=False,full_benchmark=False,quality_is_not_all_correct_gate=True)

def fixtures(args):
    pins=json.loads((D/'INPUT_PINS.json').read_bytes());encraw=Path(args.encoding_receipt).read_bytes();assert sha(encraw)==pins['encoding_SHA256']
    enc=json.loads(encraw);assert enc['passed']and enc['completed']and enc['actual_CPU']=='910407.937'and len(enc['cases'])==256
    assert enc['actual_Kimi_codec']and enc['actual_public_protocol']and enc['actual_prepared_FIFO']and not enc['Native_dlopen']and not enc['GPU']
    b=Path(args.cases).read_bytes();assert sha(b)==pins['cases_SHA256'];cases=json.loads(b)['cases']
    b=Path(args.requests).read_bytes();assert sha(b)==pins['requests_SHA256']==enc['requests_file_SHA256'];requests=json.loads(b)['requests']
    assert len(cases)==len(requests)==256 and[c['id']for c in cases]==[r['case_id']for r in requests]==[r['case_id']for r in enc['cases']]
    for c,r,e in zip(cases,requests,enc['cases']):
        assert sha(compact(r['body']))==r['request_body_SHA256']==e['request_body_SHA256']
        assert r['body']['model']=='Kimi-K3'and r['body']['max_tokens']==4096 and r['body']['temperature']==0 and r['body']['stream']is False
        assert r['body']['chat_template_kwargs']==dict(enable_thinking=False)and r['body']['tool_choice']=='none'
        assert e['prompt_tokens']+4096<=32768 and e['actual_thinking']is False
    return cases,requests,enc['cases']

def wait_row(path,ident,index,args):
    limit=min(args.deadline_unix,time.time()+90)
    while time.time()<limit:
        try:q=json.loads(Path(path).read_bytes())
        except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1);continue
        assert q['run']==args.run and q['public_source_manifest_SHA256']==args.source_sha and q['library_sha256']==NATIVE
        rows=q.get('agent_requests',[])
        if len(rows)>index and rows[index].get('normal_reset')and rows[index].get('finish'):
            row=rows[index];assert row['request_id']==ident;return row
        if q.get('error'):raise RuntimeError(q['error'])
        time.sleep(.1)
    raise TimeoutError('Original owner completed reset receipt unavailable')

def verify_row(row,e,reply):
    assert row['prefill_mode']=='W512'and row['normal_reset']and row['finish']in('eos','length')
    assert not row.get('error')and not row.get('cleanup_errors')
    assert row['input_tokens']==e['prompt_tokens']and row['output_budget']==4096
    h=row['history_render_audit'];assert h['qualified']and h['prequeue_bound']and not h['markers_sent_to_Native']and h['generation_thinking']is False
    assert h['Native_prompt_token_ids_SHA256']==e['prompt_IDs_SHA256']and h['RC_order_preserved']and h['RC_one_contribution_per_source_field']
    obs=row['long_prefill_observations'];assert len(obs)==1 and obs[0]['full_prompt_consumed']and obs[0]['input_tokens_committed']==e['prompt_tokens']
    assert obs[0]['actual_forward_records']==len(e['prefill_chunk_tokens'])and obs[0]['width8_prefill_verify_commit_count']==0
    ids=row['generated_ids'];ticks=row['backend_token_monotonic'];assert ids and all(type(i)is int and 0<=i<163840 for i in ids)
    natural=row['finish']=='eos'
    assert (ids[-1]in EOS and not any(i in EOS for i in ids[:-1]))if natural else(len(ids)==4096 and not any(i in EOS for i in ids))
    assert len(ticks)==len(ids)and all(math.isfinite(t)for t in ticks)and all(a<=b for a,b in zip(ticks,ticks[1:]))
    assert row['started_monotonic']<=ticks[0]<=ticks[-1]<=row['finished_monotonic']
    usage=reply['usage'];assert all(type(usage[k])is int for k in['prompt_tokens','completion_tokens','total_tokens'])
    assert usage['prompt_tokens']==e['prompt_tokens']and usage['completion_tokens']==len(ids)and usage['total_tokens']==e['prompt_tokens']+len(ids)
    finish=reply['choices'][0]['finish_reason'];assert finish in('stop','length','tool_calls')
    assert(finish!='length')if natural else(finish=='length')
    visible=[t for t,i in zip(ticks,ids)if i not in EOS]
    return dict(natural_EOS=natural,cap_hit=not natural,generated_including_EOS=len(ids),generated_IDs_SHA256=sha(compact(ids)),
                visible_tokens=len(visible),backend_TTFT_from_dequeue_s=ticks[0]-row['started_monotonic'],
                backend_visible_decode_tps=(len(visible)-1)/(visible[-1]-visible[0])if len(visible)>1 and visible[-1]>visible[0]else None,
                normal_reset=True,prompt_IDs_SHA256=e['prompt_IDs_SHA256'],owner_row_compact_SHA256=sha(compact(row)))

def parser():
    p=argparse.ArgumentParser()
    for n in ['base-url','runtime-dir','run','expected-step','source-sha','config-sha','model-sha','cases','requests','encoding-receipt','output-dir']:p.add_argument('--'+n,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--gpu-start-epoch',type=float,required=True)
    return p

def main(argv=None):
    a=parser().parse_args(argv);output=Path(a.output_dir);output.mkdir(exist_ok=False);result=output/'result.json';score=load_score()
    q=dict(completed=False,execution_passed=False,planned_HTTP=256,attempted_HTTP=0,completed_HTTP=0,rows=[],errors=[],pending_attempt=None,retry=False,reload=False,Native=NATIVE,HTTP=True,SSE=False,performance_comparison=False,quality_result_scope='Single qualified8df custom chat fewshot GSM128+MMLU128; no noninferiority',Native_or_Model_constructed_by_client=False,noninferiority_proven=False)
    cases=[]
    def persist():
        if cases:q['summary']=summarize(cases,q['rows'],score)
        save(result,q)
    persist()
    old=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):raise InterruptedError('Owned client signal; preserve pending attempt, no retry')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        assert a.deadline_unix==a.gpu_start_epoch+28200
        cases,requests,encoded=fixtures(a);persist()
        ready=ready_handoff(a.runtime_dir,a.run,a.expected_step,a.source_sha,a.config_sha,a.model_sha);assert ready is not None
        node0=ready['records'][0];port=urllib.parse.urlparse(a.base_url).port or 80
        wait_listener(node0['PID'],port,a.deadline_unix)
        q.update(actual_ready_handoff=ready,actual_PID_owned_public_LISTEN=True,run=a.run,actual_step=a.expected_step,provider_source_SHA256=a.source_sha,config_SHA256=a.config_sha,encoding_SHA256=json.loads((D/'INPUT_PINS.json').read_bytes())['encoding_SHA256']);persist()
        report=Path(a.runtime_dir)/('Kimi-public-r428-'+a.run+'-node0.json');seen=set()
        for i,(case,request,e)in enumerate(zip(cases,requests,encoded)):
            if stop_requested(a.runtime_dir,a.run,a.source_sha,a.config_sha,a.gpu_start_epoch):raise InterruptedError('Validated same-run STOP before next request')
            left=a.deadline_unix-time.time()
            if left<=120:raise TimeoutError('Original28200 service deadline reserve; planned denominator retained')
            where=output/('%03d-%s'%(i,case['id']));where.mkdir();raw=compact(request['body']);(where/'request.body.json').write_bytes(raw)
            q['attempted_HTTP']+=1;q['pending_attempt']=dict(index=i,case_id=case['id'],body_SHA256=sha(raw));persist()
            sent=time.monotonic();wire=b''
            try:
                req=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=raw,headers={'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(req,timeout=min(10800,left-90))as response:status=response.status;wire=response.read()
                except urllib.error.HTTPError as error:status=error.code;wire=error.read()
            finally:
                (where/'response.wire').write_bytes(wire)
            ended=time.monotonic();q['pending_attempt'].update(HTTP_status=status,wire_SHA256=sha(wire),wall_s=ended-sent);persist()
            assert status==200,('PublicHTTP failure',status)
            reply=json.loads(wire);assert 'error'not in reply and len(reply['choices'])==1 and reply['id']not in seen;seen.add(reply['id'])
            message=reply['choices'][0]['message'];assert message['role']=='assistant';content=message.get('content')or'';assert type(content)is str
            owner=wait_row(report,reply['id'],i,a);metrics=verify_row(owner,e,reply)
            graded=score.grade(case,content,natural_EOS=metrics['natural_EOS'],cap_hit=metrics['cap_hit'],unexpected_tool=bool(message.get('tool_calls')))
            q['rows'].append(dict(index=i,case_id=case['id'],suite=case['suite'],source_index=case['source_index'],request_id=reply['id'],request_body_SHA256=sha(raw),response_wire_SHA256=sha(wire),usage=reply['usage'],HTTP_finish=reply['choices'][0]['finish_reason'],client_wall_s=ended-sent,**metrics,**{k:v for k,v in graded.items()if k not in metrics}))
            q['completed_HTTP']+=1;q['pending_attempt']=None
            if i==0:q['first_public_HTTP_success']=dict(index=0,actual_step=a.expected_step,request_id=reply['id'],normal_owner_reset=True,not_full256_qualification=True)
            persist()
        q['execution_passed']=q['attempted_HTTP']==q['completed_HTTP']==len(q['rows'])==256
    except BaseException as error:q['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:
        signal.signal(signal.SIGTERM,old);q['completed']=True;persist()
    print(json.dumps({k:q[k]for k in ['completed','execution_passed','attempted_HTTP','completed_HTTP','errors']}),flush=True)
    return 0 if q['execution_passed']else 1

if __name__=='__main__':raise SystemExit(main())
