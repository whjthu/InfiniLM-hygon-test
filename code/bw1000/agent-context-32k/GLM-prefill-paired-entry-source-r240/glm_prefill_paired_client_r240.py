"""Replay one frozen real named-tool request, 2 warmups +4 ABBA blocks.
CPU HTTP client only. No synthetic model replies, changing inputs or retries.
This records emitter observations; root must audit all32 owners and cleanup.
"""
import argparse,hashlib,json,pathlib,time,urllib.request,urllib.error,subprocess,sys
PLAN=(1,2)+(1,2,2,1)*4

def validate_request(body):
    assert body['model']=='GLM-5.3' and body['stream'] is False
    assert body['max_tokens']==4096 and body['chat_template_kwargs']['enable_thinking'] is False
    assert body['tool_choice']=={'type':'function','function':{'name':'read_file'}}
    assert body['parallel_tool_calls'] is False and body['messages'] and body['tools']
    assert body['temperature']==0

def validate_reply(wire):
    assert isinstance(wire['id'],str) and wire['id'] and 'error' not in wire
    assert len(wire['choices'])==1
    choice=wire['choices'][0];message=choice['message'];calls=message.get('tool_calls',[])
    assert choice['finish_reason']=='tool_calls' and len(calls)==1
    call=calls[0];assert call['type']=='function' and isinstance(call['id'],str) and call['id']
    assert call['function']['name']=='read_file'
    assert json.loads(call['function']['arguments'])=={'path':'input.json'}
    usage=wire['usage'];assert all(type(usage[k]) is int and usage[k]>0 for k in ['prompt_tokens','completion_tokens','total_tokens'])
    assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
    assert usage['prompt_tokens']>=8192 and usage['prompt_tokens']+4096<=32768
    assert usage['completion_tokens']<=4096
    return usage

def main():
    p=argparse.ArgumentParser()
    for name in ['run','handle','request-file','request-sha256','output-root','service-report']:p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-monotonic',type=float,required=True)
    p.add_argument('--child',action='store_true');a=p.parse_args()
    R=pathlib.Path('/refactor');root=pathlib.Path(a.output_root)
    assert root.is_absolute() and root.is_relative_to(R/'results')
    source=pathlib.Path(a.request_file);raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==a.request_sha256
    body=json.loads(raw);validate_request(body)
    ready_path=R/'results'/('GLM-regular-service-r204-ready-'+a.run+'.json')
    ready=json.loads(ready_path.read_bytes())
    assert ready['ready'] and ready['run']==a.run and ready['actual_step']==a.handle
    assert ready['Native_DSO_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
    assert ready['TP']==8 and ready['PP']==4 and ready['cards']==32 and ready['context']==32768
    assert ready['prefill_pair_windows']==list(PLAN) and ready['paired_request_SHA256']==a.request_sha256
    if not a.child:
        wrapper=root.with_name(root.name+'-wrapper.json')
        row={'completed':False,'passed':False,'external_absolute_wall_bound':True,'no_retries':True,'single_child_only':True}
        with wrapper.open('x') as f:json.dump(row,f)
        try:
            remaining=a.deadline_monotonic-time.monotonic()-60
            if remaining<=180:raise TimeoutError('No bounded child budget')
            child=subprocess.run([sys.executable,'-B',str(pathlib.Path(__file__).resolve())]+sys.argv[1:]+['--child'],timeout=remaining,check=False)
            row['exit_code']=child.returncode;row['passed']=child.returncode==0 and time.monotonic()<a.deadline_monotonic-60
        except subprocess.TimeoutExpired:
            # subprocess.run kills and waits for its unique child on timeout.
            row['timed_out_and_child_reaped']=True
        except Exception as error:
            row['error_class']=type(error).__name__
        finally:
            row['completed']=True;wrapper.write_text(json.dumps(row,indent=2)+'\n')
            stop=R/'control'/('GLM-regular-service-r204-stop-'+a.run+'.json')
            if not stop.exists():
                with stop.open('x') as f:json.dump({'reason':'bounded paired wrapper terminal','passed':row['passed']},f)
        return 0 if row['passed'] else 1
    root.mkdir(exist_ok=False)
    result={'completed':False,'passed':False,'Model_qualified':False,'scope':'Emitter client observations; requires all32 owner/CCL/quality/cleanup audit',
        'actual_GPU_handle':a.handle,'run':a.run,'request_file_SHA256':a.request_sha256,'attempts':0,'completed_HTTP_200':0,
        'windows':list(PLAN),'warmup_requests':2,'measurement_requests':16,'cases':[],'no_retries':True,'wall_clock_pair_gain':None}
    request_ids=set();usage_reference=None
    def save():(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for index,window in enumerate(PLAN):
            remaining=a.deadline_monotonic-time.monotonic()
            if remaining<=180:raise TimeoutError('Common service/client deadline reserve exhausted')
            # Byte-identical complete request for every warmup and measured arm.
            req=urllib.request.Request('http://127.0.0.1:'+str(ready['port'])+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'})
            result['attempts']+=1;save();start=time.monotonic()
            with urllib.request.urlopen(req,timeout=min(3600,remaining-120)) as response:
                status=response.status;wire_raw=response.read()
            wall=time.monotonic()-start
            (root/('response-%02d.wire'%index)).write_bytes(wire_raw)
            assert status==200;result['completed_HTTP_200']+=1
            wire=json.loads(wire_raw);usage=validate_reply(wire)
            assert wire['id'] not in request_ids;request_ids.add(wire['id'])
            if usage_reference is None:usage_reference=usage['prompt_tokens']
            assert usage['prompt_tokens']==usage_reference
            result['cases'].append({'index':index,'window':window,'warmup':index<2,'request_id':wire['id'],
                'wire_SHA256':hashlib.sha256(wire_raw).hexdigest(),'usage':usage,'HTTP_wall_s':wall,'passed':True})
            save()
            if time.monotonic()>=a.deadline_monotonic-120:raise TimeoutError('Client completed beyond safe service deadline')
        owner=json.loads(pathlib.Path(a.service_report).read_bytes())
        assert owner['paired_run']==a.run and owner['paired_actual_step']==a.handle
        assert owner['library_sha256']==ready['Native_DSO_SHA256']
        rows=owner['service_requests'];assert len(rows)==18 and {r['request_id'] for r in rows}==request_ids
        by_id={r['request_id']:r for r in rows};assert len(by_id)==18
        prompt_hashes=set();sequences=[]
        for event in result['cases']:
            row=by_id[event['request_id']];sequences.append(row['sequence'])
            assert row['pair_index']==event['index'] and row['prefill_control_window']==event['window']
            assert row['paired_warmup']==event['warmup'] and row['prompt_tokens']==event['usage']['prompt_tokens']
            assert row['generated_tokens']==event['usage']['completion_tokens'] and row['requested_output_budget']==4096
            assert row['generated_natural_EOS'] and row['reason']=='eos_token' and row['original_reset_all32_completed']
            prompt_hashes.add(row['actual_prompt_token_ids_SHA256'])
        assert len(set(sequences))==18 and sequences==sorted(sequences) and len(prompt_hashes)==1
        result['emitter_request_owner_binding']={'passed':True,'actual_request_IDs_equal':True,
            'backend_sequences_ordered_unique':True,'pair_indices_and_windows_bound':True,
            'all_actual_prompt_IDs_same_SHA256':next(iter(prompt_hashes)),
            'natural_EOS_and_original_reset':True,'all32_owner_and_lifecycle_audit_still_required':True}
        result['passed']=result['attempts']==result['completed_HTTP_200']==len(result['cases'])==18
        # No speedup is published until Native phase timing, outputs, EOS and
        # matched conditions on all32 owners have independently been audited.
    except Exception as error:
        result['error_class']=type(error).__name__
        result['error_SHA256']=hashlib.sha256(str(error).encode()).hexdigest()
    finally:
        result['completed']=True;save()
        stop=R/'control'/('GLM-regular-service-r204-stop-'+a.run+'.json')
        if not stop.exists():
            with stop.open('x') as f:json.dump({'reason':'single bounded paired client terminal','passed':result['passed']},f)
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
