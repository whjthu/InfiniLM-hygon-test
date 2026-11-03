"""Same-body, explicitly selected W2/W4 quality requests to one normal residency.

All wires and partial results remain in the assigned output directory. No
Slurm, retries, model loading, tool execution, or tokenizer import is present.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request

NATIVE = '9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
PARENT = '5e944fb676f15022f1387d936cc9286c00953f0ca6884ef79b12979b4bca8ac4'
UPSTREAM = '99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'
DATA_SHA = 'ec1d98eb5dc6a45c7ab06c54df707715854afaecc046b3b657d62a1b028de523'
REQUEST_SHA = 'b536c18715a60aebd593d181436503b65ab10f70258f7abc791f1c9426797183'
EOS = {154820, 154827, 154829}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
    temporary.replace(path)


def numeric(text):
    values = re.findall(r'####[ *$]*([-+]?(?:[0-9][0-9,]*(?:[.][0-9]+)?|[.][0-9]+)(?:/[0-9]+)?)', text)
    try:
        return Fraction(values[-1].replace(',', '')) if values else None
    except (ValueError, ZeroDivisionError):
        return None


def wilson(correct, count):
    if not count:
        return None
    z = 1.959963984540054
    p = correct / count
    den = 1 + z*z/count
    mid = (p + z*z/(2*count)) / den
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / den
    return [mid-half, mid+half]


def summary(rows, expected_pairs):
    arms = {}
    for window in (2, 4):
        data = [r for r in rows if r['window'] == window]
        n, k = len(data), sum(r['correct'] for r in data)
        arms[str(window)] = dict(count=n, correct=k, accuracy=k/n if n else None,
            Wilson95=wilson(k,n), natural_EOS=sum(r['natural_EOS'] for r in data),
            caps=sum(r['cap_hit'] for r in data), parse_failures=sum(r['parse_failure'] for r in data),
            unexpected_tool_intents=sum(r['unexpected_tool_intent'] for r in data),
            mean_HTTP_wall_s=statistics.mean(r['HTTP_wall_s'] for r in data) if n else None)
    by_case = {}
    for row in rows:
        assert row['window'] not in by_case.setdefault(row['case_id'], {})
        by_case[row['case_id']][row['window']] = row
    pairs = [r for r in by_case.values() if set(r) == {2,4}]
    gain = sum(not r[2]['correct'] and r[4]['correct'] for r in pairs)
    loss = sum(r[2]['correct'] and not r[4]['correct'] for r in pairs)
    discordant = gain + loss
    exact_p = min(1., 2 * sum(math.comb(discordant,k) for k in range(min(gain,loss)+1)) / 2**discordant) if discordant else 1.
    n = len(pairs)
    delta = (gain-loss)/n if n else None
    # Paired scores are in [-1,1]. Hoeffding gives a conservative, nondegenerate
    # 95% interval even when all observed pairs tie; no bootstrap-zero illusion.
    radius = math.sqrt(2*math.log(40)/n) if n else None
    return dict(arms=arms, complete_pairs=n, expected_pairs=expected_pairs,
        pair_coverage=n/expected_pairs, W4_only_correct=gain, W2_only_correct=loss,
        paired_accuracy_W4_minus_W2=delta,
        paired_Hoeffding95=[max(-1.,delta-radius),min(1.,delta+radius)] if n else None,
        exact_two_sided_McNemar_p=exact_p,
        noninferiority_proven=False, full_benchmark=False,
        limits='Custom0shot disjoint64 screen; a high p value or equal outputs does not prove no quality loss. Wrong answers/caps remain in denominators.')


def validate_ready(args):
    ready = read(args.ready)
    assert ready['ready'] and ready['actual_step'] == args.expected_step
    assert ready['Native_SHA256'] == NATIVE
    assert (ready['TP'],ready['PP'],ready['cards'],ready['rows'],ready['capacity']) == (8,2,16,1,32768)
    assert ready['loader_parent_source_manifest_SHA256'] == args.loader_manifest_sha
    assert ready['service_profile_manifest_SHA256'] == args.profile_manifest_sha
    assert ready['prepared_entry_identity']['upstream_commit'] == UPSTREAM
    assert ready['context_policy'] == 'v403-no-resident-V' and ready['context_policy_env'] == '1'
    assert ready['P0_text_buffered_until_completion'] and ready['normal_output_budget'] == 4096
    assert Path(args.service_report).name == 'GLM-latest-full-owner-r290-'+ready['run']+'-stage1-rank0.json'
    return ready


def owner_row(args, ready, response_id, local_index, window, encoded, usage, finish):
    end = min(time.monotonic()+8, time.monotonic()+max(0,args.deadline_unix-time.time()))
    while time.monotonic() < end:
        q = read(args.service_report)
        assert q['actual_step'] == args.expected_step and q['run'] == ready['run']
        assert q['Native_SHA256'] == NATIVE and q['source_manifest_SHA256'] == args.loader_manifest_sha
        assert q['service_profile_manifest_SHA256'] == args.profile_manifest_sha
        rows = [r for r in q['service_requests'] if r.get('request_id') == response_id]
        assert len(rows) <= 1
        if rows and rows[0].get('final_metadata') and rows[0].get('normal_reset_after'):
            row = rows[0]
            break
        time.sleep(.1)
    else:
        raise RuntimeError('Completed original owner receipt unavailable; no retry')
    assert row['request_index'] == row['admitted_index'] == args.native_offset+local_index
    assert row['native_profile'] == 'glm-nov16-w'+str(window)
    assert row['prefill_control_window'] == window and row['service_profile_manifest_SHA256'] == args.profile_manifest_sha
    assert row['normal_reset_before'] and row['normal_reset_after'] and row['original_reset_all16_completed']
    a, h = row['admission'], row['admission']['history_render_audit']
    assert h['qualified'] and h['prequeue_bound'] and not h['markers_sent_to_Native']
    assert row['max_new_tokens'] == row['requested_output_budget'] == 4096
    assert row['prompt_tokens'] == encoded['prompt_tokens'] == usage['prompt_tokens'] == h['prompt_tokens']
    assert row['actual_prompt_token_ids_SHA256'] == encoded['prompt_IDs_SHA256'] == a['Native_prompt_token_ids_SHA256']
    assert sum(c['tokens'] for c in row['Native_continuation'] if c['kind']=='prefill') == row['prompt_tokens']
    assert all(c['kind'] in ('prefill','decode') for c in row['Native_continuation'])
    assert all(1 <= c['tokens'] <= 512 for c in row['Native_continuation'] if c['kind']=='prefill')
    assert row['prompt_tokens']+4096 <= 32768
    ids = row['generated_token_ids']
    assert ids and len(ids) == row['actual_generated_tokens'] == usage['completion_tokens']
    decode = [c for c in row['Native_continuation'] if c['kind']=='decode']
    assert len(decode)==len(ids)-1 and all(c['tokens']==1 for c in decode)
    assert all(type(t) is int and 0 <= t < 154880 for t in ids)
    assert digest(compact(ids)) == row['generated_token_ids_SHA256']
    natural = row['native_finish_reason'] == 'eos_token'
    assert row['native_finish_reason'] in ('eos_token','length')
    if natural:
        assert ids[-1] in EOS and not any(t in EOS for t in ids[:-1]) and finish != 'length'
    else:
        assert len(ids) == 4096 and not any(t in EOS for t in ids)
    ticks = row['backend_token_monotonic']
    assert len(ticks)==len(ids) and all(math.isfinite(t) for t in ticks)
    assert all(x<=y for x,y in zip(ticks,ticks[1:]))
    assert a['accepted_monotonic'] <= row['started_monotonic'] <= ticks[0] <= ticks[-1] <= row['ended_monotonic']
    visible = [t for t,i in zip(ticks,ids) if i not in EOS]
    gaps = sorted(y-x for x,y in zip(visible,visible[1:]))
    def percentile(p):
        if not gaps:return None
        k=(len(gaps)-1)*p;i=int(k);return gaps[i]+(gaps[min(i+1,len(gaps)-1)]-gaps[i])*(k-i)
    return dict(natural_EOS=natural, cap_hit=not natural, generated_tokens=len(ids),
        visible_tokens=len(visible), generated_IDs_SHA256=row['generated_token_ids_SHA256'],
        backend_TTFT_from_admission_s=visible[0]-a['accepted_monotonic'] if visible else None,
        backend_visible_decode_tok_s=(len(visible)-1)/(visible[-1]-visible[0]) if len(visible)>1 and visible[-1]>visible[0] else None,
        backend_ITL_P95_s=percentile(.95),backend_ITL_P99_s=percentile(.99),
        Native_forward_host_summary=row['Native_forward_host_summary'],
        owner_row_compact_SHA256=digest(compact(row)),normal_reset=True)


def run(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,execution_passed=False,quality_pass_fail_not_assigned=True,
        actual_HTTP=0,planned_HTTP=0,rows=[],errors=[],Native=NATIVE,
        shard_index=args.shard_index,shards=args.shards,profile_manifest_SHA256=args.profile_manifest_sha,
        loader_manifest_SHA256=args.loader_manifest_sha,
        client_source_SHA256=digest(Path(__file__).read_bytes()),
        client_incremental_SLO_qualified=False,raw_wires_stay_local=True,no_retries=True)
    save(out/'result.json',report)
    expected_pairs=0
    try:
        assert type(args.shards) is int and args.shards==2 and 0<=args.shard_index<args.shards
        assert args.native_offset >= 0 and math.isfinite(args.deadline_unix) and 0<args.timeout<86400
        ready=validate_ready(args)
        root=Path(args.fixtures);data_raw=(root/'gsm8k-new64.json').read_bytes();plan_raw=(root/'requests-and-pairs.json').read_bytes()
        assert digest(data_raw)==DATA_SHA and digest(plan_raw)==REQUEST_SHA
        dataset,plan=json.loads(data_raw),json.loads(plan_raw)
        cases={c['id']:c for c in dataset['cases']};bodies={r['case_id']:r for r in plan['requests']}
        # Keep consecutive even/odd question ordinals together, so BOTH replicas
        # contain 16 W2-first and 16 W4-first pairs; order is not tied to replica.
        selected={r['case_id'] for i,r in enumerate(plan['requests']) if (i//2)%args.shards==args.shard_index}
        schedule=[r for r in plan['comparison_schedule'] if r['case_id'] in selected]
        expected_pairs=len(selected);assert expected_pairs==32 and len(schedule)==64
        enc_raw=Path(args.encoding_receipt).read_bytes();assert digest(enc_raw)==args.encoding_sha
        encoding=json.loads(enc_raw)
        assert encoding['completed'] and encoding['passed'] and encoding['actual_checkpoint_CPU_processor']
        assert encoding['source_manifest_SHA256']==args.encoding_source_sha and not encoding['Torch_CUDA_initialized']
        assert not encoding['Native_constructed'] and not encoding['weights_loaded']
        assert encoding['requests_file_SHA256']==REQUEST_SHA and encoding['matching_body_pairs']==64
        encoded={r['case_id']:r for r in encoding['cases']};assert set(encoded)==set(cases)==set(bodies)
        report.update(planned_HTTP=len(schedule),planned_case_ids=sorted(selected),
            sharding='(selected_question_ordinal//2)%2;16 W2-first and16 W4-first pairs per replica',ready_SHA256=digest(Path(args.ready).read_bytes()),
            encoding_receipt_SHA256=args.encoding_sha,encoding_source_SHA256=args.encoding_source_sha,
            fixture_SHA256=DATA_SHA,requests_SHA256=REQUEST_SHA,actual_step=args.expected_step)
        save(out/'result.json',report)
        for local_index,item in enumerate(schedule):
            left=args.deadline_unix-time.time()
            if left<=120:raise TimeoutError('Original residency deadline reserve; no new request')
            case_id=item['case_id'];window=item['window'];row=bodies[case_id]
            raw=compact(row['body']);assert digest(raw)==item['compact_body_SHA256']==row['compact_body_SHA256']==encoded[case_id]['request_body_SHA256']
            where=out/('%03d-%s-W%d'%(local_index,case_id,window));where.mkdir()
            (where/'request.body.json').write_bytes(raw)
            profile='glm-nov16-w'+str(window)
            headers={'Content-Type':'application/json','X-InfiniLM-Native-Profile':profile}
            save(where/'headers.json',headers)
            req=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',data=raw,headers=headers)
            report['actual_HTTP']+=1;save(out/'result.json',report)
            started=time.monotonic()
            try:
                with urllib.request.urlopen(req,timeout=min(args.timeout,left-90)) as response:
                    status=response.status;wire=response.read()
            except urllib.error.HTTPError as error:
                status=error.code;wire=error.read()
            wall=time.monotonic()-started;(where/'response.wire').write_bytes(wire)
            assert status==200,('HTTP failure',status)
            reply=json.loads(wire);assert 'error'not in reply and len(reply['choices'])==1
            choice=reply['choices'][0];message=choice['message'];usage=reply['usage'];finish=choice['finish_reason']
            assert message['role']=='assistant' and finish in ('stop','length','tool_calls')
            assert all(type(usage[k]) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
            assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
            owner=owner_row(args,ready,reply['id'],local_index,window,encoded[case_id],usage,finish)
            content=message.get('content')or'';assert isinstance(content,str)
            value=numeric(content);unexpected_tool=bool(message.get('tool_calls'))
            result=dict(case_id=case_id,source_index=cases[case_id]['source_index'],window=window,
                global_schedule_index=item['index'],local_owner_index=args.native_offset+local_index,
                request_body_SHA256=digest(raw),response_wire_SHA256=digest(wire),response_id=reply['id'],
                HTTP_status=status,HTTP_wall_s=wall,HTTP_finish=finish,**owner,
                parse_failure=value is None,unexpected_tool_intent=unexpected_tool,
                correct=bool(value is not None and value==Fraction(cases[case_id]['gold_numeric']) and owner['natural_EOS'] and not unexpected_tool),
                parsed_numeric=str(value)if value is not None else None,output_budget=4096)
            report['rows'].append(result);report['summary']=summary(report['rows'],expected_pairs);save(out/'result.json',report)
        report['execution_passed']=report['actual_HTTP']==len(report['rows'])==len(schedule)
    except BaseException as error:
        import traceback
        report['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:
        report['completed']=True
        if expected_pairs:report['summary']=summary(report['rows'],expected_pairs)
        save(out/'result.json',report)
    return report


def parser():
    p=argparse.ArgumentParser()
    for name in ('base-url','ready','service-report','fixtures','encoding-receipt','encoding-sha','encoding-source-sha','profile-manifest-sha','loader-manifest-sha','expected-step','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shards',type=int,default=2)
    p.add_argument('--native-offset',type=int,default=0);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--timeout',type=float,default=3600)
    return p


if __name__=='__main__':
    q=run(parser().parse_args())
    print(json.dumps({k:q[k] for k in ('completed','execution_passed','actual_HTTP','planned_HTTP','errors')}))
    raise SystemExit(0 if q['execution_passed'] else 1)
