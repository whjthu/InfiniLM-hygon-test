"""Frozen paired few-shot HTTP client for the real public425 Run namespace.

No weights, Native, tokenizer, retry, service, or Slurm construction. Quality
outcomes continue; infrastructure/state faults keep the complete planned set.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
import urllib.error
import urllib.request

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from owner_validation import owner_row, verify_trace, verify_native_calls

NATIVE = '9520769d28742aba489b38ecbd1e1692f6830168c699d7111f1e0c2ae4cd598b'
ENCODING = '1e083b8241422ed87511d83fa8585bfc19eb356e539d42acfa80b4c10cc8351c'
ENCODER_SOURCE = '56254b9f9b703f61339f1db912dbe9650d93913ca83b298cf1c2aa6f19127f7f'
DATA = 'c350be0128d1304ecbe5ae0d9ddd53d2d2137faac37f733b60bf0ba7cdb08a38'
REQUESTS = '0224a12e37732ed9823f662300d49b419777c914d2cbee56819eb6e066c0b694'
TEMPLATE = '3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
UPSTREAM = '99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())
    temporary.replace(path)


def load_leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures(args):
    root = Path(args.fixtures)
    data, plan, enc = [(root/'cases.json').read_bytes(),
                       (root/'requests-and-pairs.json').read_bytes(), Path(args.encoding_receipt).read_bytes()]
    assert digest(data)==DATA and digest(plan)==REQUESTS and digest(enc)==ENCODING
    cases, plan, encoding = json.loads(data)['cases'], json.loads(plan), json.loads(enc)
    assert len(cases)==256 and len(plan['requests'])==256 and len(plan['comparison_schedule'])==512
    assert encoding['passed'] and encoding['completed'] and encoding['actual_checkpoint_CPU_processor']
    assert encoding['source_manifest_SHA256']==ENCODER_SOURCE and encoding['template_SHA256']==TEMPLATE
    assert not encoding['Torch_CUDA_initialized'] and not encoding['Native_constructed'] and not encoding['weights_loaded']
    assert encoding['requests_file_SHA256']==REQUESTS and encoding['matching_body_pairs']==256
    encoded = {c['case_id']:c for c in encoding['cases']}
    assert set(encoded)=={c['id']for c in cases}=={r['case_id']for r in plan['requests']}
    assert args.shards in (2,4,6,8) and 0<=args.shard_index<args.shards
    selected = {r['case_id']for i,r in enumerate(plan['requests']) if (i//2)%args.shards==args.shard_index}
    schedule = [r for r in plan['comparison_schedule'] if r['case_id']in selected]
    chosen = [c for c in cases if c['id']in selected]
    assert len(schedule)==2*len(chosen) and len(chosen)>0
    for c in chosen:
        entries = [r for r in schedule if r['case_id']==c['id']]
        assert sorted(r['window']for r in entries)==[2,4]
        assert len({r['compact_body_SHA256']for r in entries})==1
    return chosen, plan, encoded, schedule


def bind_identity(args):
    root = Path(args.provider_root).resolve()
    raw = (root/'SOURCE_MANIFEST.json').read_bytes()
    assert digest(raw)==args.provider_source_sha
    manifest = json.loads(raw)
    for name, expected in manifest['files_SHA256'].items():
        target = root/name
        assert target.resolve().is_relative_to(root) and digest(target.read_bytes())==expected
    assert 'public_glm_native/prefill_window_audit.py'in manifest['files_SHA256']
    assert digest((root/'public_glm_native/native_owner.py').read_bytes())==args.audited_owner_sha
    config_raw = Path(args.config).read_bytes()
    assert digest(config_raw)==args.config_sha
    config = json.loads(config_raw)
    assert config['Native_SHA256']==NATIVE and config['backend']=='qualified-native-glm-nov16'
    assert (config['TP'],config['PP'],config['rows'],config['context'])==(8,2,1,32768)
    assert (config['Native_seconds'],config['service_seconds'])==(28800,28200)
    assert args.deadline_unix==args.gpu_start_epoch+config['service_seconds']
    config.update(run=args.run,actual_step=args.expected_step,gpu_start_epoch=args.gpu_start_epoch,
                  source_manifest_SHA256=args.provider_source_sha,provider_config_SHA256=args.config_sha)
    namespace = Path(config['runtime_directory'])/args.run
    assert namespace.is_dir(), 'Client cannot create a missing service namespace'
    protocol = load_leaf('actual_glm444_protocol',root/'public_glm_native/protocol.py')
    run = protocol.Run(config)
    run.raise_if_failed()
    assert not run.stop_requested(), 'Already stopped residency'
    ready = run.read('MODEL_READY.json')
    assert ready['Model_READY'] and ready['Native_constructed_loaded_by_provider']
    assert ready['actual_owner_count']==16 and (ready['TP'],ready['PP'],ready['rows'],ready['capacity'])==(8,2,1,32768)
    assert ready['Native_SHA256']==NATIVE and ready['all16_loaded_postload4GiB_and_service_agreement']
    prepared = run.read('PREPARED.json')['prepared_identity']
    assert ready['prepared_identity']==prepared and prepared['upstream_commit']==UPSTREAM
    assert set(prepared['eos_token_ids'])=={154820,154827,154829}
    records = run.all_ready()  # actual Run validates 16 identities/Native/masks
    assert len(records)==16
    assert Path(args.service_report).resolve()==(run.path/'owner-1-0.json').resolve()
    # The actual425 worker binds BOTH fields to the new provider source MF.
    args.loader_manifest_sha=args.profile_manifest_sha=args.provider_source_sha
    return run, ready


def verify_all_owner_rows(args, run, rows, encoded):
    signatures=[]
    for stage in range(2):
        for rank in range(8):
            raw=(run.path/('owner-%d-%d.json'%(stage,rank))).read_bytes()
            owner=json.loads(raw)
            assert owner['run']==args.run and owner['actual_step']==args.expected_step
            assert owner['source_manifest_SHA256']==owner['service_profile_manifest_SHA256']==args.provider_source_sha
            assert owner['Native_SHA256']==NATIVE
            assert (owner['TP'],owner['PP'],owner['all_serving_cards'],owner['cache_rows'],owner['cache_capacity'])==(8,2,16,1,32768)
            records=owner['service_requests']
            for result in rows:
                index=result['local_owner_index'];row=records[index];entry=encoded[result['case_id']]
                assert row['request_index']==row['admitted_index']==index
                assert row['prefill_control_window']==result['window']
                assert row['native_profile']=='glm-nov16-w'+str(result['window'])
                assert row['actual_prompt_token_ids_SHA256']==entry['prompt_IDs_SHA256']
                assert row['generated_token_ids_SHA256']==result['generated_IDs_SHA256']
                assert row['generated_natural_EOS']==result['natural_EOS'] and row['original_reset_all16_completed']
                assert row['prompt_tokens']==entry['prompt_tokens'] and row['requested_output_budget']==4096
                verify_trace(row,result['window'],entry);verify_native_calls(row,finite_expected=stage==1)
            signatures.append(dict(stage=stage,rank=rank,report_SHA256=digest(raw),verified_requests=len(rows)))
    return signatures


def run(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    score=load_leaf('actual_glm444_scoring',D.parent/'scoring.py')
    # Fixed full global denominator exists before even config/ready can fail.
    report=dict(completed=False,execution_passed=False,planned_global_questions=256,planned_global_HTTP=512,
        planned_HTTP=2*sum((i//2)%args.shards==args.shard_index for i in range(256)) if args.shards in (2,4,6,8) and 0<=args.shard_index<args.shards else None,
        actual_HTTP=0,rows=[],errors=[],pending_attempt=None,quality_is_not_all_correct_gate=True,
        no_retries=True,GPU_or_Model_constructed_by_client=False,Native=NATIVE,
        shard_index=args.shard_index,shards=args.shards,client_source_SHA256=digest(Path(__file__).read_bytes()),
        noninferiority_proven=False,full_benchmark=False,raw_wires_stay_R_L=True)
    save(out/'result.json',report)
    chosen=[];encoded={};service=None
    try:
        assert args.native_offset==0 and math.isfinite(args.deadline_unix) and 0<args.timeout<=86400
        chosen,plan,encoded,schedule=fixtures(args)
        exposure=[c['id']for c in chosen if encoded[c['id']]['planned_pair_control_exposed']]
        report.update(planned_HTTP=len(schedule),planned_case_ids=[c['id']for c in chosen],
            summary=score.summarize(chosen,[],exposure_case_ids=exposure),
            sharding='(selected_ordinal//2)%shards; both arms same replica; balanced first-arm order',
            provider_source_SHA256=args.provider_source_sha,encoding_SHA256=ENCODING,
            actual_step=args.expected_step,run=args.run)
        save(out/'result.json',report)
        service,ready=bind_identity(args)
        bodies={r['case_id']:r for r in plan['requests']};cases={c['id']:c for c in chosen}
        for local_index,item in enumerate(schedule):
            if service.stop_requested():raise InterruptedError('Same-run STOP before next request; no retry')
            left=args.deadline_unix-time.time()
            if left<=120:raise TimeoutError('Original service deadline reserve; no new request')
            case_id,window=item['case_id'],item['window'];entry=encoded[case_id]
            raw=compact(bodies[case_id]['body'])
            assert digest(raw)==item['compact_body_SHA256']==bodies[case_id]['compact_body_SHA256']==entry['request_body_SHA256']
            where=out/('%03d-%s-W%d'%(local_index,case_id,window));where.mkdir()
            (where/'request.body.json').write_bytes(raw)
            headers={'Content-Type':'application/json','X-InfiniLM-Native-Profile':'glm-nov16-w'+str(window)}
            save(where/'headers.json',headers)
            report['pending_attempt']=dict(case_id=case_id,window=window,local_owner_index=local_index,
                request_body_SHA256=digest(raw),global_schedule_index=item['index'])
            report['actual_HTTP']+=1;save(out/'result.json',report)
            req=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',data=raw,headers=headers)
            started=time.monotonic()
            try:
                with urllib.request.urlopen(req,timeout=min(args.timeout,left-90))as response:
                    status=response.status;wire=response.read()
            except urllib.error.HTTPError as error:
                status=error.code;wire=error.read()
            wall=time.monotonic()-started;(where/'response.wire').write_bytes(wire)
            assert status==200,('HTTP infrastructure failure',status)
            reply=json.loads(wire);assert 'error'not in reply and len(reply['choices'])==1
            choice=reply['choices'][0];message=choice['message'];usage=reply['usage'];finish=choice['finish_reason']
            assert message['role']=='assistant' and finish in ('stop','length','tool_calls')
            assert all(type(usage[k])is int and usage[k]>=0 for k in ['prompt_tokens','completion_tokens','total_tokens'])
            assert usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
            owner=owner_row(args,ready,reply['id'],local_index,window,entry,usage,finish)
            content=message.get('content')or'';assert isinstance(content,str)
            graded=score.grade(cases[case_id],content,natural_EOS=owner['natural_EOS'],cap_hit=owner['cap_hit'],unexpected_tool=bool(message.get('tool_calls')))
            result=dict(case_id=case_id,suite=cases[case_id]['suite'],window=window,source_index=cases[case_id]['source_index'],
                global_schedule_index=item['index'],local_owner_index=local_index,response_id=reply['id'],
                request_body_SHA256=digest(raw),response_wire_SHA256=digest(wire),HTTP_wall_s=wall,HTTP_status=status,
                HTTP_finish=finish,output_budget=4096,**owner)
            result.update(graded);report['rows'].append(result);report['pending_attempt']=None
            report['summary']=score.summarize(chosen,report['rows'],exposure_case_ids=exposure);save(out/'result.json',report)
        report['all16_owner_request_signatures']=verify_all_owner_rows(args,service,report['rows'],encoded)
        report['execution_passed']=len(schedule)==report['actual_HTTP']==len(report['rows'])
    except BaseException as error:
        report['errors'].append(dict(error=repr(error),traceback=traceback.format_exc()))
        # Do not retry/cancel/reset an owner locally; the controller owns bounded
        # fail-stop. Preserve a partial active attempt for its original cleanup.
    finally:
        report['completed']=True
        if chosen:report['summary']=score.summarize(chosen,report['rows'],exposure_case_ids=[c['id']for c in chosen if encoded[c['id']]['planned_pair_control_exposed']])
        save(out/'result.json',report)
    return report


def parser():
    p=argparse.ArgumentParser()
    for name in ['base-url','provider-root','provider-source-sha','audited-owner-sha','config','config-sha','run','expected-step','service-report','fixtures','encoding-receipt','output-dir']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--shards',type=int,required=True);p.add_argument('--shard-index',type=int,required=True)
    p.add_argument('--native-offset',type=int,default=0)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--gpu-start-epoch',type=float,required=True)
    p.add_argument('--timeout',type=float,default=5400)
    return p


def main(argv=None):
    def interrupted(signum,frame):raise InterruptedError('Client signal '+str(signum)+'; original attempt is not retried')
    old=signal.signal(signal.SIGTERM,interrupted)
    try:
        q=run(parser().parse_args(argv))
        print(json.dumps({k:q[k]for k in ['completed','execution_passed','actual_HTTP','planned_HTTP','errors']}))
        return 0 if q['execution_passed']else 1
    finally:signal.signal(signal.SIGTERM,old)


if __name__=='__main__':raise SystemExit(main())
