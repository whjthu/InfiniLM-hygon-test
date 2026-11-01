"""Run actual supervisor/client mains and wire parser with synthetic urllib replies.

The fixed production input files and source pins are real. Owner receipts and
answers are deliberately synthetic CPU fixtures, not Model quality evidence.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from unittest import mock

SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NATIVE = '038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoDeviceImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ['torch', 'infinilm', '_infinilm']:
            raise RuntimeError('CPU client validation forbids device import: ' + fullname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--source-manifest-sha', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    source = a.source.resolve()
    pin = a.source_manifest_sha
    assert not a.output.exists()
    result = {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'completed': False, 'passed': False, 'CPU_only': True, 'GPU': 0,
              'Model': 0, 'Native_dlopen': 0, 'production_HTTP': 0,
              'source_manifest_SHA256': pin, 'CPU_source_SHA256': SHA(__file__),
              'actual_CPU': os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
              'real_supervisor_and_client_mains_and_wire_parser': True,
              'synthetic_owner_receipts_and_gold_answers': True, 'cases': []}
    guard = NoDeviceImport()
    sys.meta_path.insert(0, guard)
    try:
        own = Path(__file__).resolve().parent
        assert SHA(own/'SOURCE_MANIFEST.json') == os.environ['GLM_CPU_SOURCE_SHA']
        own_manifest = json.loads((own/'SOURCE_MANIFEST.json').read_bytes())
        assert own_manifest['owner_source_manifest_SHA256'] == pin
        assert all(SHA(own/n) == h for n,h in own_manifest['files_SHA256'].items())
        assert SHA(source/'SOURCE_MANIFEST.json') == pin
        mf = json.loads((source/'SOURCE_MANIFEST.json').read_bytes())
        assert all(SHA(source/n) == h for n,h in mf['files_SHA256'].items())
        for name in mf['files_SHA256']:
            if name.endswith('.py'): ast.parse((source/name).read_bytes(), filename=name)
        sys.path.insert(0, str(source))
        import scope_plan_r328 as plan
        expected_windows = [2]*68 + [2,4,2,4,4,2]
        assert list(plan.PREFILL_WINDOWS) == expected_windows
        assert [v[2] for v in plan.PHASES] == [4,68,74]
        for i in range(74):
            r = plan.request_plan(i)
            assert r['admitted_index'] == i and r['prefill_control_window'] == expected_windows[i]
            assert r['pair_index'] == (None if i < 68 else i-68)
            assert r['paired_warmup'] is (68 <= i < 70)
        for invalid in [-1,74,True,1.0]:
            try: plan.request_plan(invalid)
            except ValueError: pass
            else: raise AssertionError('Invalid owner schedule index accepted')
        prep = json.loads((source/'PREPARATION.json').read_bytes())
        assert prep['planned_HTTP'] == prep['planned_Native'] == 74
        assert not prep['GPU_GO'] and not prep['original121_plan_replayed']
        from source_pin_r290 import verify,profiles
        verify(pin); profiles(pin)
        clients = {n:load(source/(n+'.py'),'CPU329_'+n) for n in
                   ['client_multicall_r290','client_quality_r290','client_paired_r290','client_driver_r290']}
        pair = clients['client_paired_r290']
        fixture = Path('/refactor/results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json')
        assert SHA(fixture) == pair.FROZEN_REQUEST_SHA256
        frozen = fixture.read_bytes(); frozen_body = json.loads(frozen)
        pair_gold = pair.validate_request(frozen_body)
        work = a.output.parent/'cases'; work.mkdir(exist_ok=False)
        scenarios = [('positive-bare',74),('positive-fence',74),('wrong-manifest',0),('wrong-step',0),
                     ('fence-prefix',2),('fence-suffix',2),('duplicate-key',2),('nonfinite',2),
                     ('wrong-nonce',2),('bool-sum',2),('multicall-index',2),
                     ('quality-index',5),('quality-clock',5),('quality-cap',68),
                     ('pair-window',71),('pair-prompt',70),('pair-tool',69)]
        def no_network(*args, **kwargs):
            raise RuntimeError('CPU tests forbid production network')
        with mock.patch.object(socket.socket,'connect',no_network), mock.patch.object(socket,'create_connection',no_network):
            for scenario,expected_calls in scenarios:
                root = work/scenario; root.mkdir()
                tag = 'CPU329b-'+result['actual_CPU'].replace('.','-')+'-'+scenario
                handle = 'CPU_SYNTHETIC_GPU_NOT_SUBMITTED'
                report = Path('/refactor/results')/('GLM-latest-full-owner-r290-'+tag+'-stage3-rank0.json')
                identity_path = root/'identity.json'
                identity = dict(run=tag,actual_step=handle,Native_SHA256=NATIVE,
                    TP=8,PP=4,cards=32,rows=1,capacity=32768,source_manifest_SHA256=pin,
                    prepared_entry_identity={'upstream_commit':'99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'})
                owner = dict(identity,service_requests=[],queue_journal={'max_distinct_P1_sessions_unfinished':1})
                if scenario == 'wrong-step': identity['actual_step'] = 'CPU_WRONG_STEP'
                assert not report.exists(); dump(report,owner); dump(identity_path,identity)
                calls = []
                def urlopen(req,timeout):
                    assert req.full_url == 'http://127.0.0.1:43843/v1/chat/completions' and timeout > 0
                    body = json.loads(req.data); i = len(calls); assert i < 74
                    calls.append(copy.deepcopy(body)); messages = body['messages']
                    thinking = body['chat_template_kwargs'].get('enable_thinking',False)
                    budget = body['max_tokens']; assert budget == (8192 if thinking else 4096)
                    assert not body['stream']; ids = [7,8,154820]; finish = 'stop'; native_finish = 'eos_token'
                    if i < 4:
                        prompt = 9000
                        assert body['parallel_tool_calls'] and body['tools'][0]['function']['name']=='read_file'
                        if i % 2 == 0:
                            assert body['tool_choice'] == 'required'
                            finish = 'tool_calls'
                            message = {'role':'assistant','content':'','tool_calls':[
                                {'id':'CPUopaque-%d-%d'%(i,k),'type':'function','function':{
                                    'name':'read_file','arguments':json.dumps({'path':name})}}
                                for k,name in enumerate(['left.json','right.json'])]}
                        else:
                            assert body['tool_choice']=='none'
                            old = next(m['tool_calls'] for m in messages if m.get('tool_calls'))
                            tools = [m for m in messages if m['role']=='tool']
                            assert [m['tool_call_id'] for m in tools] == list(reversed([m['id']for m in old]))
                            byid = {m['tool_call_id']:json.loads(m['content'])for m in tools}; answer = {}
                            for c in old:
                                data = byid[c['id']]; name = json.loads(c['function']['arguments'])['path']
                                values = {r['id']:r['value']for r in data['records']}
                                answer[name] = {'nonce':data['nonce'],'sum':sum(values[k]for k in data['query_ids'])}
                            answer['total'] = sum(v['sum'] for v in answer.values())
                            if scenario=='wrong-nonce': answer['left.json']['nonce']='WRONG'
                            if scenario=='bool-sum': answer['total']=True
                            content = json.dumps(answer)
                            if scenario in ['positive-fence','fence-prefix','fence-suffix']:
                                content = '```json\n'+content+'\n```'
                            if scenario=='fence-prefix': content='Explanation '+content
                            if scenario=='fence-suffix': content+=' Extra'
                            if scenario=='duplicate-key': content=content[:-1]+',"total":0}'
                            if scenario=='nonfinite': content=content[:-1]+',"bad":NaN}'
                            message = {'role':'assistant','content':content}
                    elif i < 68:
                        assert not thinking and body['tool_choice']=='none'; prompt=900
                        suite = 'gsm8k-fixed32.json' if i<36 else 'mmlu-fixed32.json'
                        case = json.loads((source/'benchmark'/suite).read_bytes())['cases'][i-4 if i<36 else i-36]
                        question = case['question']
                        if i>=36: question+='\n'+'\n'.join('%s. %s'%(chr(65+k),v)for k,v in enumerate(case['choices']))
                        assert messages[-1]['content']==question
                        answer = case['gold_numeric'] if i<36 else chr(65+case['answer'])
                        message={'role':'assistant','content':'#### '+answer}
                        if scenario=='quality-cap' and i==4:
                            finish='length'; native_finish='length'; ids=[7]*4096
                    else:
                        assert req.data==frozen
                        prompt=8900; finish='tool_calls'
                        message={'role':'assistant','tool_calls':[{'id':'CPU-pair-%d'%i,'type':'function',
                            'function':{'name':'wrong' if scenario=='pair-tool' else 'calculate',
                                        'arguments':json.dumps(pair_gold)}}]}
                    if thinking: message['reasoning_content']='CPU_SYNTHETIC_REASONING'
                    rid='CPU_SYNTHETIC_%03d'%i; start=100.+i*10
                    ticks=[start+.2+k*.001 for k in range(len(ids))]; end=ticks[-1]+.1
                    if scenario=='quality-clock' and i==4: ticks[0]=float('nan')
                    rc=[hashlib.sha256((m.get('reasoning_content')or m.get('reasoning')).encode()).hexdigest()
                        for m in messages if m.get('role')=='assistant'and(m.get('reasoning_content')or m.get('reasoning'))]
                    h=dict(qualified=True,prequeue_bound=True,markers_sent_to_Native=False,prompt_tokens=prompt,
                        generation_prefix_correct=True,generation_thinking=True,pending_thinking_prefix_open=True,
                        source_RC_SHA256_sequence=rc,RC_one_contribution_per_source_field=True,RC_order_preserved=True,
                        RC_contributions_nonoverlapping=True,content_one_contribution_each_and_ordered=True,
                        all_content_contribution_slices_equal=True,actual_prompt_SHA256='CPU_SYNTHETIC',
                        Native_prompt_token_ids_SHA256='CPU_SYNTHETIC')
                    metadata=plan.request_plan(i)
                    if (scenario=='multicall-index'and i==0)or(scenario=='quality-index'and i==4): metadata['admitted_index']+=1
                    if scenario=='pair-window'and i==70: metadata['prefill_control_window']=1
                    r=dict(metadata,request_id=rid,prompt_tokens=prompt,max_new_tokens=budget,
                        actual_generated_tokens=len(ids),generated_token_ids=ids,backend_token_monotonic=ticks,
                        started_monotonic=start,ended_monotonic=end,normal_reset_before=True,normal_reset_after=True,
                        native_finish_reason=native_finish,final_metadata={'CPU_SYNTHETIC':True},
                        Native_continuation=[{'kind':'prefill','tokens':prompt}],
                        Native_forward_host_summary={'prefill_calls':18,'prefill_s':.01,'decode_calls':len(ids)-1,'decode_s':.002},
                        actual_prompt_token_ids_SHA256='WRONG'if scenario=='pair-prompt'and i==69 else'CPU_SYNTHETIC',
                        admission={'diagnostic_user':body['user'],'accepted_monotonic':start-.1,'history_render_audit':h})
                    owner['service_requests'].append(r); dump(report,owner)
                    usage={'prompt_tokens':prompt,'completion_tokens':len(ids),'total_tokens':prompt+len(ids)}
                    response=io.BytesIO(json.dumps({'id':rid,'choices':[{'message':message,'finish_reason':finish}],
                                                   'usage':usage}).encode())
                    response.status=200; response.headers={'Content-Type':'application/json'}; return response
                def child(argv,**kwargs):
                    name=Path(argv[2]).stem; assert name in clients and name!='client_driver_r290'
                    saved=sys.argv; sys.argv=[argv[2],*argv[3:]]
                    try: code=clients[name].main()
                    finally: sys.argv=saved
                    return SimpleNamespace(returncode=code)
                saved=sys.argv
                sys.argv=[str(source/'client_driver_r290.py'),'--run',tag,'--handle',handle,
                    '--identity',str(identity_path),'--service-report',str(report),'--manifest-sha',
                    '0'*64 if scenario=='wrong-manifest'else pin,'--deadline-unix',str(time.time()+8000)]
                with mock.patch.object(urllib_request(),'urlopen',urlopen), mock.patch.object(subprocess,'run',child), \
                     mock.patch.dict(os.environ,{k:'-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}):
                    try: code=clients['client_driver_r290'].main()
                    finally: sys.argv=saved
                supervisor=Path('/refactor/results')/('GLM-latest-full-owner-r290-supervisor-'+tag+'.json')
                q=json.loads(supervisor.read_bytes()); positive=scenario.startswith('positive-')
                assert code==(0 if positive else 1) and q['passed']is positive and len(calls)==expected_calls,(scenario,code,len(calls),q)
                assert q['completed']and q['planned_HTTP']==q['planned_Native']==74
                if positive:
                    assert list(q['phases'])==['multicall','quality','paired']and q['actual_HTTP']==q['actual_Native']==74
                    assert [q['phases']['quality']['result']['suites'][s]['correct']for s in ['GSM8K','MMLU']]==[32,32]
                    assert [r['admitted_index']for r in owner['service_requests']]==list(range(74))
                elif len(calls)<=4: assert 'quality'not in q['phases']and 'paired'not in q['phases']
                elif len(calls)<=68: assert 'paired'not in q['phases']
                stop=Path('/refactor/control')/('GLM-latest-full-owner-r290-stop-'+tag+'.json')
                assert json.loads(stop.read_bytes())['actual_GPU_handle']==handle
                result['cases'].append({'name':scenario,'passed':True,'synthetic_HTTP':len(calls),
                    'returncode':code,'supervisor_SHA256':SHA(supervisor),'stop_SHA256':SHA(stop)})
                print(json.dumps(result['cases'][-1]),flush=True)
        assert 'torch'not in sys.modules
        result['passed']=len(result['cases'])==len(scenarios)
    except BaseException as e:
        result.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        sys.meta_path.remove(guard); result['completed']=True; dump(a.output,result)
        print(json.dumps({k:result.get(k)for k in ['completed','passed','actual_CPU','error']}),flush=True)
    return 0 if result['passed']else 1


def urllib_request():
    import urllib.request
    return urllib.request


if __name__=='__main__': raise SystemExit(main())
