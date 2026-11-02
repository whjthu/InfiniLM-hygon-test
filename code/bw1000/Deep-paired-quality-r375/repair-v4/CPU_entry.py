"""Actual worker->identity/report->actual client66 glue with recording leaves.

No real HTTP, GPU context, Native dlopen, model weight payload or Slurm is used.
The encoder receipt comes from a separate actual prepared-ingress CPU process.
"""
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.request
from types import SimpleNamespace as NS
import numpy as np

D = pathlib.Path(__file__).resolve().parent
S = D / 'owner'
R = pathlib.Path('/refactor')
O = pathlib.Path(sys.argv[1])
O.mkdir(exist_ok=False)
sys.path.insert(0, str(S))
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')

q = dict(completed=False, passed=False, CPU_only=True, GPU=False, Native_dlopen=False,
         weights_opened=False, actual_step=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], cases=[])
try:
    from source_pin_r281 import verify
    verify()
    from native_binding_r206b import binding
    b, receipt, norm = binding()
    assert b['library_SHA256'] == '976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
    assert len(norm['link']['full']['objects']) == 49
    from quality_plan import encoding, load_data, plan, descriptor_for_index, messages
    enc = encoding(); refs, cases = load_data()
    prior_path=R/'candidates/Deep-paired-quality-r375/CPU-910407.749/entry/CPU-entry.json'
    assert sha(prior_path)=='a471abdaefdf46bb127a2189c5f3932a320c7d7c88d9e2bb22568596e61dd016'
    prior=json.loads(prior_path.read_bytes()); assert prior['completed'] and not prior['passed'] and prior['actual_step']=='910407.749'
    inherited=[x for x in prior['cases'] if x.get('passed') is True]
    seams=[x for x in inherited if x['kind']=='actual_worker_report_to_actual_client_main66']; assert {x['replica']for x in seams}==set(range(4)) and all(x['completed_HTTP']==66 for x in seams)
    q['cases']=[dict(x,proof_origin='inherited749_same_owner_client_bytes')for x in inherited]
    q['inherited_749_entry_SHA256']=sha(prior_path);q['only_new_tail_scenarios']=['HC-hit-failure','client-failure']
    import preweight_r206b, bootstrap_r206b
    os.environ['DEEP_R277_GPUSTART_UNIX'] = '0'
    try:
        preweight_r206b.run()
    except TimeoutError:
        pass
    else:
        raise AssertionError('Expired provider reached GPU imports')
    tree, proof = bootstrap_r206b.transform(bootstrap_r206b.BASE.read_bytes())
    assert proof['Native_forward_AST_unchanged'] and proof['original_finally_destroy_retained']
    q['loader_AST'] = proof
    original_provider = preweight_r206b.run
    original_path = sys.path.copy()
    preweight_r206b.run = lambda: {'proof': {'CPU_provider_leaf': True}, 'prepared': {}}
    names = ['INFINILM_DEEP_QUALITY_PREFLIGHT_ONLY_R9101', 'INFINILM_OWNER_CPU_WIDTH_R2392',
             'INFINILM_DEEP_MOE_TAIL512_R7211', 'INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750', 'DEEP_REPLICA_R9278']
    previous = {k: os.environ.get(k) for k in names}
    os.environ.update(INFINILM_DEEP_QUALITY_PREFLIGHT_ONLY_R9101='1', INFINILM_OWNER_CPU_WIDTH_R2392='4',
                      INFINILM_DEEP_MOE_TAIL512_R7211='0', INFINILM_DEEP_PREFILL_PACKUPLOAD_R8750='0', DEEP_REPLICA_R9278='n01')
    try:
        try:
            bootstrap_r206b.main()
        except SystemExit as e:
            assert e.code == 0
        else:
            raise AssertionError('Actual bootstrap did not stop before Torch')
    finally:
        preweight_r206b.run = original_provider; sys.path[:] = original_path
        for k, v in previous.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    assert 'torch' not in sys.modules
    q['cases'].append(dict(kind='actual_bootstrap_preTorch_prefix', passed=True, objects=49))
    for shard in range(4):
        os.environ['DEEP_R375_REPLICA'] = str(shard)
        assert len(plan()) == 66
        for bad in [-1, 66, True, '1']:
            try: descriptor_for_index(b, bad)
            except ValueError: pass
            else: raise AssertionError('Invalid finite index admitted')
    from native_decode_r206b import generate
    reasons = NS(LENGTH='length', EOS_TOKEN='eos_token', CANCELED='canceled')
    for cancel in ['none', 'before', 'midprefill', 'expired', 'Native-error']:
        os.environ['DEEP_R277_GPUSTART_UNIX'] = '0' if cancel == 'expired' else str(time.time())
        req = NS(prompt_token_ids=[7] * 1025, max_new_tokens=4096, cancel_event=threading.Event(), abort_reason=None, emitted=[])
        req.emit_token = req.emitted.append
        if cancel == 'before': req.cancel_event.set()
        calls = []
        def fw(ids, pos, rows, past, total, offsets):
            calls.append(len(ids))
            if cancel == 'midprefill': req.cancel_event.set()
            if cancel == 'Native-error': raise RuntimeError('recording Native failure')
            return 0, None, [1 if total[0] >= 1027 else 8 if total[0] == 1026 else 7]
        error = None
        try: reason, trace = generate(req, fw, lambda x: None, lambda x: None, reasons)
        except BaseException as e: error = e
        if cancel == 'none': assert error is None and req.emitted == [7, 8, 1] and calls == [1024, 1, 1, 1]
        elif cancel in ['before', 'midprefill']: assert error is None and reason == 'canceled' and not req.emitted
        elif cancel == 'expired': assert isinstance(error, TimeoutError) and not calls
        else: assert isinstance(error, RuntimeError) and len(calls) == 1
        q['cases'].append(dict(kind='actual_generate', cancel=cancel, passed=True, calls=calls))
    # Real worker with only HTTP server/thread, client process and GPU leaves recorded.
    server_module = types.ModuleType('infinilm.server.inference_server')
    server_module.__file__ = str(R / 'candidates/latest-family-service-r270/Deep/python/infinilm/server/inference_server.py')
    server_module.DEFAULT_STREAM_TIMEOUT = 1800; server_module.DEFAULT_REQUEST_TIMEOUT = 7200
    root_module = types.ModuleType('infinilm'); root_module.__path__ = []
    server_package = types.ModuleType('infinilm.server'); server_package.__path__ = []
    root_module.server = server_package; server_package.inference_server = server_module
    sys.modules.update({'infinilm': root_module, 'infinilm.server': server_package, 'infinilm.server.inference_server': server_module})
    servers = []
    class Server:
        def __init__(self, *a): self.started=False; self.should_exit=False; self.force_exit=False; servers.append(self)
        def run(self): raise AssertionError('Recording HTTP must not listen')
    sys.modules['uvicorn'] = NS(Server=Server, Config=lambda *a, **k: NS(**k))
    import worker_r206b as worker
    import client_quality_pair as client_module
    from CPU_fake_hc import fake_lib, attach_client_hc
    class Response:
        status = 200
        headers = {'content-type': 'application/json'}
        def __init__(self, value): self.value=value
        def read(self): return self.value
        def __enter__(self): return self
        def __exit__(self, *a): pass
    scenarios = [(0, 'HC-hit-failure'), (0, 'client-failure')]
    for shard, scenario in scenarios:
        os.environ['DEEP_R375_REPLICA'] = str(shard)
        frozen = plan(); threads=[]; children=[]; state={'past': 0, 'prompt': 0}
        tag = 'CPU375-' + q['actual_step'].replace('.', '_') + '-r%d-' % shard + scenario
        eng = NS(_admission_lock=threading.Lock(), _fatal=None, finished=0, stopped=False)
        eng.stop = lambda: setattr(eng, 'stopped', True)
        class Req:
            def __init__(self, index):
                entry = frozen[index]; encoded = enc['cases'][entry['case_id']]
                self.request_id = 'CPU375-%d' % index
                private = pathlib.Path(encoded['private_IDs_path']); assert sha(private) == encoded['private_IDs_file_SHA256']
                self.prompt_token_ids = json.loads(private.read_bytes()); self.max_new_tokens=4096
                self.cancel_event=threading.Event(); self.abort_reason=None; self.generated_token_ids=[]
                self._admission_metadata_ready=True; self.finish_reason=None
                self.metadata={'diagnostic_user': 'Deep375-r%d-i%d-mode%d' % (shard, index, entry['arm']),
                    'accepted_monotonic': time.monotonic()-1,
                    'history_render_audit': {'qualified': True, 'prequeue_bound': True, 'markers_sent_to_Native': False,
                         'prompt_tokens': len(self.prompt_token_ids), 'Native_prompt_token_ids_SHA256': encoded['prompt_IDs_SHA256']}}
            def emit_token(self, token): self.generated_token_ids.append(token)
            def finish(self, reason): self.finish_reason=reason; eng.finished+=1
            def fail(self, error): self.finish_reason='error'; eng.finished+=1
        requests = [Req(i) for i in range(66)]
        def take(timeout):
            req=requests.pop(0); state['prompt']=len(req.prompt_token_ids); return req
        eng.take=take
        class Thread:
            def __init__(self, *a, **k): self.ident=None; self.alive=False; threads.append(self)
            def start(self): self.ident=1; self.alive=True; servers[-1].started=True
            def is_alive(self): return self.alive
            def join(self, timeout): assert eng.stopped; self.alive=False
        class Child:
            returncode=None
            def poll(self):
                if self.returncode is not None: return self.returncode
                if scenario=='client-failure': raise RuntimeError('recording client failure')
                if eng.finished != 66: return None
                actual_path=pathlib.Path(self.args[self.args.index('--service-report')+1]); actual=json.loads(actual_path.read_bytes()); seen=[]
                def transport(request, timeout):
                    body=json.loads(request.data); index=len(seen); r=actual['requests'][index]; entry=frozen[index]; case=cases[entry['case_id']]
                    assert body['messages']==messages(case) and body['max_tokens']==4096
                    assert body['user']==r['admission']['diagnostic_user']; seen.append(body)
                    text='deliberately unparseable' if index%3==0 else '#### '+str(case['gold_numeric'] if case['suite']=='GSM8K' else chr(65+case['answer']))
                    return Response(json.dumps({'id':r['request_id'],'choices':[{'message':{'role':'assistant','content':text},'finish_reason':'stop' if r['native_finish_reason']=='eos_token' else 'length'}],
                        'usage':{'prompt_tokens':r['prompt_tokens'],'completion_tokens':r['actual_generated_tokens'],'total_tokens':r['prompt_tokens']+r['actual_generated_tokens']}}).encode())
                argv=sys.argv; url=urllib.request.urlopen; sys.argv=self.args[2:]; urllib.request.urlopen=transport
                try: rc=client_module.main()
                finally: sys.argv=argv; urllib.request.urlopen=url
                result=json.loads((pathlib.Path(self.args[self.args.index('--output-dir')+1])/'result.json').read_bytes())
                assert rc==0 and result['passed'] and len(seen)==66
                assert result['actual_HTTP_attempts']==result['actual_HTTP_completed']==result['actual_Native_completed']==66
                assert sum(r['parse_failure'] for r in result['requests'])==22
                q['cases'].append(dict(kind='actual_worker_report_to_actual_client_main66', replica=shard, passed=True,
                      worker_identity_report_argv_real=True, HTTP_transport_recorded=True, completed_HTTP=66, unparseable_retained=22))
                if shard==0:
                    # Client-only cap fixture, explicitly modified recording.
                    # Do not present it as a Native/GPU-generated cap receipt.
                    actual=copy.deepcopy(actual); cap=actual['requests'][2]
                    cap.update(generated_token_ids=[7]*4096,actual_generated_tokens=4096,native_finish_reason='length')
                    cap['backend_token_monotonic']=[cap['started_monotonic']+.001+i*.00001 for i in range(4096)]
                    cap['ended_monotonic']=cap['backend_token_monotonic'][-1]+.001
                    prefill=[x for x in cap['Native_continuation'] if x['kind']=='prefill']
                    cap['Native_continuation']=prefill+[{'kind':'decode_summary','calls':4095,'first_past':cap['prompt_tokens'],'last_total':cap['prompt_tokens']+4095}]
                    cap['Native_forward_host_summary'].update(decode_calls=4095,decode_s=.04)
                    attach_client_hc(cap,frozen[2]['arm'])
                    for item in cap['HC_forward_coverage']: item['eligible_T1_BF16']=item['input_tokens']==1
                    cap_path=O/'cap-fixture'/actual_path.name; dump(cap_path,actual)
                    cap_args=self.args[2:].copy(); cap_args[cap_args.index('--service-report')+1]=str(cap_path)
                    cap_output=O/'cap-client'; cap_args[cap_args.index('--output-dir')+1]=str(cap_output)
                    seen=[]; argv=sys.argv; url=urllib.request.urlopen; sys.argv=cap_args; urllib.request.urlopen=transport
                    try: cap_rc=client_module.main()
                    finally: sys.argv=argv; urllib.request.urlopen=url
                    cap_result=json.loads((cap_output/'result.json').read_bytes())
                    assert cap_rc==0 and len(seen)==66 and cap_result['actual_HTTP_completed']==66
                    assert sum(r['cap_hit'] for r in cap_result['requests'])==1 and not cap_result['requests'][2]['correct']
                    q['cases'].append(dict(kind='actual_client_main_recorded4096cap_continues66',passed=True,Native_generated=False,cap_retained=1,completed_HTTP=66))
                self.returncode=rc; return rc
            def terminate(self): self.returncode=-15
            def kill(self): self.returncode=-9
            def wait(self, timeout): return self.returncode
        def popen(args, **kwargs):
            from client_owner_r281 import bind_identity
            identity=pathlib.Path(args[args.index('--identity')+1]); report=args[args.index('--service-report')+1]
            actual=bind_identity(identity, os.environ['DEEP_SOURCE_MANIFEST_SHA'], report)
            assert actual['actual_step']==tag and actual['replica']==shard
            assert actual['prepared_entry_identity']==enc['prepared_entry_identity'] and actual['comparison_contract']==enc['comparison_contract']
            bad=copy.deepcopy(actual); bad.pop('prepared_entry_identity'); badfile=O/('bad-identity-'+tag+'.json'); dump(badfile,bad)
            badargs=args[2:].copy(); badargs[badargs.index('--identity')+1]=str(badfile); badout=O/('bad-client-'+tag); badargs[badargs.index('--output-dir')+1]=str(badout)
            argv=sys.argv; url=urllib.request.urlopen; sys.argv=badargs; count=[0]
            def forbidden(*a,**k): count[0]+=1; raise AssertionError('Missing identity reached HTTP')
            urllib.request.urlopen=forbidden
            try: rc=client_module.main()
            finally: sys.argv=argv; urllib.request.urlopen=url
            badresult=json.loads((badout/'result.json').read_bytes())
            assert rc==1 and count==[0] and badresult['actual_HTTP_attempts']==0
            q['cases'].append(dict(kind='actual_worker_identity_to_actual_client_bind_and_main', replica=shard, scenario=scenario, passed=True, negative_HTTP=0))
            child=Child(); child.args=args; children.append(child); return child
        worker.threading=NS(Thread=Thread); worker.subprocess=NS(Popen=popen,STDOUT=subprocess.STDOUT)
        os.environ['DEEP_R277_GPUSTART_UNIX']='0' if scenario=='coldload' else str(time.time())
        def reset(ctx): state['past']=0; return 0
        def forward(ids, pos, rows, past, total, offsets):
            if scenario=='Native-failure': raise RuntimeError('recording Native failure')
            lib.advance(len(ids)); assert past[0]==state['past']; state['past']=total[0]
            token=1 if total[0]>=state['prompt']+2 else 8 if total[0]==state['prompt']+1 else 7
            return 0,np.zeros((1,129280),dtype=np.uint16),[token]
        lib=fake_lib(reset,scenario)
        prepared={'engine':eng,'FinishReason':reasons,'server':object(),'app':object(),'tokenizer':object(),'identity':enc['prepared_entry_identity']}
        namespace={'__latest_prepared_r277__':prepared,'tag':tag,'torch':NS(cuda=NS(mem_get_info=lambda i:(29*1024**3,64*1024**3))),
                   'np':np,'lib':lib,'ctx':None,'check':lambda code:None,'forward':forward,'cfg':{'vocab_size':129280,'num_hidden_layers':43},
                   'model':'/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8','__bootstrap_proof_r206b__':{'CPU_recording':True},
                   '__provider_proof_r206b__':{'CPU_recording':True},'__create_proof_r206b__':{'CPU_recording':True}}
        error=None
        try: worker.serve(namespace)
        except BaseException as e: error=repr(e)
        actual=json.loads((R/'results'/('Deep-paired-quality-r375-'+tag+'.json')).read_bytes())
        assert eng.stopped and not threads[0].is_alive() and not actual['cleanup']['HTTP_thread_alive_after_cleanup']
        assert (error is None)==(scenario=='success66'),error
        if scenario=='success66':
            assert actual['actual_HTTP']==actual['actual_Native']==actual['attempted_Native']==actual['completed_Native']==66
            assert actual['normal_server_shutdown'] and lib.state['set_modes']==[e['arm'] for e in frozen]
        if scenario in ['coldload','HC-ready-failure']: assert not children and threads[0].ident is None
        if scenario=='Native-failure': assert len(actual['requests'])==1 and actual['completed_Native']==0
        if scenario=='HC-hit-failure':
            first_on=next(i for i,e in enumerate(frozen) if e['arm']==1)
            assert len(actual['requests'])==first_on+1 and actual['completed_Native']==first_on
            assert 'HC366 actual forward coverage differs' in actual['requests'][-1]['error']
        q['cases'].append(dict(kind='actual_worker_serve_cleanup',replica=shard,scenario=scenario,passed=True,Native=len(actual['requests']),error=error))
    q['passed']=True
except BaseException as e:
    q.update(error=repr(e),traceback=traceback.format_exc())
q['completed']=True
dump(O/'CPU-entry.json',q)
print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error']}))
raise SystemExit(0 if q['passed'] else 1)
