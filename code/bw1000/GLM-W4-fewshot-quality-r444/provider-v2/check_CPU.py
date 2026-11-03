"""New public/actor handoffs only; no repeat of r422/r423/r424 suites.

Actual production Actors/GLMProvider/Run/public lifespan/queue are consumed.
Native owner bodies and Process execution are recording leaves. No Native or
installed public imports, device, model payload or external job is executed.
"""
import argparse
import ast
import asyncio
from contextlib import asynccontextmanager
from enum import Enum
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from public_glm_native import actors
from public_glm_native.protocol import Run
from public_glm_native.provider import GLMProvider
from public_glm_native.prepared_queue import prepare_engine, bind_public_sampling
from public_glm_native.service_policy import bind_prequeue
from public_glm_native.owner_gates import NativeGates


class Reason(str, Enum):
    CANCELED='canceled';ERROR='error';TIMEOUT='timeout';EOS_TOKEN='eos_token';LENGTH='length'


def run(template_path):
    root=Path(__file__).parent
    template=Path(template_path).read_text()
    assert hashlib.sha256(template.encode()).hexdigest()=='3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c'
    class Processor:
        def get_tokenizer(self):return self
        def get_chat_template(self):return template
        def apply_chat_template(self,messages,*,tokenize,add_generation_prompt,**kwargs):
            value=''.join('<'+m['role']+'>'+str(m.get('content','')) for m in messages)
            return value+('<|assistant|><think>' if add_generation_prompt else '')
        def __call__(self,prompt,return_tensors):return {'input_ids':[[ord(c)for c in prompt]]}
        def decode(self,ids,**kwargs):return ''.join(chr(v)for v in ids)
    cases=[]
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp);model=path/'model';model.mkdir()
        (model/'generation_config.json').write_text(json.dumps(dict(eos_token_id=[154820])))
        cfg=json.loads((root/'config.template.json').read_text())
        cfg.update(runtime_directory=str(path),run='new-public-glue',actual_step='CPU.recording',
            source_manifest_SHA256='CPU-recording-not-device',gpu_start_epoch=time.time(),
            provider_config_SHA256='CPU-recording-provider-config',
            stage=1,model_path=str(model))
        prepared=prepare_engine('GLM',model,processor=Processor(),finish_reasons=Reason)
        bind_prequeue(prepared['engine'],cfg['profile'])
        bridge=prepared['engine']
        request=bridge.add_chat_request(messages=[dict(role='user',content='complete input')],
            sampling_params=SimpleNamespace(temperature=0.0,max_tokens=128,stop=None),
            request_id='same-process-locks',request_data=dict(max_tokens=128,_agent_schemas={},_agent_choice='none'))
        native_run=Run(cfg)
        native_run.write('COMMUNICATORS.json',dict(payload=dict(world='base64-world',tp=['a','b'],control='c')))
        native_run.write('PREPARED.json',dict(prepared_identity=prepared['identity']))
        def ready_row(stage,rank):
            owners=[dict(device=i,changes=int(i==rank),verified_mask=','.join(str(v)for v in range(rank*4,rank*4+4)) if i==rank else '')for i in range(8)]
            native_run.write('ready-%d-%d.json'%(stage,rank),dict(stage=stage,rank=rank,
                Native_SHA256=cfg['Native_SHA256'],config_SHA256=cfg['config_SHA256'],
                loaded=True,postload_4GiB=True,service_agreement=True,
                affinity=dict(actual_owner_report=dict(owners=owners))))
        for rank in range(8):
            ready_row(0,rank)
            native_run.write('closed-0-%d.json'%rank,dict(normal=True,post_close_Torch_liveness=True))
        process_records=[]
        class ProcessLeaf:
            def __init__(self,*,target,args,name):
                assert target is actors.process_owner
                assert len(args)==2 and isinstance(args[0],int)
                json.dumps(args[1])  # queue/NativeRequest cannot be serialized here
                self.target,self.args,self.name=target,args,name
                self.pid=10000+args[0];self.exitcode=None;self.alive=False
                process_records.append(self)
            def start(self):
                self.alive=True
                ready_row(1,self.args[0])
            def is_alive(self):return self.alive
            def join(self,timeout=None):pass
            def terminate(self):self.alive=False;self.exitcode=-15
            def kill(self):self.alive=False;self.exitcode=-9
        provider=GLMProvider(cfg,prepared)
        group=provider.group
        stop_seen=threading.Event();allow_drain=threading.Event();owner_completed=threading.Event()
        owner_observed={}
        def owner_leaf(rank,config,actual_prepared,stop_intent):
            assert rank==0 and actual_prepared is prepared
            assert actual_prepared['engine'] is bridge
            ready_row(1,0)
            assert stop_intent.wait(3),'stop event not delivered to emitter owner'
            with bridge._admission_lock:
                owner_observed.update(closed=not bridge.is_healthy(),
                    canceled=request.cancel_event.is_set(),finished=request.is_finished())
            stop_seen.set()
            assert allow_drain.wait(3)
            # The unchanged real queue method is only called after this
            # recorded collective-drain boundary, never by public stop itself.
            bridge.cancel_pending_after_global_drain(Reason.CANCELED)
            for process in group.processes:
                process.alive=False;process.exitcode=0
                native_run.write('closed-1-%d.json'%process.args[0],dict(normal=True,post_close_Torch_liveness=True))
            native_run.write('closed-1-0.json',dict(normal=True,post_close_Torch_liveness=True))
            owner_completed.set()
        # Actual changed public constructor/lifespan/sampling methods, only app
        # construction and route registration recorded; no installed module stub.
        public=root/'public-python/infinilm/server/inference_server.py'
        cls=next(n for n in ast.parse(public.read_bytes()).body if isinstance(n,ast.ClassDef)and n.name=='InferenceServer')
        cls.body=[n for n in cls.body if getattr(n,'name',None)in['__init__','_create_app','_build_sampling_params']]
        class App:
            def __init__(self,*,lifespan):self.lifespan=lifespan
        generic_calls=[]
        def generic(**kw):generic_calls.append(kw);raise AssertionError('generic registry must not be selected')
        ns=dict(os=os,Optional=__import__('typing').Optional,KVTransferConfig=object,TOOL_CALL_FORMATS={'glm5':None},
            AsyncLLMEngine=generic,FastAPI=App,asynccontextmanager=asynccontextmanager,
            logger=logging.getLogger('public425'),SamplingParams=SimpleNamespace,
            DEFAULT_STREAM_TIMEOUT=100.,DEFAULT_REQUEST_TIMEOUT=1000.)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(public),'exec'),ns)
        Server=ns['InferenceServer'];Server._register_routes=lambda self,app:None
        server=Server(model_path=str(model),device='cuda',dtype='bfloat16',tensor_parallel_size=8,
            pipeline_parallel_size=2,pipeline_parallel_stage=1,cache_type='static',max_tokens=4096,
            max_batch_size=1,max_cache_len=32768,temperature=0.,top_p=1.,top_k=1,
            enable_prefix_caching=False,tool_call_parser='glm5',engine_factory=provider,
            stream_timeout_s=3600,request_timeout_s=7200)
        bind_public_sampling(server,bridge)
        server.ignore_eos=True
        try:server._build_sampling_params(dict(messages=[dict(role='user',content='x')],
            _agent_schemas={},_agent_choice='none',max_tokens=128))
        except ValueError:pass
        else:raise AssertionError('unsupported ignore_eos silently accepted')
        server.ignore_eos=False
        app=server._create_app()
        async def lifespan():
            async with app.lifespan(app):
                assert provider.is_healthy() and len(group.processes)==7
                assert native_run.read('MODEL_READY.json')['actual_owner_count']==16
                assert not request.is_finished()
                assert server.stream_timeout_s==3600 and server.request_timeout_s==7200
                def permit():
                    assert stop_seen.wait(3)
                    assert owner_observed==dict(closed=True,canceled=True,finished=False)
                    allow_drain.set()
                watcher=threading.Thread(target=permit)
                watcher.start()
                # Exiting the actual public lifespan calls provider.stop, whose
                # stop event cancels active intent before the owner drains.
            watcher.join(timeout=3)
        with patch.object(actors.multiprocessing,'get_context',return_value=SimpleNamespace(Process=ProcessLeaf)),patch.object(actors,'owner_main',side_effect=owner_leaf):
            asyncio.run(lifespan())
        assert owner_completed.is_set() and request.is_finished() and request.finish_reason==Reason.CANCELED
        assert not generic_calls and not provider.is_healthy()
        assert all(r.args[1]is not prepared and len(r.args)==2 for r in process_records)
        cases.append(dict(case='actual_public_lifespan_real_actor_supervisor_same_queue_and_bounded_stop',passed=True,
            recording_leaves=['Native owner body','spawn Process execution','FastAPI constructor/routes'],
            NativeRequest_crossed_multiprocessing=False))

        # A duplicate same-label group cannot poison the old run's markers.
        second=actors.Actors(cfg,prepared)
        try:second.start()
        except FileExistsError:pass
        else:raise AssertionError('duplicate leader accepted')
        assert not second._claimed
        second._trip(RuntimeError('duplicate'))
        assert not (native_run.path/'FAILED.json').exists()
        cases.append(dict(case='duplicate_intent_refuses_before_actor_start_without_old_run_poison',passed=True))

        badcfg=dict(cfg,run='failure-propagation')
        failed=actors.Actors(badcfg,prepared);failed._claimed=True
        reason=RuntimeError('Native owner failure')
        failed._trip(reason)
        assert failed.failed.is_set() and failed.failure is reason and bridge._fatal is reason
        assert failed.run.read('FAILED.json')['error']==repr(reason)
        cases.append(dict(case='actual_actor_failure_marker_and_frontend_fatal_propagation',passed=True))

        stale=dict(cfg,source_manifest_SHA256='different-source')
        try:Run(stale).read('MODEL_READY.json')
        except AssertionError:pass
        else:raise AssertionError('foreign source record accepted')
        cases.append(dict(case='actual_readiness_identity_rejects_source_change',passed=True))

        # Exact postdestroy cache field positions: eviction history is allowed;
        # nonzero active bytes must fail. This consumes the real new gate.
        class Fn:
            def __init__(self,values):self.values=values
            def __call__(self,*a):
                out=a[-2]
                for i,v in enumerate(self.values):out[i]=v
                return 0
        lib=SimpleNamespace(modern_glm_fp8_runtime_stats_r809=Fn([8,8,0,0,99,0,0,88,1,0]),
            modern_glm_m15_oproj_stats_r125=Fn([0]*17),modern_glm_projected_host_counts_r4504=Fn([0]*7),
            modern_glm_cross_journal_r5775=lambda *a:b'{"identities":[]}')
        gates=NativeGates(torch=None,rank=0,stage=0,stream=None,config={},publish_failure=lambda error:None)
        report={};gates.after_destroy(SimpleNamespace(lib=lib),report)
        assert report['cache_after_destroy']['evictions']==99
        lib.modern_glm_fp8_runtime_stats_r809.values[6]=1
        try:gates.after_destroy(SimpleNamespace(lib=lib),{})
        except AssertionError:pass
        else:raise AssertionError('active bytes leak accepted')
        cases.append(dict(case='actual_postdestroy_gate_keeps_evictions_but_rejects_active_byte_leak',passed=True))

    for p in (root/'public_glm_native').glob('*.py'):
        tree=ast.parse(p.read_bytes())
        local={q.stem for q in (root/'public_glm_native').glob('*.py')}
        assert not any(isinstance(n,ast.ImportFrom)and n.level==0 and n.module in local for n in ast.walk(tree))
        assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id in['exec','eval']for n in ast.walk(tree))
    return dict(passed=True,cases=cases,CPU_only=True,GPU=False,Native_dlopen=False,
        actual_Native_owners_constructed=0,checkpoint_payload_read=False,Model_READY=False,jobs=0,
        old_422_423_424_suites_repeated=False,
        actual_New_provider_and_public_lifespan_consumed=True,
        source_only_limits=['actual CLI installed imports/Native cold construction not executed',
            'worker Native owner body and spawn execution are recording leaves',
            'all16 readiness files in this local test are recording actor outputs, not device evidence'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--template',required=True);a=parser.parse_args()
    result=run(a.template)
    (Path(__file__).parent/'local-CPU-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
