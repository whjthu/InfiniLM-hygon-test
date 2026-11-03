"""Actual public factory/queue/AR/four-leader socket glue; device leaves recorded."""
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
import socket
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from public_kimi_native.prepared import prepare
from public_kimi_native.provider import KimiProvider
from public_kimi_native.request_loop import serve
from public_kimi_native.runtime.long_ar_append_r248 import generate
from public_kimi_native.runtime.stage_control_r2386 import StageControl


class Reason(str, Enum):
    CANCELED='canceled';ERROR='error';TIMEOUT='timeout';EOS_TOKEN='eos_token';LENGTH='length'


class CodecLeaf:
    identity = dict(CPU_recording=True, actual_checkpoint_tokenizer_executed=False)
    def encode_chat(self, messages, kwargs, generation):
        prompt = ''.join('<' + m['role'] + '>' + str(m.get('content', '')) + str(m.get('reasoning_content', '')) for m in messages)
        prompt += '<|open|>think<|sep|>' if generation else ''
        return prompt, list(prompt.encode())
    def decode(self, ids): return ''.join(chr(i) for i in ids if i < 128)


def run():
    root = Path(__file__).resolve().parent
    public = root.parent / 'public-GLM-native-provider-r425'
    sys.path.insert(0, str(public))
    from public_glm_native.prepared_queue import bind_public_sampling
    cases = []
    histories = [dict(role='user', content='complete-' + 'x' * 600),
                 dict(role='assistant', content='prior answer', reasoning_content='preserved reasoning'),
                 dict(role='tool', content='full integer result 12345')]
    prepared = prepare('/qualified/model', Reason, CodecLeaf())
    engine = prepared['engine']
    params = SimpleNamespace(temperature=0.0, max_tokens=4096, stop=None)
    request = engine.add_chat_request(messages=histories, sampling_params=params, request_id='complete-history',
        request_data=dict(messages=histories, max_tokens=4096), chat_template_kwargs={})
    canceled = engine.add_chat_request(messages=[dict(role='user', content='cancel before Native')],
        sampling_params=params, request_id='cancel-before-native', request_data=dict(max_tokens=4096))
    canceled.mark_abort(Reason.CANCELED)
    assert request.metadata['history_render_audit']['input_RC_fields'] == 1
    assert request.max_new_tokens == 4096 and len(request.prompt_token_ids) > 512

    class NativeLeaf:
        def __init__(self, rank):
            self.rank = rank
            self.calls = []
            self.resets = []
            self.memory_guard = None
            self.model_events = SimpleNamespace(set_next=lambda *a: None)
            self.input_commit_progress = None
        def reset(self, state): self.resets.append(state)
        def arm(self, *args): self.final_arm = args
        def call(self, kind, ids, past, tokens, capture):
            assert ids == [1] and not capture
            self.calls.append(dict(kind=kind, past=past, width=len(tokens[0]), CAPI_maxhost_s=0.0))
            return [163585 if kind == 'verify' else 65], None, {}
        def call_append(self, logical, past, tokens):
            assert logical == 1 and 1 <= len(tokens) <= 512
            self.calls.append(dict(kind='append', past=past, width=len(tokens), CAPI_maxhost_s=0.0))
            return [65], None, {}
        def commit(self, ids, decisions): self.committed = ids, decisions

    probe = socket.socket();probe.bind(('127.0.0.1', 0));port = probe.getsockname()[1];probe.close()
    native = [NativeLeaf(i) for i in range(4)]
    reports = [dict() for _ in range(4)]
    peer_errors = []
    closed = []

    def owner(config, same_prepared, stop, on_ready):
        node = config['node']
        assert (same_prepared is prepared) == (node == 0)
        control = StageControl(node, '127.0.0.1', port, 'actual-local-four-leader-CPU', 4, 10)
        try:
            from public_kimi_native.admission import admit
            admit(control, config, 'CPU-identical-model-manifest')
            # Native create/load/finish are recorded here; no actual devices or
            # objects. The public/thread/queue/AR/control code below is real.
            ids = control.collect('recorded-create-load-finish', dict(node=node, owners=8))
            assert [v['node'] for v in ids] == list(range(4)) and sum(v['owners'] for v in ids) == 32
            on_ready()
            g = dict(node=node, control=control, report=reports[node], save=lambda: None,
                service_deadline_monotonic=time.monotonic()+10)
            serve(g, same_prepared, stop, native=native[node], generate=generate)
            assert reports[node]['serving_protocol_complete']
            control.collect('recorded-normal-destroy', dict(node=node))
            closed.append(node)
        finally:
            control.close()

    config = dict(node=0, model_path='/qualified/model', gpu_start_epoch=time.time(),
        service_seconds=10, Native_seconds=610, run='four-real-leaders',
        source_manifest_SHA256='CPU-source-identity', config_SHA256='CPU-config-identity')
    provider = KimiProvider(config, prepared, owner=owner)
    peers = []
    for node in [1, 2, 3]:
        def peer(n=node):
            try: owner(dict(config, node=n), None, threading.Event(), lambda: None)
            except BaseException as error: peer_errors.append(repr(error))
        thread = threading.Thread(target=peer)
        peers.append(thread);thread.start()
    source = public / 'public-python/infinilm/server/inference_server.py'
    cls = next(n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == 'InferenceServer')
    cls.body = [n for n in cls.body if getattr(n, 'name', None) in ['__init__', '_create_app', '_build_sampling_params']]
    class AppLeaf:
        def __init__(self, *, lifespan): self.lifespan = lifespan
    generic_calls = []
    def generic(**kw):
        generic_calls.append(kw)
        raise AssertionError('Generic model must not be constructed')
    ns = dict(os=os, Optional=__import__('typing').Optional, KVTransferConfig=object,
        TOOL_CALL_FORMATS={'kimi_k3': None}, AsyncLLMEngine=generic, FastAPI=AppLeaf,
        asynccontextmanager=asynccontextmanager, logger=logging.getLogger('public428'),
        SamplingParams=SimpleNamespace, DEFAULT_STREAM_TIMEOUT=100., DEFAULT_REQUEST_TIMEOUT=1000.)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), 'exec'), ns)
    Server = ns['InferenceServer']
    Server._register_routes = lambda self, app: None
    server = Server(model_path='/qualified/model', dtype='bfloat16', tensor_parallel_size=16,
        pipeline_parallel_size=2, pipeline_parallel_stage=0, cache_type='static', max_tokens=4096,
        max_batch_size=1, max_cache_len=32768, temperature=0., top_p=1., top_k=1,
        enable_prefix_caching=False, tool_call_parser='kimi_k3', engine_factory=provider,
        stream_timeout_s=3600, request_timeout_s=10800)
    bind_public_sampling(server, engine)
    try: server._build_sampling_params(dict(messages=histories, max_tokens=4.0))
    except ValueError: pass
    else: raise AssertionError('Float budget accepted')
    app = server._create_app()

    async def lifespan():
        async with app.lifespan(app):
            assert server.engine is provider
            until = time.monotonic() + 10
            while not (request.is_finished() and canceled.is_finished()):
                if provider.failure is not None: raise provider.failure
                assert time.monotonic() < until
                await asyncio.sleep(.01)
    asyncio.run(lifespan())
    for thread in peers: thread.join(timeout=3)
    assert not peer_errors and all(not t.is_alive() for t in peers)
    assert closed and sorted(closed) == list(range(4))
    assert not generic_calls and provider.closed and not provider.is_healthy()
    assert request.finish_reason == Reason.EOS_TOKEN and request.generated_token_ids == [65, 163585]
    assert canceled.finish_reason == Reason.CANCELED and canceled.generated_token_ids == []
    for node in range(4):
        assert native[node].calls == native[0].calls
        assert native[node].resets == [False] * 4
        assert len(reports[node]['agent_requests']) == 2
        assert all(r['normal_reset'] for r in reports[node]['agent_requests'])
    cases.append(dict(case='actual_public_lifespan_factory_samequeue_four_real_TCP_leaders_original_AR_reset_then_finish',
        passed=True, recording_leaves=['checkpoint codec render/tokenize', 'Native create/load/finish/destroy/forward', 'FastAPI constructor/routes'],
        local_real_sockets=4, actual_Native_or_SDK=False, NativeRequest_IPC=False))
    for settings in [dict(tensor_parallel_size=8), dict(pipeline_parallel_stage=1)]:
        wrong = KimiProvider(config, prepared, owner=owner)
        actual = dict(model_path='/qualified/model', device='cuda', dtype='bfloat16',
            tensor_parallel_size=16, pipeline_parallel_size=2, pipeline_parallel_stage=0,
            cache_type='static', max_batch_size=1, max_cache_len=32768, max_tokens=4096,
            temperature=0., top_p=1., top_k=1, enable_graph=False, skip_load=False,
            enable_prefix_caching=False, attn_backend='default', use_mla=False,
            moe_ep_backend='disabled', moe_ep_size=1, weight_load_mode='async',
            kv_transfer_config=None, pre_transpose=False)
        actual.update(settings)
        try: wrong(**actual)
        except AssertionError: pass
        else: raise AssertionError('Wrong topology accepted')
    cases.append(dict(case='TP8PP4_or_frontend_sampling_node_confusion_rejected_before_owner', passed=True))
    try: provider.start()
    except RuntimeError: pass
    else: raise AssertionError('Provider restarted')
    cases.append(dict(case='no_factory_or_owner_replay', passed=True))
    failed_prepared = prepare('/qualified/model', Reason, CodecLeaf())
    waiting = failed_prepared['engine'].add_chat_request(messages=[dict(role='user', content='pending')],
        sampling_params=params, request_id='startup-failure', request_data=dict(max_tokens=4096))
    primary = RuntimeError('recorded Native loader failure')
    def failed_owner(*args): raise primary
    failed = KimiProvider(config, failed_prepared, owner=failed_owner)
    try: failed.start()
    except RuntimeError as error: assert error is primary
    else: raise AssertionError('Owner startup failure hidden')
    try: failed.stop()
    except RuntimeError as error: assert error is primary
    else: raise AssertionError('Owner failure changed or hidden at cleanup')
    assert waiting.is_finished() and waiting.finish_reason == Reason.ERROR and failed.closed
    assert not failed.thread.is_alive()
    cases.append(dict(case='actual_provider_start_failure_preserves_primary_fails_pending_and_joins_own_thread',
        passed=True, recording_leaf='Native owner load body'))
    pending_prepared = prepare('/qualified/model', Reason, CodecLeaf())
    pending = pending_prepared['engine'].add_chat_request(messages=[dict(role='user', content='queued')],
        sampling_params=params, request_id='pending-stop', request_data=dict(max_tokens=4096))
    drained = threading.Event()
    def stop_owner(config, same_prepared, stop, ready):
        ready()
        assert stop.wait(3)
        assert pending.cancel_event.is_set() and not pending.is_finished()
        drained.set()  # Recorded global destroy completion, before Python thread exit.
    stopping = KimiProvider(config, pending_prepared, owner=stop_owner)
    stopping.start();stopping.request_stop()
    try:
        pending_prepared['engine'].add_chat_request(messages=[dict(role='user', content='late')],
            sampling_params=params, request_id='late', request_data=dict(max_tokens=4096))
    except RuntimeError: pass
    else: raise AssertionError('Admission reopened after stop intent')
    stopping.stop()
    assert drained.is_set() and pending.is_finished() and pending.finish_reason == Reason.CANCELED
    cases.append(dict(case='atomic_admission_close_cancel_intent_then_pending_finish_only_after_owner_join',
        passed=True, recording_leaf='Native global destroy completion'))
    from public_kimi_native.cleanup import close_owner
    for sampler_error in [False, True]:
        actions = []
        def sample_close():
            actions.append('sampler-close')
            if sampler_error: raise RuntimeError('recorded sampler failure')
        sampler = SimpleNamespace(close=sample_close, thread=SimpleNamespace(is_alive=lambda: False))
        lib = SimpleNamespace(modern_kimi_engine_destroy_r2475=lambda ctx: actions.append(('destroy', ctx)) or 0)
        control = SimpleNamespace(collect=lambda *a: actions.append('four-leader-ack'), close=lambda: actions.append('control-close'))
        report = dict(protocol_before_destroy_passed=True)
        primary = RuntimeError('recorded original request failure') if sampler_error else None
        close_owner(ctx='same-original-context', lib=lib, check=lambda rc: None,
            sampler=sampler, control=control, events=None, report=report,
            save=lambda: actions.append('saved'), primary=primary)
        assert ('destroy', 'same-original-context') in actions and 'control-close' in actions
        assert report['completed'] is (not sampler_error)
        assert ('four-leader-ack' in actions) is (not sampler_error)
    cases.append(dict(case='actual_owner_cleanup_sampler_error_cannot_skip_Native_destroy_control_close_or_primary',
        passed=True, recording_leaves=['Native destroy', 'sampler close', 'four-leader destroy ACK']))
    from public_kimi_native.admission import admit
    identity = {k: config[k] for k in ['run', 'source_manifest_SHA256', 'config_SHA256', 'gpu_start_epoch']}
    foreign = [dict(node=i, manifest='same-model', identity=dict(identity)) for i in range(4)]
    foreign[2]['identity']['source_manifest_SHA256'] = 'foreign-source'
    try: admit(SimpleNamespace(collect=lambda *args: foreign), config, 'same-model')
    except AssertionError: pass
    else: raise AssertionError('Foreign source leader admitted to Native create')
    cases.append(dict(case='actual_precreate_admission_rejects_foreign_leader_source_before_Native', passed=True))
    import tempfile
    from public_kimi_native.cli import watch_shutdown
    for wrong_source in [False, True]:
        with tempfile.TemporaryDirectory() as tmp:
            stopped_prepared = prepare('/qualified/model', Reason, CodecLeaf())
            stop_config = dict(config, runtime_directory=tmp)
            stopped = KimiProvider(stop_config, stopped_prepared, owner=owner)
            packet = dict(identity={k: stop_config[k] for k in ['run', 'source_manifest_SHA256', 'config_SHA256', 'gpu_start_epoch']}, action='stop')
            if wrong_source: packet['identity']['source_manifest_SHA256'] = 'wrong-source'
            (Path(tmp) / ('STOP-' + config['run'] + '.json')).write_text(json.dumps(packet))
            http = SimpleNamespace(should_exit=False)
            watch_shutdown(stopped, http, threading.Event())
            assert http.should_exit and stopped.stop_intent.is_set()
            assert (stopped.failure is not None) is wrong_source
    cases.append(dict(case='actual_public_watcher_same_run_STOP_and_foreign_STOP_trip_both_exit_HTTP',
        passed=True, recording_leaf='HTTP should_exit property'))
    # Consume the final actual argparse -> package/hash -> configure path for
    # all four roles. Only Linux hostname/env and POSIX template file I/O are
    # recorded; every source/config byte read by verification is real.
    from public_kimi_native.cli import main
    config_bytes = (root / 'config.template.json').read_bytes()
    config = json.loads(config_bytes)
    model_bytes = (root / 'qualified-model.template.json').read_bytes()
    original_read = Path.read_bytes
    def mapped_read(path):
        return model_bytes if str(path).replace('\\', '/') == config['model_manifest'] else original_read(path)
    source_sha = hashlib.sha256((root / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'config.json';path.write_bytes(config_bytes)
        for node in range(4):
            env = dict(SLURM_PROCID=str(node), SLURM_JOB_ID='CPU', SLURM_STEP_ID='recording')
            with patch.dict(os.environ, env, clear=True), patch.object(Path, 'read_bytes', mapped_read), patch('socket.gethostname', return_value=config['nodes'][node]):
                main(['--config', str(path), '--config-sha', hashlib.sha256(config_bytes).hexdigest(),
                      '--source-sha', source_sha, '--node', str(node), '--run', 'CPU-only',
                      '--nonce', 'CPU-only-nonce-no-model-or-auth-qualification',
                      '--actual-step', 'CPU.recording', '--gpu-start-epoch', '1'])
                assert os.environ['KIMI_CAPACITY_ONLY_R9145'] == '0'
                assert os.environ['INFINILM_KIMI_SPEC_MATH_R3676'] == '0'
                assert os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665'] == '0'
    cases.append(dict(case='actual_final_CLI_four_role_source_config_topology_and_fixed_AR_policy', passed=True,
        recording_leaves=['Linux hostname/env', 'POSIX model manifest file mapping'], Native_or_Torch_import=False))
    return dict(passed=True, cases=cases, GPU=False, Slurm_submitted=0, Native_dlopen=False,
        weights_loaded=False, actual_checkpoint_encoding=False, actual_public_module_import=False,
        scope='Changed glue only; extracted real public constructor/lifespan/sampling, real prepared queue/history, four actual StageControl sockets and original AR producer. All device/model/codec leaves recorded; no math qualification.')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try: q = run()
    except BaseException as error:
        import traceback
        a.output.write_text(json.dumps(dict(passed=False, error=repr(error), traceback=traceback.format_exc(),
            GPU=False, Slurm_submitted=0, Native_dlopen=False), indent=2)+'\n', encoding='utf-8')
        raise
    a.output.write_text(json.dumps(q, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=q['passed'], cases=len(q['cases']), SHA256=hashlib.sha256(a.output.read_bytes()).hexdigest())))
