"""Actual frozen six-request client entry; only HTTP is synthetic. No Model."""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import urllib.request
from unittest import mock

SHA = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve()
    expected = '3508075aa744f3bfb8f3631af73b1112f8905b0c810800b7ec0819885b718545'
    assert SHA(source / 'SOURCE_MANIFEST.json') == expected
    manifest = json.loads((source / 'SOURCE_MANIFEST.json').read_bytes())
    assert all(SHA(source / n) == h for n, h in manifest['files_SHA256'].items())
    sys.path.insert(0, str(source))
    def module(name):
        spec = importlib.util.spec_from_file_location('CPU307_' + name, source / (name + '.py'))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    client, driver = module('client_paired_r290'), module('client_driver_r290')
    assert client.PLAN == (2, 4, 2, 4, 4, 2)
    fixture = Path('/refactor/results/GLM-regular-live-r204-GLM-regular-service-r204-b34db8d1eb674914/HTTP/request-002.request.json')
    assert SHA(fixture) == client.FROZEN_REQUEST_SHA256
    body = json.loads(fixture.read_bytes())
    gold = client.validate_request(body)
    actual_cpu = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
    records = []
    for mode, expected_calls, expected_rc in [('positive', 6, 0), ('wrong-step', 0, 1),
                                               ('wrong-window', 3, 1), ('wrong-tool', 1, 1)]:
        tag = 'CPU-r307-' + actual_cpu.replace('.', '-') + '-' + mode
        root = Path('/refactor/results')
        owner_path = root / ('GLM-latest-full-owner-r290-' + tag + '-stage3-rank0.json')
        identity_path = root / (tag + '-identity.json')
        assert not owner_path.exists() and not identity_path.exists()
        handle = 'CPU_SYNTHETIC_GPU_NOT_SUBMITTED'
        identity = {'run': tag, 'actual_step': handle, 'Native_SHA256': client_owner_native(),
                    'TP': 8, 'PP': 4, 'cards': 32, 'rows': 1, 'capacity': 32768,
                    'source_manifest_SHA256': expected,
                    'prepared_entry_identity': {'upstream_commit': '99f5c90a40f9b2546bd0202cba3397ed1d55c5a0'}}
        owner = dict(identity, service_requests=[], queue_journal={'max_distinct_P1_sessions_unfinished': 1})
        if mode == 'wrong-step': identity['actual_step'] = 'WRONG_CPU_SYNTHETIC_IDENTITY'
        identity_path.write_text(json.dumps(identity))
        owner_path.write_text(json.dumps(owner))
        calls = 0
        def urlopen(request, timeout):
            nonlocal calls
            assert request.full_url == 'http://127.0.0.1:43843/v1/chat/completions'
            assert request.data == fixture.read_bytes() and timeout > 0
            index = calls; calls += 1
            assert calls <= 6
            window = client.PLAN[index]
            rid = 'CPU_ONLY_' + mode + '_' + str(index)
            t = time.monotonic()
            row = {'request_id': rid, 'admitted_index': index, 'pair_index': index,
                   'prefill_control_window': 1 if mode == 'wrong-window' and index == 2 else window,
                   'paired_warmup': index < 2, 'prompt_tokens': 8900, 'max_new_tokens': 4096,
                   'actual_generated_tokens': 2, 'generated_token_ids': [7, 154820],
                   'backend_token_monotonic': [t + .002, t + .003],
                   'started_monotonic': t + .001, 'ended_monotonic': t + .004,
                   'normal_reset_before': True, 'normal_reset_after': True,
                   'native_finish_reason': 'eos_token', 'final_metadata': {'CPU_synthetic': True},
                   'Native_continuation': [{'kind': 'prefill', 'tokens': 8900}],
                   'Native_forward_host_summary': {'prefill_calls': 18, 'prefill_s': .001, 'decode_calls': 1, 'decode_s': .001},
                   'actual_prompt_token_ids_SHA256': 'CPU_SYNTHETIC_NOT_REAL_TOKEN_PROOF',
                   'admission': {'diagnostic_user': body['user'], 'accepted_monotonic': t,
                                 'history_render_audit': {'qualified': True, 'prequeue_bound': True,
                                                         'markers_sent_to_Native': False, 'prompt_tokens': 8900}}}
            owner['service_requests'].append(row); owner_path.write_text(json.dumps(owner))
            wire = {'id': rid, 'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
                    'tool_calls': [{'id': 'CPU_ONLY_CALL', 'type': 'function', 'function': {
                    'name': 'wrong' if mode == 'wrong-tool' else 'calculate', 'arguments': json.dumps(gold)}}]}}],
                    'usage': {'prompt_tokens': 8900, 'completion_tokens': 2, 'total_tokens': 8902}}
            reply = io.BytesIO(json.dumps(wire).encode()); reply.status = 200; return reply
        def child(argv, **kwargs):
            assert Path(argv[2]).name == 'client_paired_r290.py'
            saved = sys.argv; sys.argv = [argv[2], *argv[3:]]
            try: rc = client.main()
            finally: sys.argv = saved
            return types.SimpleNamespace(returncode=rc)
        argv = [str(source / 'client_driver_r290.py'), '--run', tag, '--handle', handle,
                '--identity', str(identity_path), '--service-report', str(owner_path),
                '--manifest-sha', expected, '--deadline-unix', str(time.time() + 600)]
        saved = sys.argv; sys.argv = argv
        with mock.patch.object(urllib.request, 'urlopen', urlopen), mock.patch.object(subprocess, 'run', child), \
             mock.patch.dict(os.environ, {k: '-1' for k in ['CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']}):
            try: rc = driver.main()
            finally: sys.argv = saved
        assert rc == expected_rc and calls == expected_calls, (mode, rc, calls)
        result_path = root / ('GLM-latest-full-owner-r290-supervisor-' + tag + '.json')
        result = json.loads(result_path.read_bytes())
        assert result['completed'] and result['passed'] is (mode == 'positive')
        assert result['planned_HTTP'] == result['planned_Native'] == 6
        if mode == 'positive':
            assert result['actual_HTTP'] == result['actual_Native'] == 6
            assert list(result['phases']) == ['paired']
            assert len(result['phases']['paired']['result']['cases']) == 6
        stop = root / ('GLM-latest-full-owner-r290-stop-' + tag + '.json')
        assert stop.exists() and json.loads(stop.read_bytes())['actual_GPU_handle'] == handle
        records.append({'mode': mode, 'passed': True, 'synthetic_HTTP_attempts': calls,
                        'returncode': rc, 'result_SHA256': SHA(result_path), 'stop_marker_SHA256': SHA(stop)})
    assert 'torch' not in sys.modules
    output = {'completed': True, 'passed': True, 'CPU_only': True, 'actual_CPU': actual_cpu,
              'source_manifest_SHA256': expected, 'CPU_source_SHA256': SHA(__file__), 'cases': records,
              'real_driver_and_client_main_executed': True, 'actual_frozen_request_read': True,
              'mock_boundary': 'HTTP urlopen only; nested actual client main runs in process instead of child spawn',
              'GPU': 0, 'Model': 0, 'Native_dlopen': 0, 'production_HTTP': 0,
              'synthetic_gold_tool_arguments_are_not_model_quality': True,
              'qualification': 'Six-request client orchestration and negative controls only, no actual Model/CCL/device/numerical/performance proof'}
    assert not args.output.exists(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


def client_owner_native():
    return '038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'


if __name__ == '__main__': main()
