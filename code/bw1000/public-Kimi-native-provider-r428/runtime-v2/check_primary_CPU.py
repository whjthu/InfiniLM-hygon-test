"""Only changed primary/cleanup error branch, actual serve; no old9 replay."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import threading
import time

from public_kimi_native.request_loop import serve
from public_kimi_native.runtime.native_engine_bridge_r143 import NativeRequest
from public_kimi_native.runtime.history_prequeue_r221 import ids_SHA256


def run():
    primary = RuntimeError('recorded Native primary')
    reset_error = RuntimeError('recorded final reset secondary')
    save_error = RuntimeError('recorded persistence secondary')
    reasons = SimpleNamespace(ERROR='error', CANCELED='canceled', TIMEOUT='timeout')
    req = NativeRequest('primary-case', 'full prompt', [1, 2], 4096, {163585}, reasons)
    req.metadata['history_render_audit'] = dict(qualified=True, Native_prompt_token_ids_SHA256=ids_SHA256([1, 2]))
    control_calls, reset_calls, arm_calls, saves = [], [], [], []
    class Control:
        def collect(self, label, value):
            control_calls.append(label)
            assert label.startswith('public-command:'), 'Do not add a reset collect after failed reset'
            return [value]
    class Native:
        calls = []
        memory_guard = None
        input_commit_progress = None
        model_events = SimpleNamespace(set_next=lambda *a: None)
        def reset(self, state):
            reset_calls.append(state)
            if len(reset_calls) == 2: raise reset_error
        def arm(self, *args): arm_calls.append(args)
    def save():
        saves.append('save')
        raise save_error
    def generate(*args): raise primary
    engine = SimpleNamespace(take=lambda timeout: req)
    report = {}
    g = dict(node=0, control=Control(), report=report, save=save,
        service_deadline_monotonic=time.monotonic() + 3)
    try: serve(g, dict(engine=engine, FinishReason=reasons), threading.Event(), native=Native(), generate=generate)
    except RuntimeError as error: assert error is primary
    else: raise AssertionError('Primary swallowed')
    assert reset_calls == [False, False] and len(arm_calls) == 1 and len(saves) == 2
    assert len(control_calls) == 1 and req.finish_reason == 'error'
    item = report['agent_requests'][0]
    assert item['error'] == repr(primary) and item['reset_error'] == repr(reset_error)
    assert [v['phase'] for v in item['cleanup_errors']] == ['request-reset', 'request-save']
    assert report['service_cleanup_errors'] == [dict(phase='service-save', error=repr(save_error))]
    return dict(passed=True, actual_consumer='public_kimi_native.request_loop.serve',
        primary_identity_preserved=True, reset_calls=2, capture_disable_calls=1,
        reset_collect_calls=0, save_attempts=2, injected_errors=['Native primary', 'final reset', 'request/service persistence'],
        original9cases_repeated=False, Native_or_SDK_executed=False, GPU=False, Slurm_submitted=0,
        math_or_reset_collective_schedule_changed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True);a = p.parse_args()
    q = run();a.output.write_text(json.dumps(q, indent=2)+'\n', encoding='utf-8');print(json.dumps(q))
