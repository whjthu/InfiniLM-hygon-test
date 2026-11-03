"""Four leaders execute original AR; only node0 owns Python NativeRequests."""
import os
import queue
import time


def serve(g, prepared, stop, *, native=None, generate=None):
    if native is None:
        from .runtime.native_append_r248 import Native
        native = Native(g)
    if generate is None:
        from .runtime.long_ar_append_r248 import generate
    from .runtime.history_prequeue_r221 import ids_SHA256
    node, control, report, save = [g[k] for k in ['node', 'control', 'report', 'save']]
    engine = prepared['engine'] if node == 0 else None
    reasons = prepared['FinishReason'] if node == 0 else None
    deadline = g['service_deadline_monotonic']
    requests = report.setdefault('agent_requests', [])
    ordinal = 0
    try:
        while True:
            command = req = None
            if node == 0:
                if stop.is_set() or time.monotonic() >= deadline:
                    command = dict(action='stop')
                else:
                    try:
                        req = engine.take(timeout=.2)
                        command = dict(action='stop') if req is None else dict(action='request',
                            id=req.request_id, prompt_ids=list(req.prompt_token_ids),
                            max_new_tokens=req.max_new_tokens, cancelled=req.cancel_event.is_set(),
                            history_render_audit=dict(req.metadata['history_render_audit']))
                    except queue.Empty:
                        command = dict(action='idle')
            command = control.collect('public-command:%d' % ordinal, dict(command=command))[0]['command']
            ordinal += 1
            if command['action'] == 'stop': break
            if command['action'] == 'idle': continue
            assert command['action'] == 'request'
            prompt, cap, ident = [command[k] for k in ['prompt_ids', 'max_new_tokens', 'id']]
            assert 1 <= len(prompt) < 32768 and 1 <= cap <= 8192 and len(prompt) + cap <= 32768
            assert all(type(i) is int and 0 <= i < 163840 for i in prompt)
            h = command['history_render_audit']
            assert h['qualified'] and h['Native_prompt_token_ids_SHA256'] == ids_SHA256(prompt)
            item = dict(request_id=ident, input_tokens=len(prompt), output_budget=cap,
                history_render_audit=h, prefill_mode='W512', generated_ids=[],
                started_monotonic=time.monotonic(), normal_reset=False)
            requests.append(item)
            failure = None
            ended = None
            observation_start = len(getattr(native, 'long_prefill_observations', []))
            try:
                native.model_events.set_next('public-' + ident, False)
                native.reset(False)
                # Exact qualified AR policy, fixed for every ordinary request.
                os.environ['INFINILM_KIMI_I192_GROUPED_R8669Y'] = '0'
                os.environ['INFINILM_KIMI_MOE_EXECUTION_R4665'] = '0'

                def emit(token):
                    item['generated_ids'].append(token)
                    item.setdefault('backend_token_monotonic', []).append(time.monotonic())
                    if node == 0: req.emit_token(token)

                def cancelled(step):
                    return control.collect('public-cancel:%s:%d' % (ident, step),
                        dict(cancelled=bool(stop.is_set() or req.cancel_event.is_set()) if node == 0 else None))[0]['cancelled']

                def progress(committed, total, transactions, wall):
                    item['long_prefill_progress'] = dict(input_tokens_committed=committed,
                        prompt_tokens=total, append_input_transactions=transactions, elapsed_s=wall)
                    save()
                    if node == 0: req._events.put(('progress', None))

                native.input_commit_progress = progress
                if not command['cancelled']:
                    actual, ended = generate(native, prompt, cap, {163585, 163586}, emit, cancelled, deadline)
                    assert actual == item['generated_ids']
                else: ended = 'canceled'
                if native.memory_guard: native.memory_guard('public-end:' + ident)
            except BaseException as error:
                failure = error
                item['error'] = repr(error)
                if node == 0: req.fail(error)
                raise
            finally:
                try:
                    native.reset(False)
                    control.collect('public-request-reset:' + ident, dict(node=node))
                    item['normal_reset'] = True
                except BaseException as error:
                    item['reset_error'] = repr(error)
                    if node == 0: req.fail(error)
                    raise
                finally:
                    native.input_commit_progress = None
                    item['finish'] = ended
                    item['finished_monotonic'] = time.monotonic()
                    item['long_prefill_observations'] = getattr(native, 'long_prefill_observations', [])[observation_start:]
                    save()
            if node == 0 and failure is None:
                req.finish(dict(eos=reasons.EOS_TOKEN, length=reasons.LENGTH,
                    canceled=reasons.CANCELED, timeout=reasons.TIMEOUT)[ended])
        report['serving_protocol_complete'] = True
    finally:
        report['agent_native_forward_calls'] = len(native.calls)
        native.arm([], [], [], False)
        save()
