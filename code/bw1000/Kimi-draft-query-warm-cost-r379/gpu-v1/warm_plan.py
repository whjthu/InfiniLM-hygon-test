"""First-query priming only: original full long-query math, no cache/crop.

Four independent child processes follow OFF/ON/ON/OFF. Priming costs remain
visible; this cannot prove overlap with target prefill or a Model speedup.
"""
import time

LENGTH = 32638
RAW_DIM = 35840
ANCHOR = 163584


def process_plan():
    return [dict(index=i, warm_query=warm) for i, warm in enumerate([False, True, True, False])]


def run_contexts(session, warm_query, body, *, save=lambda value: None,
                 remaining=lambda: None, clock=time.perf_counter):
    assert type(warm_query) is bool
    result = dict(completed=False, warm_query=warm_query, phases=[],
                  target_requests=0, target_loaded=False, speculative_acceptance='UNKNOWN',
                  scope='Synthetic long features, original draft query; no target/online qualification')
    sequence, capture = 0, 0

    def command(cmd, epoch, payload=b'', **fields):
        nonlocal sequence
        remaining()
        meta = dict(cmd=cmd, seq=sequence)
        if cmd != 'close':
            meta.update(epoch=epoch, request=1)
        meta.update(fields)
        response = session.dispatch(meta, payload)
        assert response['seq'] == sequence
        sequence += 1
        return response

    roles = ['priming', 'measured'] if warm_query else ['measured']
    save(result)
    for epoch, role in enumerate(roles, 1):
        phase = dict(role=role, epoch=epoch, context_tokens=LENGTH, completed=False,
                     query_is_first_process_query=epoch == 1,
                     prefix_bytes=LENGTH * session.codec.raw_dim * 2,
                     prefix_append_commands=0)
        result['phases'].append(phase)
        command('begin', epoch, prompt_tokens=LENGTH)
        prefix_start = clock()
        for past in range(0, LENGTH, 512):
            count = min(512, LENGTH-past)
            payload = body(role, count)
            assert type(payload) is bytes and len(payload) == count * session.codec.raw_dim * 2
            capture += 1
            command('append_prompt', epoch, payload, past=past, rows=count, capture_epoch=capture)
            phase['prefix_append_commands'] += 1
        command('seal', epoch, past=LENGTH)
        phase['prefix_and_seal_wall_s'] = clock()-prefix_start
        save(result)
        try:
            started = clock()
            answer = command('query', epoch, past=LENGTH, anchor=ANCHOR, K=7)
            phase['complete_dispatch_wall_s'] = clock()-started
            phase['handler_wall_s'] = answer['handler_wall_s']
            assert answer['mode'] == 'draft' and len(answer['proposals']) == 7
            assert all(type(value) is int and 0 <= value < session.codec.vocab for value in answer['proposals'])
            phase['proposals'] = answer['proposals']
            phase['complete_phase_wall_s'] = clock()-prefix_start
            phase['original_query_mandatory_finite_checks'] = 'Pinned r333 hidden/head/codec checks; no bypass'
        except BaseException as error:
            phase['error'] = repr(error)
            result.update(protocol_commands=sequence, complete_query_error=True)
            save(result)
            raise
        assert command('retire', epoch)['removed']
        assert session.context.raw is None and session.context.projected is None
        phase['completed'] = True
        save(result)
    assert command('close', len(roles))['closed'] and session.closed
    expected = 137 if warm_query else 69
    assert sequence == expected
    result.update(completed=True, normal_session_close=True, protocol_commands=sequence,
                  priming_query_count=int(warm_query), measured_query_count=1,
                  no_query_retries=True, process_cold_not_physical_GPU_cold=True)
    save(result)
    return result
