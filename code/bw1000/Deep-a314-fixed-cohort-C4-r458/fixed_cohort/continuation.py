"""Single owner, four fixed sequences. No row reuse or online admission."""
import hashlib
import json
import math
import time
import numpy as np
from .profile import PROFILE, check_modes


def run(context, records, *, mode, deadline, save, canceled=lambda: False,
        clock=time.perf_counter, wall=time.time):
    assert mode in ('serial', 'cohort') and len(records) == 4
    assert context.config['profile'] == PROFILE and context.config['rows'] == 4
    assert [r['row'] for r in records] == list(range(4))
    context.check_thread()
    check_modes(context.binding.lib)
    signatures = [(r['case_id'], tuple(r['prompt_token_ids']), r['max_new_tokens']) for r in records]
    assert all(type(b) is int and b == 4096 and ids and len(ids) + b <= 32768 for _, ids, b in signatures)
    states = [dict(row=i, case_id=r['case_id'], past=0, generated_ids=[],
                   token_events=[], finish=None, sampled=None) for i, r in enumerate(records)]
    report = dict(profile=PROFILE, mode=mode, rows=4, capacity_per_row=32768,
                  completed=False, failed=False, calls=[], states=states,
                  Native_reset_before=False, Native_reset_after=False,
                  all_arrival_monotonic=clock(), Native_SHA256=context.config['Native_SHA256'])

    def check_deadline():
        if wall() >= deadline:
            raise TimeoutError('Original fixed-cohort deadline expired')

    def checkpoint(phase):
        assert [(r['case_id'], tuple(r['prompt_token_ids']), r['max_new_tokens']) for r in records] == signatures
        save(phase, report)

    def call(rows, seqs, kind):
        check_deadline(); context.check_thread()
        assert rows and len(rows) == len(set(rows)) and all(states[i]['finish'] is None for i in rows)
        sizes = [len(v) for v in seqs]
        assert all(n > 0 for n in sizes) and sum(sizes) <= 4096
        past = [states[i]['past'] for i in rows]
        assert all(p + n <= 32768 for p, n in zip(past, sizes))
        offsets = [0]
        for n in sizes:
            offsets.append(offsets[-1] + n)
        ids = [t for v in seqs for t in v]
        positions = [p + j for p, n in zip(past, sizes) for j in range(n)]
        start = clock()
        code, logits, sampled = context.forward(ids, positions, rows, past,
            [p + n for p, n in zip(past, sizes)], offsets)
        returned = clock()
        rec = dict(kind=kind, rows=list(rows), past=past, widths=sizes,
                   offsets=offsets, start_monotonic=start, return_monotonic=returned,
                   seconds=returned-start, rc=int(code), output_rows=len(rows))
        report['calls'].append(rec)
        # Persist actual returned status before validation; no retry.
        checkpoint('call_returned')
        context.binding.check(code)
        assert logits.dtype == np.uint16 and logits.shape == (len(rows), context.cfg['vocab_size'])
        bits = np.asarray(logits, dtype=np.uint16)
        assert not np.any((bits & 0x7f80) == 0x7f80), 'Nonfinite BF16 logits'
        assert sampled.shape == (len(rows),) and sampled.dtype == np.int64
        assert all(0 <= int(t) < context.cfg['vocab_size'] for t in sampled)
        rec['finite'] = True
        for index, row in enumerate(rows):
            states[row]['past'] += sizes[index]
            states[row]['sampled'] = int(sampled[index])
        check_deadline()
        return returned

    def emit(row, timestamp):
        s = states[row]; token = s['sampled']
        assert token is not None
        s['generated_ids'].append(token)
        s['token_events'].append(dict(token=token, monotonic=timestamp, visible=token != 1))
        if token == 1:
            s['finish'] = 'EOS'
        elif len(s['generated_ids']) == records[row]['max_new_tokens']:
            s['finish'] = 'cap'
        checkpoint('token_emitted')

    def prefill(row):
        ids = signatures[row][1]
        for offset in range(0, len(ids), 1024):
            if canceled():
                return False
            timestamp = call([row], [list(ids[offset:offset+1024])], 'prefill')
            if offset + 1024 >= len(ids):
                # First output is recorded immediately at the actual final
                # prefill return, before any other row's prefill is started.
                emit(row, timestamp)
        return True

    def decode(rows):
        timestamp = call(rows, [[states[i]['sampled']] for i in rows], 'decode')
        for row in rows:
            emit(row, timestamp)

    failure = None
    try:
        check_deadline(); context.reset(); report['Native_reset_before'] = True
        report['memory_before'] = context.floor(); checkpoint('cohort_start')
        if mode == 'serial':
            for row in range(4):
                if not prefill(row):
                    break
                while states[row]['finish'] is None and not canceled():
                    decode([row])
                if canceled():
                    break
        else:
            for row in range(4):
                if not prefill(row):
                    break
            while all(states[i]['past'] > 0 for i in range(4)) and not canceled():
                active = [i for i in range(4) if states[i]['finish'] is None]
                if not active:
                    break
                decode(active)
        if canceled():
            for s in states:
                if s['finish'] is None:
                    s['finish'] = 'canceled'
        report['completed'] = all(s['finish'] in ('EOS', 'cap', 'canceled') for s in states)
        assert report['completed'], 'Incomplete fixed cohort'
        check_modes(context.binding.lib)
    except BaseException as e:
        failure = e
        report.update(failed=True, error=repr(e))
        checkpoint('failed_before_drain')
    finally:
        # Original reset is synchronous on all8 workers, after the returned
        # current call. It resets all rows once; never resets an active row alone.
        try:
            context.check_thread(); context.reset()
            report['Native_reset_after'] = True
            report['memory_after'] = context.floor()
        except BaseException as e:
            report.update(failed=True, cleanup_error=repr(e))
            if failure is None:
                failure = e
        report['completion_monotonic'] = clock()
        report['whole_cohort_seconds'] = report['completion_monotonic'] - report['all_arrival_monotonic']
        report['normal'] = failure is None and report['completed'] and report['Native_reset_after']
        checkpoint('terminal')
    if failure is not None:
        raise failure
    return report
