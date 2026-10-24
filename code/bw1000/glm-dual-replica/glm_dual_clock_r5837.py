"""Two existing GLM feedback hubs, one bounded CPU receipt clock. No Torch.

This observes protocol-accepted packet receipt, not tail readiness or client IO.
The native owner and original per-replica publish/take order are unchanged.
"""
import copy
import json
import math
import pathlib
import secrets
import threading
import time

from glm_continuous_feedback_r5466 import FeedbackHub

PLAN = (('isolated-A-1', (0,)), ('isolated-B-1', (1,)),
        ('concurrent-1', (0, 1)), ('concurrent-2', (0, 1)),
        ('isolated-B-2', (1,)), ('isolated-A-2', (0,)))


class ReceiptLedger:
    """At most 2*16*128 decisions in a phase, including terminal EOS."""
    def __init__(self, active, start, eos, limit=128):
        assert tuple(active) in ((0,), (1,), (0, 1))
        assert math.isfinite(start) and limit == 128
        self.active, self.start, self.eos, self.limit = tuple(active), start, set(eos), limit
        assert self.eos and all(type(v) is int and 0 <= v < 154880 for v in self.eos)
        self.rows = {r: [[] for _ in range(16)] for r in self.active}
        self.last_stamp = start

    def record(self, replica, cohort, step, rows, ids, stamp, first=False):
        assert replica in self.rows and type(cohort) is int and cohort in (0, 1)
        assert type(step) is int and 0 <= step < 128
        assert type(first) is bool and first == (step == 0)
        assert len(rows) == len(ids) and rows and len(set(rows)) == len(rows)
        assert all(type(v) is int and 0 <= v < 16 for v in rows)
        assert all(type(v) is int and 0 <= v < 154880 for v in ids)
        assert math.isfinite(stamp) and stamp >= self.last_stamp
        group = list(range(8*cohort, 8*cohort+8))
        histories = self.rows[replica]
        expected = group if first else [i for i in group if histories[i]
                    and histories[i][-1]['id'] not in self.eos and len(histories[i]) < self.limit]
        assert rows == expected, ('missing/duplicate/reordered/after-terminal rows', replica, rows, expected)
        assert all(len(histories[i]) == step for i in rows), 'skipped/duplicate decision index'
        # Whole packet validated before any row is changed.
        for i, token in zip(rows, ids):
            histories[i].append({'id': token, 'receipt_seconds': stamp-self.start})
        self.last_stamp = stamp

    def finish(self):
        all_rows = [(replica, i, h) for replica, rows in self.rows.items() for i, h in enumerate(rows)]
        incomplete = [{'replica': r, 'row': i, 'decisions': len(h),
                       'missing_first': not h, 'censored': True}
                      for r, i, h in all_rows
                      if not h or (h[-1]['id'] not in self.eos and len(h) < self.limit)]
        if incomplete:
            return {'completed': False, 'censored_rows': incomplete, 'global_decode_tokens_s': None,
                    'effective_requests': len(all_rows), 'padding_tokens': 0,
                    'receipt_rows': self.rows}
        anchor = max(h[0]['receipt_seconds'] for _, _, h in all_rows)
        end = max(h[-1]['receipt_seconds'] for _, _, h in all_rows)
        useful = sum(v['id'] not in self.eos for _, _, h in all_rows for v in h)
        count = sum(v['id'] not in self.eos and anchor < v['receipt_seconds'] <= end
                    for _, _, h in all_rows for v in h)
        assert end >= anchor
        duration = end-anchor
        user = []
        for r, i, h in all_rows:
            span = h[-1]['receipt_seconds']-h[0]['receipt_seconds']
            user.append({'replica': r, 'row': i, 'TTFT_collector_seconds': h[0]['receipt_seconds'],
                         'decode_useful_tokens_s': sum(v['id'] not in self.eos for v in h[1:])/span if span else None,
                         'terminal_EOS': h[-1]['id'] in self.eos, 'decisions': len(h),
                         'censored': False})
        rate = count/duration if duration else 0.0
        return {'completed': True, 'censored_rows': [], 'effective_requests': len(all_rows),
                'padding_tokens': 0, 'global_common_anchor_seconds': anchor,
                'global_last_decision_seconds': end, 'decode_window_seconds': duration,
                'decode_useful_tokens_strictly_after_anchor': count,
                'global_decode_tokens_s': rate, 'all_provisioned_service_cards': 32,
                'decode_tokens_s_per_provisioned_card': rate/32,
                'active_service_cards': len(self.active)*16,
                'decode_tokens_s_per_active_card': rate/(len(self.active)*16),
                'E2E_useful_tokens_s': useful/end if end else 0.0,
                'useful_tokens_including_prefill_decision': useful,
                'per_request_collector_metrics': user,
                'observer': 'single CPU coordinator accepted packet receipt; phase release origin; not tail/client',
                'receipt_rows': self.rows}


class DualClock:
    def __init__(self, output, eos, clock=time.perf_counter, timeout=3600):
        self.output, self.eos, self.clock, self.timeout = pathlib.Path(output), set(eos), clock, timeout
        assert not self.output.exists()
        self.cv = threading.Condition()
        self.index, self.ready, self.done, self.closed = 0, set(), {}, set()
        self.phase = None
        self.results, self.failure = [], None
        self.save()

    def save(self):
        q = {'plan': [{'label': p, 'active': list(a)} for p, a in PLAN],
             'phase_index': self.index, 'ready_replicas': sorted(self.ready),
             'done_replicas': sorted(self.done), 'closed_replicas': sorted(self.closed),
             'completed_phases': self.results, 'failure': self.failure,
             'completed': len(self.results) == len(PLAN) and not self.failure,
             'normal_quality_rejection': bool(self.results and not self.results[-1]['qualified']),
             'in_progress_diagnostic_only': self.phase.finish() if self.phase is not None else None,
             'clock': 'one process time.perf_counter accepted-receipt clock',
             'no_GPU_or_Engine_calls_in_coordinator': True}
        tmp = self.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(q, allow_nan=False, indent=2)); tmp.replace(self.output)

    def wait(self, predicate):
        deadline = time.monotonic()+self.timeout
        while not predicate():
            if self.failure: raise RuntimeError(self.failure)
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError('dual phase wait')
            self.cv.wait(min(remaining, 5))

    def fail(self, message):
        with self.cv:
            if self.failure is None: self.failure = str(message)[:2000]
            self.save(); self.cv.notify_all()

    def begin(self, replica, label):
        with self.cv:
            assert not self.failure and replica in (0, 1) and self.index < len(PLAN)
            expected, active = PLAN[self.index]
            assert label == expected and replica not in self.ready
            self.ready.add(replica)
            if len(self.ready) == 2:
                self.phase = ReceiptLedger(active, self.clock(), self.eos)
                self.cv.notify_all()
            self.save(); self.wait(lambda: self.phase is not None)
            return {'phase': expected, 'active': replica in active,
                    'phase_index': self.index, 'clock_origin': 'common coordinator release'}

    def record(self, replica, case, cohort, step, rows, ids, first=False):
        with self.cv:
            assert not self.failure and self.phase is not None and case == PLAN[self.index][0]
            self.phase.record(replica, cohort, step, rows, ids, self.clock(), first)

    def finish(self, replica, label, quality, generated, stopped):
        with self.cv:
            assert not self.failure and self.phase is not None and label == PLAN[self.index][0]
            assert replica in self.ready and replica not in self.done and type(quality) is bool
            phase_index = self.index
            active = replica in self.phase.active
            if active:
                histories = self.phase.rows[replica]
                if generated == stopped == [] and not quality:
                    assert not any(histories), 'pre-forward rejection cannot hide emitted decisions'
                else:
                    assert len(generated) == len(stopped) == 16
                    assert all(type(v) is bool for v in stopped)
                    assert generated == [[v['id'] for v in h] for h in histories], 'receipt/owner final IDs mismatch'
                    assert stopped == [bool(h and h[-1]['id'] in self.eos) for h in histories]
            else:
                assert generated == stopped == [], 'inactive replica emitted decisions'
            self.done[replica] = quality
            if len(self.done) == 2:
                result = self.phase.finish()
                result.update(label=label, phase_index=phase_index,
                              quality_passed=all(self.done.values()))
                result['qualified'] = result['completed'] and result['quality_passed']
                self.results.append(result)
                self.index += 1; self.ready, self.done, self.phase = set(), {}, None
                if not result['qualified']:
                    # Controlled quality rejection, both replicas still restore normally.
                    self.index = len(PLAN)
                self.save(); self.cv.notify_all()
            else:
                self.save()
                self.wait(lambda: len(self.results) > phase_index)
            result = self.results[phase_index]
            return {k: result[k] for k in ('label', 'qualified', 'completed', 'quality_passed')}

    def close_replica(self, replica):
        with self.cv:
            assert replica in (0, 1) and replica not in self.closed and self.phase is None
            assert self.index == len(PLAN) or self.failure
            self.closed.add(replica); self.save(); self.cv.notify_all()
            return {'closed': True}


class ReceiptHub(FeedbackHub):
    def __init__(self, replica, coordinator, *args, **kwargs):
        self.replica, self.coordinator = replica, coordinator
        super().__init__(*args, **kwargs)

    def fail_case(self, message):
        super().fail_case(message)
        if self.replica not in self.coordinator.closed: self.coordinator.fail(message)

    def dispatch(self, rank, req):
        op = req.get('op')
        if op in ('dual_begin', 'dual_done', 'dual_close', 'observe_first'):
            assert rank == self.emitter
            if op == 'dual_begin':
                assert self.case is None and not self.pending
                return self.coordinator.begin(self.replica, req['case'])
            if op == 'dual_done':
                assert self.case is None and not self.pending
                return self.coordinator.finish(self.replica, req['case'], req['quality'],
                                               req['generated'], req['stopped'])
            if op == 'dual_close': return self.coordinator.close_replica(self.replica)
            assert req['case'] == self.case
            return self.coordinator.record(self.replica, req['case'], req['cohort'], 0,
                                           req['rows'], req['ids'], True)
        value = super().dispatch(rank, req)
        if op == 'publish':
            packet = req['packet']
            self.coordinator.record(self.replica, req['case'], req['cohort'], req['step']+1,
                                    packet['rows'], packet['ids'])
        return value


def serve(root, run, host, eos):
    root = pathlib.Path(root)
    endpoint = root/'control'/('glm-dual-r5837-'+run+'-endpoints.json')
    assert not endpoint.exists()
    coordinator = DualClock(root/'results'/('glm-dual-clock-r5837-'+run+'.json'), eos)
    hubs = []
    try:
        for replica in (0, 1):
            hubs.append(ReceiptHub(replica, coordinator, '0.0.0.0', 0, secrets.token_hex(24),
                        participants=16, emitter=8, timeout=180, idle_timeout=7200))
        record = {'run': run, 'host': host, 'replicas': [
            {'host': host, 'port': h.address[1], 'nonce': h.nonce, 'participants': 16,
             'emitter': 8, 'replica': i} for i, h in enumerate(hubs)],
            'one_clock_process': True, 'pid': __import__('os').getpid()}
        tmp = endpoint.with_suffix('.tmp'); tmp.write_text(json.dumps(record)); tmp.replace(endpoint)
        with coordinator.cv:
            coordinator.wait(lambda: len(coordinator.closed) == 2)
        assert not coordinator.failure
    except BaseException as error:
        coordinator.fail(type(error).__name__+': '+str(error))
        raise
    finally:
        for hub in hubs: hub.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--eos-json', required=True)
    args = parser.parse_args()
    serve(args.root, args.run, args.host, json.loads(pathlib.Path(args.eos_json).read_text())['provenance']['eos_token_ids'])
