"""Bounded online PP scheduling core. CPU control only; no GPU performance claim.

One coordinator owns this object. Every PP owner consumes the same immutable
job sequence, locally synchronously, with deferred native token feedback.
An emitter completion makes its requests ready again, rather than waiting for
an arbitrary fixed-B round. No physical cache row is reused within an epoch.
This module deliberately has no network server or Engine adapter yet.
"""
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class Request:
    key: str
    prompt: tuple
    arrival: float
    limit: int
    row: int = 0
    past: int = 0
    output: list = field(default_factory=list)
    state: str = 'queued'


@dataclass(frozen=True)
class Job:
    seq: int
    kind: str
    keys: tuple
    rows: tuple
    pasts: tuple
    inputs: tuple


class Scheduler:
    def __init__(self, *, stages, micro, max_users, max_inflight, rows=32, eos=(), max_decode_burst=2):
        assert 1 <= stages <= 4 and 1 <= micro <= max_users <= rows <= 128
        assert 1 <= max_inflight <= rows
        assert type(max_decode_burst) is int and max_decode_burst >= 1
        self.stages, self.micro, self.max_users = stages, micro, max_users
        self.max_inflight, self.rows, self.eos = max_inflight, rows, set(eos)
        self.requests, self.jobs, self.events = {}, [], []
        self.queue, self.ready = deque(), deque()
        self.next_row = 1
        self.cursor = [0] * stages
        self.busy = [None] * stages
        self.acks, self.published = {}, set()
        self.failed = None
        self.decode_burst, self.max_decode_burst = 0, max_decode_burst

    def event(self, kind, now, **kw):
        assert math.isfinite(now) and now >= 0
        self.events.append(dict(kind=kind, t=now, **kw))

    def arrive(self, key, prompt, limit, now):
        assert key not in self.requests and prompt and 0 < limit
        assert all(type(t) is int and 0 <= t < 163840 for t in prompt)
        assert len(prompt) + limit - 1 <= 1024
        self.requests[key] = Request(key, tuple(prompt), now, limit)
        self.queue.append(key)
        self.event('arrival', now, request=key)

    def dispatch(self, now):
        """Nonblocking: call after arrival or emitter completion; no fill wait.

        Prefer ready decode, but bound decode bursts if new admissions fit.
        Separate homogeneous prefill jobs avoid unqualified mixed native calls.
        No padding: each row is a real request with distinct KDA/cache state.
        """
        assert not self.failed
        if len(self.jobs) - len(self.published) >= self.max_inflight:
            return None
        live = sum(r.state in ('ready', 'running') for r in self.requests.values())
        n = min(self.micro, self.max_users-live, self.rows-self.next_row+1, len(self.queue))
        if self.ready and (n <= 0 or self.decode_burst < self.max_decode_burst):
            keys = tuple(self.ready.popleft() for _ in range(min(self.micro, len(self.ready))))
            kind = 'decode'
            self.decode_burst += 1
        else:
            if n <= 0:
                return None
            keys = tuple(self.queue.popleft() for _ in range(n))
            kind = 'prefill'
            self.decode_burst = 0
            for k in keys:
                r = self.requests[k]
                r.row, self.next_row = self.next_row, self.next_row+1
                self.event('admit', now, request=k, row=r.row)
        rs = [self.requests[k] for k in keys]
        assert all(r.state == ('ready' if kind == 'decode' else 'queued') for r in rs)
        j = Job(len(self.jobs), kind, keys, tuple(r.row for r in rs),
                tuple(r.past for r in rs),
                tuple((r.output[-1],) if kind == 'decode' else r.prompt for r in rs))
        self.jobs.append(j)
        self.acks[j.seq] = set()
        for r in rs:
            r.state = 'running'
        self.event('dispatch', now, job=j.seq, requests=keys, micro=len(keys), mode=kind)
        return j

    def take(self, stage, now):
        assert not self.failed and 0 <= stage < self.stages and self.busy[stage] is None
        i = self.cursor[stage]
        if i == len(self.jobs):
            return None
        self.busy[stage] = i
        self.event('stage_start', now, stage=stage, job=i)
        return self.jobs[i]

    def finish(self, stage, seq, now, ids=None):
        assert not self.failed and self.busy[stage] == seq == self.cursor[stage]
        j = self.jobs[seq]
        if stage == self.stages - 1:
            assert ids is not None and len(ids) == len(j.keys)
            assert all(type(v) is int and 0 <= v < 163840 for v in ids)
        else:
            assert ids is None
        self.event('stage_end', now, stage=stage, job=seq)
        self.busy[stage] = None
        self.cursor[stage] += 1
        self.acks[seq].add(stage)
        if stage != self.stages - 1:
            return
        self.published.add(seq)
        for k, past, ins, token in zip(j.keys, j.pasts, j.inputs, ids):
            r = self.requests[k]
            assert r.state == 'running' and r.past == past
            r.past += len(ins)
            r.output.append(token)
            self.event('token', now, request=k, index=len(r.output)-1,
                       token=token, eos=token in self.eos, boundary='emitter')
            done = token in self.eos or len(r.output) == r.limit
            r.state = 'completed' if done else 'ready'
            if done:
                self.event('terminal', now, request=k, status='completed')
            else:
                self.ready.append(k)

    def fail(self, now, reason):
        self.failed = reason
        self.event('failure', now, reason=str(reason))
        # GPU/CCL teardown remains the external controller's responsibility.

    def can_reset_epoch(self):
        return (not self.failed and all(v is None for v in self.busy)
                and all(c == len(self.jobs) for c in self.cursor)
                and not self.ready
                and all(r.state in ('completed', 'queued') for r in self.requests.values()))

    def reset_epoch_after_native_world_drain(self, now, *, owners_reset_verified):
        assert owners_reset_verified and self.can_reset_epoch()
        # Callback caller must actually reset all owners and collect their
        # success before this logical reset. No assertion substitutes for it.
        self.next_row = 1
        self.event('epoch_reset', now, native_world_drain_verified=True)
