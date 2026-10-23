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
    key: int
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
    def __init__(self, *, stages, micro, max_users, max_inflight, rows=32, eos=(), max_decode_burst=2, max_queue=32, max_events=4096):
        assert 1 <= stages <= 4 and 1 <= micro <= max_users <= rows <= 128
        assert 1 <= max_inflight <= rows
        assert type(max_decode_burst) is int and max_decode_burst >= 1
        assert type(max_queue) is int and 1 <= max_queue <= 1024
        assert type(max_events) is int and max_events >= 2*micro+3
        self.stages, self.micro, self.max_users = stages, micro, max_users
        self.max_inflight, self.rows, self.eos = max_inflight, rows, set(eos)
        self.requests, self.jobs, self.events = {}, {}, []
        self.queue, self.ready = deque(), deque()
        self.next_row = 1
        self.cursor = [0] * stages
        self.busy = [None] * stages
        self.acks, self.published = {}, set()
        self.failed = None
        self.decode_burst, self.max_decode_burst = 0, max_decode_burst
        self.max_queue, self.max_events = max_queue, max_events
        self.last_request_key, self.next_job, self.next_event = -1, 0, 0
        self.last_event_time = -1.0
        self.rejected = 0

    def reserve_events(self, count, now):
        """Backpressure BEFORE mutations; broker must persist/ack buffered events.

        Caller retries the same operation after draining; never auto-drop trace.
        All times must be the coordinator's single monotonic clock.
        """
        assert math.isfinite(now) and now >= 0 and now >= self.last_event_time
        if len(self.events)+count > self.max_events:
            raise BufferError('event journal full; persist and drain before retry')

    def drain_events(self, sink):
        """Sink must durably accept immutable event IDs idempotently, or raise.

        No background writer or unbounded in-memory archive is hidden here.
        Caller selects disk/stream retention and its backpressure policy.
        """
        batch = tuple(dict(e) for e in self.events)
        sink(batch)
        del self.events[:len(batch)]
        return len(batch)

    def event(self, kind, now, **kw):
        self.reserve_events(1, now)
        self.events.append(dict(event_id=self.next_event, kind=kind, t=now, **kw))
        self.next_event += 1
        self.last_event_time = now

    def arrive(self, key, prompt, limit, now):
        """Broker assigns strictly increasing nonnegative integer keys.

        Constant-space high-water mark rejects stale/replayed identities even
        after historical requests are retired. False explicitly means rejected.
        """
        assert not self.failed and type(key) is int and self.last_request_key < key < 2**63
        assert prompt and type(limit) is int and 0 < limit
        assert all(type(t) is int and 0 <= t < 163840 for t in prompt)
        assert len(prompt) + limit - 1 <= 1024
        self.reserve_events(2, now)
        self.last_request_key = key
        self.event('arrival', now, request=key)
        if len(self.queue) >= self.max_queue:
            self.rejected += 1
            self.event('terminal', now, request=key, status='rejected', reason='queue_full')
            return False
        self.requests[key] = Request(key, tuple(prompt), now, limit)
        self.queue.append(key)
        return True

    def dispatch(self, now):
        """Nonblocking: call after arrival or emitter completion; no fill wait.

        Prefer ready decode, but bound decode bursts if new admissions fit.
        Separate homogeneous prefill jobs avoid unqualified mixed native calls.
        No padding: each row is a real request with distinct KDA/cache state.
        """
        assert not self.failed
        if len(self.jobs) >= self.max_inflight:
            return None
        live = sum(r.state in ('ready', 'running') for r in self.requests.values())
        n = min(self.micro, self.max_users-live, self.rows-self.next_row+1, len(self.queue))
        if self.ready and (n <= 0 or self.decode_burst < self.max_decode_burst):
            self.reserve_events(1, now)
            keys = tuple(self.ready.popleft() for _ in range(min(self.micro, len(self.ready))))
            kind = 'decode'
            self.decode_burst += 1
        else:
            if n <= 0:
                return None
            # Bridge's actual packed token cap is8192, independent of M.
            n_tokens = 0
            packed_n = 0
            for key in list(self.queue)[:n]:
                width = len(self.requests[key].prompt)
                if n_tokens+width > 8192:
                    break
                n_tokens += width
                packed_n += 1
            n = packed_n
            assert n > 0 # individual prompt<=1024 was checked on arrival.
            self.reserve_events(n+1, now)
            keys = tuple(self.queue.popleft() for _ in range(n))
            kind = 'prefill'
            self.decode_burst = 0
            for k in keys:
                r = self.requests[k]
                r.row, self.next_row = self.next_row, self.next_row+1
                self.event('admit', now, request=k, row=r.row)
        rs = [self.requests[k] for k in keys]
        assert all(r.state == ('ready' if kind == 'decode' else 'queued') for r in rs)
        j = Job(self.next_job, kind, keys, tuple(r.row for r in rs),
                tuple(r.past for r in rs),
                tuple((r.output[-1],) if kind == 'decode' else r.prompt for r in rs))
        assert 0 < sum(map(len,j.inputs)) <= 8192
        self.jobs[j.seq] = j
        self.next_job += 1
        self.acks[j.seq] = set()
        for r in rs:
            r.state = 'running'
        self.event('dispatch', now, job=j.seq, requests=keys, micro=len(keys), mode=kind)
        return j

    def take(self, stage, now):
        assert not self.failed and 0 <= stage < self.stages and self.busy[stage] is None
        i = self.cursor[stage]
        if i == self.next_job:
            return None
        self.reserve_events(1, now)
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
        self.reserve_events(1+(2*len(j.keys) if stage==self.stages-1 else 0),now)
        self.event('stage_end', now, stage=stage, job=seq)
        self.busy[stage] = None
        self.cursor[stage] += 1
        self.acks[seq].add(stage)
        if stage == self.stages-1:
            self.publish(j, ids, now)
        self.retire_acknowledged()

    def publish(self, j, ids, now):
        self.published.add(j.seq)
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

    def retire_acknowledged(self):
        """Release job payload only after ALL owner completion acknowledgments.

        Completed request payload waits until no retained job references it;
        physical row still remains unavailable until verified world epoch reset.
        """
        retired = [seq for seq in self.jobs if len(self.acks[seq])==self.stages and seq in self.published]
        for seq in retired:
            del self.jobs[seq]
            del self.acks[seq]
            self.published.remove(seq)
        referenced = {key for j in self.jobs.values() for key in j.keys}
        for key in list(self.requests):
            if self.requests[key].state=='completed' and key not in referenced:
                del self.requests[key]

    def fail(self, now, reason):
        self.reserve_events(1, now)
        self.failed = str(reason)[:1024] or 'unspecified failure'
        self.event('failure', now, reason=self.failed)
        # GPU/CCL teardown remains the external controller's responsibility.

    def can_reset_epoch(self):
        return (not self.failed and all(v is None for v in self.busy)
                and all(c == self.next_job for c in self.cursor)
                and not self.jobs and not self.acks and not self.published
                and not self.ready
                and all(r.state in ('completed', 'queued') for r in self.requests.values()))

    def reset_epoch_after_native_world_drain(self, now, *, owners_reset_verified):
        assert owners_reset_verified and self.can_reset_epoch()
        self.reserve_events(1, now)
        # Callback caller must actually reset all owners and collect their
        # success before this logical reset. No assertion substitutes for it.
        self.next_row = 1
        self.event('epoch_reset', now, native_world_drain_verified=True)
