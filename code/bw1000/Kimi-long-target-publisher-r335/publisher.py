"""Default-OFF C1 target publisher. Recording tests do not qualify Native execution."""
import threading
import time
from batch_policy_r9128 import greedy
from batch_wire_r9128 import send, recv

RUNTIME_SHA = '358016b6b9e334817d2d4929ed613d67548dcd6764a20f2372b7748be3c99a2e'
NATIVE_SHA = '4dfdc9a7809b367ef91b476280f12bca6da0c12511eb14dae679147382246505'
ROW_BYTES = 35840 * 2


def integer(x, low, high):
    if type(x) is not int or not low <= x <= high:
        raise ValueError('Integer outside explicit C1 range')
    return x


class DraftClient:
    """An already connected, bounded socket; no reconnect or execution retry."""
    def __init__(self, connection, nonce):
        if type(nonce) is not str or len(nonce) < 32:
            raise ValueError('Explicit session nonce required')
        self.connection = connection
        self.sequence = 0
        self.poisoned = self.closed = False
        try:
            send(connection, dict(cmd='hello', nonce=nonce, protocol='Kimi-C1-long-r334',
                runtime_manifest_SHA256=RUNTIME_SHA, target_Native_SHA256=NATIVE_SHA))
            meta, body = recv(connection)
            expected = dict(ready=True, protocol='Kimi-C1-long-r334', C=1, context=32768,
                block_size=8, proposals=7, prompt_append_max=512, committed_append_max=8,
                runtime_manifest_SHA256=RUNTIME_SHA, target_Native_SHA256=NATIVE_SHA)
            if meta != expected or body:
                raise ValueError('Draft ready identity or geometry mismatch')
        except BaseException:
            self.poisoned = True
            connection.close()
            raise

    def command(self, cmd, epoch=None, body=b'', **fields):
        if self.poisoned or self.closed:
            raise RuntimeError('Closed or poisoned draft client')
        meta = dict(cmd=cmd, seq=self.sequence)
        if cmd != 'close':
            meta.update(epoch=integer(epoch, 1, 2**63-1), request=1, **fields)
        try:
            send(self.connection, meta, body)
            answer, payload = recv(self.connection)
            if payload or type(answer.get('seq')) is not int or answer['seq'] != self.sequence:
                raise ValueError('Draft sequence or binary response mismatch')
            if cmd == 'close':
                if answer != dict(seq=self.sequence, closed=True):
                    raise ValueError('Draft close acknowledgement mismatch')
            else:
                if any(answer.get(k) != v or type(answer.get(k)) is not type(v)
                       for k, v in dict(cmd=cmd, epoch=epoch, request=1).items()):
                    raise ValueError('Draft response identity mismatch')
                integer(answer.get('context_tokens'), 0, 32768)
                if cmd in ('append_prompt', 'append_committed'):
                    want = fields['past'] + fields['rows']
                    if answer.get('past_after') != want or answer['context_tokens'] != want or answer.get('capture_epoch') != fields['capture_epoch']:
                        raise ValueError('Draft feature acknowledgement mismatch')
            self.sequence += 1
            if cmd == 'close':
                self.closed = True
                self.connection.close()
            return answer
        except BaseException:
            self.poisoned = True
            self.connection.close()
            raise

    def abort(self):
        self.poisoned = True
        self.connection.close()


class Publisher:
    """All four target leaders call in the same order; only node0 owns draft RPC."""
    def __init__(self, native, rpc, *, enabled=False):
        if enabled is not True:
            raise RuntimeError('Long publisher default OFF')
        self.native, self.rpc = native, rpc
        integer(native.node, 0, 3)
        if (native.node == 0) != (rpc is not None):
            raise ValueError('Only node0 owns the draft socket')
        self.owner = threading.get_ident()
        self.serial = 0
        self.epoch = 0
        self.phase = 'new'
        self.poisoned = False
        self.costs = []

    def _owner(self):
        if threading.get_ident() != self.owner or self.poisoned:
            raise RuntimeError('Foreign or poisoned publisher')

    def _gather(self, kind, value):
        packets = self.native.control.collect('long-publish:%d:%s' % (self.serial, kind),
                                               dict(node=self.native.node, value=value))
        self.serial += 1
        if len(packets) != 4 or {p['node'] for p in packets} != set(range(4)):
            raise ValueError('Exact four target leader records required')
        return {p['node']:p['value'] for p in packets}

    def _rows(self, expected=None, terminal=False):
        records = self._gather('rows', self.native.rows([1]))
        rows = records[0]
        if any(x != rows for x in records.values()) or len(rows) != 1 or len(rows[0]) != 4:
            raise ValueError('Target leader row disagreement')
        if terminal:
            if rows[0][3] != 4: raise ValueError('Target did not retire terminal request')
        elif expected is not None and rows != [expected]:
            raise ValueError('Actual target row/past/phase mismatch')
        return rows[0]

    def _rpc(self, cmd, body=b'', **fields):
        tick = time.perf_counter()
        local = None
        if self.native.node == 0:
            try:
                answer = self.rpc.command(cmd, self.epoch, body, **fields)
                if cmd == 'query':
                    values = answer.get('proposals')
                    if answer.get('mode') != 'draft' or type(values) is not list or len(values) != fields['K']:
                        raise ValueError('Draft query shape/mode mismatch')
                    for token in values: integer(token, 0, self.native.vocab-1)
                    if answer.get('context_tokens') != fields['past']:
                        raise ValueError('Stale draft context')
                local = dict(ok=True, answer=answer)
            except BaseException as error:
                # No nonce, tensor data or potentially sensitive remote exception text.
                local = dict(ok=False, error_type=type(error).__name__)
        result = self._gather('rpc-'+cmd, local)[0]
        if not result or result['ok'] is not True:
            raise RuntimeError('Draft command failed; no retry')
        self.costs.append(dict(operation=cmd, local_wall_s=time.perf_counter()-tick))
        return result['answer']

    def _features(self, features, record, past, width, committed):
        error = None
        try:
            capture = integer(record.get('feature_capture_epoch'), 1, 2**63-1)
            if record.get('feature_inputs_committed') is not committed:
                raise ValueError('Wrong capture commitment flag')
            if record.get('past') != [past] or record.get('width') != [width]:
                raise ValueError('Capture geometry mismatch')
            if self.native.node == 0:
                if type(features) is not dict or set(features) != {1} or type(features[1]) is not bytes or len(features[1]) != width*ROW_BYTES:
                    raise ValueError('Exact combined five-tap BF16 capture required')
            elif features is not None:
                raise ValueError('Non-root cannot publish combined features')
        except (ValueError, TypeError):
            capture, error = None, True
        records = self._gather('capture', dict(epoch=capture, error=error))
        if any(r['error'] or r['epoch'] != records[0]['epoch'] for r in records.values()):
            raise ValueError('Capture metadata disagreement')
        return capture

    def _failed(self):
        self.poisoned = True
        if self.rpc is not None: self.rpc.abort()

    def prefill(self, tokens, epoch, emit):
        self._owner()
        try:
            if self.phase not in ('new', 'retired'):
                raise ValueError('Previous request must retire')
            epoch = integer(epoch, self.epoch+1, 2**63-1)
            if not isinstance(tokens, (list, tuple)) or not 1 <= len(tokens) <= 32768-128:
                raise ValueError('Frozen prompt must leave 128 output positions')
            for token in tokens: integer(token, 0, self.native.vocab-1)
            self.epoch = epoch
            self.native.reset(False)
            self._rpc('begin', prompt_tokens=len(tokens))
            for past in range(0, len(tokens), 512):
                chunk = list(tokens[past:past+512])
                callback = (lambda selected: emit(tuple(selected))) if past+len(chunk) == len(tokens) else None
                if past == 0:
                    selected, features, record = self.native.call('prefill', [1], [0], [chunk], True, callback)
                else:
                    selected, features, record = self.native.call_append(1, past, chunk, capture=True, on_selected=callback)
                self._rows([1, 2, past+len(chunk), 1])
                capture = self._features(features, record, past, len(chunk), True)
                self._rpc('append_prompt', features[1] if self.native.node == 0 else b'',
                          past=past, rows=len(chunk), capture_epoch=capture)
            self.native.seal_prompt(len(tokens), speculate=True)
            self._rows([1, 2, len(tokens), 1])
            self._rpc('seal', past=len(tokens))
            self.past, self.phase = len(tokens), 'decode'
            return selected[0]
        except BaseException:
            self._failed()
            raise

    def step(self, anchor, remaining, eos, emit):
        self._owner()
        try:
            if self.phase != 'decode': raise ValueError('Sealed decode required')
            integer(anchor, 0, self.native.vocab-1)
            integer(remaining, 1, 128)
            before = self._rows()
            if before[2:] != [self.past, 1]: raise ValueError('Stale target position')
            K = min(7, remaining-1, 32768-self.past-1)
            if self.past > 32760: K = 0
            if K < 0: raise ValueError('No target context capacity')
            proposals = self._rpc('query', past=self.past, anchor=anchor, K=K)['proposals'] if K else []
            capture_on = self.past <= 32760
            decisions = []
            def chosen(selected):
                decision = greedy(anchor, proposals, selected, eos, remaining)
                decisions.append(decision)
                emit(decision.emitted)  # before harvest/transfer/commit, as qualified emitter scope
            _, features, record = self.native.call('verify', [1], [self.past], [[anchor]+proposals], capture_on, chosen)
            if len(decisions) != 1: raise ValueError('Selected-ID callback must run exactly once')
            decision = decisions[0]
            width = 1+len(proposals)
            capture = self._features(features, record, self.past, width, False) if capture_on else None
            tick = time.perf_counter()
            self.native.commit([1], [decision])
            if decision.needs_replay:
                self.native.replay([(1, self.past, len(decision.commit_inputs))], state=True)
            after = self.past+len(decision.commit_inputs)
            if decision.terminal:
                self._rows(terminal=True)
                self._rpc('retire')
                self.phase = 'retired'
            else:
                self._rows([before[1], before[0], after, 1])
                if capture_on:
                    accepted = len(decision.commit_inputs)
                    raw = features[1][:accepted*ROW_BYTES] if self.native.node == 0 else b''
                    receipt = dict(capture_epoch=capture, past_before=self.past, past_after=after,
                                   accepted_inputs=accepted, complete=True)
                    self._rpc('append_committed', raw, past=self.past, rows=accepted,
                              capture_epoch=capture, target_commit=receipt)
            self.costs.append(dict(operation='commit_replay_publish', local_wall_s=time.perf_counter()-tick,
                accepted_proposals=decision.accepted, emitted=len(decision.emitted),
                replay=decision.needs_replay, terminal=decision.terminal, ar_tail=not capture_on))
            self.past = after
            return decision
        except BaseException:
            self._failed()
            raise

    def retire(self):
        """Cancellation/first-token EOS: caller owns Native reset/destruction, not replay."""
        self._owner()
        try:
            if self.phase != 'retired': self._rpc('retire')
            self.phase = 'retired'
        except BaseException:
            self._failed()
            raise

    def close(self):
        self._owner()
        try:
            self._rpc('close')
            self.phase = 'closed'
        except BaseException:
            self._failed()
            raise
