"""Private C1 phase RPC. Target commit receipts require a real target publisher.

No listener, checkpoint loader, target launcher or legacy worker is modified.
One connection owns one session; any invalid command poisons that session.
"""
import threading
import time
from batch_wire_r9128 import recv, send


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Integer metadata outside explicit range')
    return value


class TorchFeatureCodec:
    def __init__(self, device, raw_dim=35840, vocab=163840):
        self.device = device
        self.raw_dim = integer(raw_dim, 1, 35840)
        self.vocab = integer(vocab, 1, 2**24)

    def decode(self, body, rows):
        import torch
        part = torch.frombuffer(bytearray(body), dtype=torch.bfloat16).reshape(1, rows, self.raw_dim).to(self.device)
        if not bool(torch.isfinite(part).all()):
            raise ValueError('Nonfinite target feature frame')
        return part

    def fence(self):
        import torch
        if str(self.device).startswith('cuda'): torch.cuda.synchronize(self.device)

    def proposals(self, values, debug, K):
        import torch
        if not bool(torch.isfinite(debug['hidden']).all()):
            raise ValueError('Nonfinite actual long hidden states')
        if tuple(values.shape) != (1, K): raise ValueError('Proposal shape')
        result = values.cpu().tolist()[0]
        for v in result: integer(v, 0, self.vocab-1)
        return result


class DraftSession:
    def __init__(self, context, head, codec, *, enabled=False):
        if enabled is not True: raise RuntimeError('Long RPC default OFF')
        self.owner = threading.get_ident()
        self.context, self.head, self.codec = context, head, codec
        self.sequence = 0
        self.last_capture_epoch = 0
        self.closed = self.poisoned = False

    def _owner(self):
        if threading.get_ident() != self.owner: raise RuntimeError('Foreign RPC owner')

    def close(self):
        self._owner()
        if not self.closed:
            self.context.close()
            self.closed = True

    def dispatch(self, meta, body):
        self._owner()
        if self.closed or self.poisoned: raise RuntimeError('Closed or poisoned session')
        try:
            return self._dispatch(meta, body)
        except BaseException:
            self.poisoned = True
            raise

    def _dispatch(self, meta, body):
        if type(meta) is not dict or type(body) is not bytes: raise ValueError('Frame types')
        if integer(meta.get('seq'), 0, 2**63-1) != self.sequence: raise ValueError('Stale or skipped sequence')
        cmd = meta.get('cmd')
        common = {'cmd','seq','epoch','request'}
        fields = {'begin':{'prompt_tokens'}, 'append_prompt':{'past','rows','capture_epoch'},
                  'append_committed':{'past','rows','capture_epoch','target_commit'},
                  'seal':{'past'}, 'query':{'past','anchor','K'}, 'retire':set()}
        if cmd == 'close':
            if set(meta) != {'cmd','seq'} or body: raise ValueError('Close metadata/body')
            self.close(); self.sequence += 1
            return {'seq':meta['seq'],'closed':True}
        if cmd not in fields or set(meta) != common | fields[cmd]: raise ValueError('Explicit phase fields required')
        epoch = integer(meta['epoch'], 1, 2**63-1)
        request = integer(meta['request'], 1, 1)
        c = self.context
        if cmd != 'begin': c._check(epoch, request)
        started = time.perf_counter()
        response = {'seq':meta['seq'],'epoch':epoch,'request':request,'cmd':cmd}
        if cmd == 'begin':
            if body: raise ValueError('Begin cannot carry features')
            c.begin(epoch, request, integer(meta['prompt_tokens'], 1, 32768))
        elif cmd in ['append_prompt','append_committed']:
            past = integer(meta['past'], 0, 32767)
            rows = integer(meta['rows'], 1, 512 if cmd=='append_prompt' else 8)
            capture = integer(meta['capture_epoch'], 1, 2**63-1)
            if past != c.length or capture <= self.last_capture_epoch: raise ValueError('Stale feature position or capture epoch')
            if c.phase != ('prompt' if cmd=='append_prompt' else 'decode'): raise ValueError('Wrong feature phase')
            if past+rows > (c.prompt_tokens if cmd=='append_prompt' else 32768): raise ValueError('Feature capacity')
            if len(body) != rows*self.codec.raw_dim*2: raise ValueError('Exact BF16 feature byte length required')
            if cmd == 'append_committed':
                proof = meta['target_commit']
                if type(proof) is not dict or set(proof) != {'capture_epoch','past_before','past_after','accepted_inputs','complete'}:
                    raise ValueError('Exact target commit receipt required')
                if proof['complete'] is not True: raise ValueError('Target commit incomplete')
                for k in ['capture_epoch','past_before','past_after','accepted_inputs']:
                    integer(proof[k], 0, 2**63-1)
                if (proof['capture_epoch'],proof['past_before'],proof['past_after'],proof['accepted_inputs']) != (capture,past,past+rows,rows):
                    raise ValueError('Commit prefix/frame mismatch; rejected suffix cannot publish')
            part = self.codec.decode(body, rows)
            getattr(c, cmd)(epoch, request, past, part, self.codec.fence)
            self.last_capture_epoch = capture
            response.update(past_after=c.length, capture_epoch=capture)
        else:
            if body: raise ValueError('Only explicit feature append carries a body')
            if cmd in ['seal','query'] and integer(meta['past'], 1, 32768) != c.length:
                raise ValueError('Stale phase position')
            if cmd == 'seal': c.seal_prompt(epoch, request)
            elif cmd == 'retire': response['removed'] = c.retire(epoch, request)
            elif cmd == 'query':
                anchor = integer(meta['anchor'], 0, self.codec.vocab-1)
                K = integer(meta['K'], 1, 7)
                self.codec.fence()
                result = c.query(epoch, request, self.head, anchor, K, return_debug=True)
                if result['mode'] == 'ar_tail': response.update(mode='ar_tail',proposals=[])
                else:
                    values, debug = result['values']
                    response.update(mode='draft',proposals=self.codec.proposals(values,debug,K))
                self.codec.fence()
        self.sequence += 1
        response.update(context_tokens=c.length, phase=c.phase, handler_wall_s=time.perf_counter()-started)
        return response


def serve_connection(connection, session):
    """Existing bounded binary wire; fail closed and release contexts on EOF/error."""
    try:
        while not session.closed:
            meta, body = recv(connection)
            send(connection, session.dispatch(meta, body))
    finally:
        session.close()
