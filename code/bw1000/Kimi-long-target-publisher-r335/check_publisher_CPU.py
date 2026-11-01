"""Recording target boundary + real greedy policy + actual binary socket client.
No actual Native Context, checkpoint or GPU is executed here.
"""
import json
import socket
import threading
from publisher import Publisher, DraftClient, ROW_BYTES, RUNTIME_SHA, NATIVE_SHA
from batch_wire_r9128 import send, recv


class Control:
    def collect(self, label, value):
        return [dict(node=n, value=value['value'] if n == 0 or not label.endswith('rpc-query') else None)
                for n in range(4)]


class FakeRPC:
    def __init__(self, events):
        self.events = events
        self.frames = []
        self.proposals = [10, 11, 12, 13, 14, 15, 16]
        self.failure = None
    def command(self, cmd, epoch, body=b'', **fields):
        if cmd == self.failure: raise EOFError('recording failure')
        self.events.append(cmd)
        self.frames.append((cmd, fields, body))
        return dict(mode='draft', proposals=self.proposals[:fields.get('K', 0)],
                    context_tokens=fields.get('past', 0))
    def abort(self): self.events.append('abort')


class RecordingNative:
    node = 0
    vocab = 1000
    def __init__(self, events):
        self.events = events
        self.control = Control()
        self.row = [1, 2, 0, 1]
        self.epoch = 0
        self.top = None
        self.failure = None
    def reset(self, state):
        assert state is False
        self.events.append('reset')
        self.row = [1, 2, 0, 1]
    def rows(self, ids):
        return [self.row[:]]
    def seal_prompt(self, past, speculate):
        assert speculate is True and self.row == [1, 2, past, 1]
        self.events.append('native_seal')
    def call(self, kind, ids, past, tokens, capture, on_selected=None):
        width = len(tokens[0]);self.epoch += 1
        self.events.append(kind)
        if kind == 'prefill': self.row[2] += width
        selected = [9] if kind == 'prefill' else self.top or ([10, 99]+list(range(20, 20+width)))[:width]
        if on_selected: on_selected(selected)
        self.events.append('harvest')
        body = b''.join(bytes([i%255+1])*ROW_BYTES for i in range(width))
        return selected, {1:body} if capture else None, dict(past=past, width=[width],
            feature_capture_epoch=self.epoch if capture else None, feature_inputs_committed=kind=='prefill')
    def call_append(self, logical, past, tokens, capture, on_selected=None):
        self.events.append('append_call')
        self.row[2] += len(tokens)
        self.epoch += 1
        if on_selected: on_selected([9])
        self.events.append('harvest')
        return [9], {1:b'P'*(len(tokens)*ROW_BYTES)}, dict(past=[past], width=[len(tokens)],
            feature_capture_epoch=self.epoch, feature_inputs_committed=True)
    def commit(self, ids, decisions):
        self.events.append('commit')
        if self.failure == 'commit': raise RuntimeError('recording commit failure')
        self.decision = d = decisions[0]
        if d.terminal: self.row[3] = 4
        elif d.needs_replay: self.row[3] = 3
        else: self.apply()
    def apply(self):
        self.row = [self.row[1], self.row[0], self.row[2]+len(self.decision.commit_inputs), 1]
    def replay(self, partial, state):
        assert state is True
        self.events.append('replay')
        if self.failure == 'replay': raise RuntimeError('recording replay failure')
        self.apply()


def setup():
    events = [];n = RecordingNative(events);r = FakeRPC(events)
    p = Publisher(n, r, enabled=True)
    return p, n, r, events


def raises(fn):
    try: fn()
    except (ValueError, RuntimeError, EOFError): return
    raise AssertionError('Expected fail-closed boundary')


def run():
    checks = []
    p,n,r,e = setup()
    assert p.prefill([1]*513, 1, lambda x:e.append('emit')) == 9
    assert e.index('append_call') < e.index('emit') < len(e)-1
    j=e.index('emit');assert e[j+1:j+3] == ['harvest','append_prompt']
    assert [f[1]['rows'] for f in r.frames if f[0]=='append_prompt'] == [512,1]
    checks.append('continuous prompt seed/append; final token emitted before harvest/RPC; no reset at seal')
    start=len(e);d=p.step(9,127,(),lambda x:e.append('emit'))
    assert d.accepted == 1 and d.commit_inputs == (9,10) and d.needs_replay
    events=e[start:]
    assert events.index('emit') < events.index('harvest') < events.index('commit') < events.index('replay') < events.index('append_committed')
    cmd,fields,body=r.frames[-1]
    assert fields['target_commit'] == dict(capture_epoch=3,past_before=513,past_after=515,accepted_inputs=2,complete=True)
    assert body == b'\x01'*ROW_BYTES+b'\x02'*ROW_BYTES
    checks.append('real greedy partial rejection; actual call order commit/replay/rows precedes cropped-prefix publication')
    for failure in ('commit','replay'):
        p,n,r,e=setup();p.prefill([1],1,lambda x:None);n.failure=failure
        raises(lambda:p.step(9,127,(),lambda x:None))
        assert not any(f[0]=='append_committed' for f in r.frames) and p.poisoned
        raises(lambda:p.step(9,127,(),lambda x:None))
    checks.append('commit/replay exception never publishes; poisoned publisher rejects another step')
    p,n,r,e=setup();p.prefill([1],1,lambda x:None)
    n.top=[10,11,12,13,14,15,16,17]
    d=p.step(9,127,(),lambda x:None)
    assert d.accepted==7 and not d.needs_replay and 'replay' not in e
    assert r.frames[-1][1]['rows']==8 and n.row==[2,1,9,1]
    checks.append('full acceptance publishes eight committed inputs without replay')
    p,n,r,e=setup();p.prefill([1],1,lambda x:None)
    d=p.step(9,1,(),lambda x:None)
    assert d.terminal and r.frames[-1][0]=='retire' and n.row[3]==4
    assert not any(f[0]=='append_committed' for f in r.frames)
    checks.append('output-cap terminal retires context without publishing terminal capture')
    p,n,r,e=setup();p.prefill([1],1,lambda x:None)
    n.top=[10,11,12,13,14,15,16,17]
    d=p.step(9,127,(11,),lambda x:None)
    assert d.terminal and d.emitted==(10,11) and r.frames[-1][0]=='retire'
    checks.append('EOS inside accepted proposals retires without rejected or post-EOS publication')
    p,n,r,e=setup();p.phase='decode';p.epoch=1;p.past=32761;n.row=[1,2,32761,1];n.top=[99]
    d=p.step(9,7,(),lambda x:None)
    assert d.pending_anchor==99 and not any(f[0] in ('query','append_committed') for f in r.frames)
    assert p.past==32762
    checks.append('32761+ AR tail uses one target input, no draft query or stale-context append')
    p,n,r,e=setup();p.prefill([1],1,lambda x:None);r.failure='query'
    raises(lambda:p.step(9,127,(),lambda x:None));assert 'verify' not in e
    checks.append('query failure stops before target forward')
    p,n,r,e=setup();p.prefill([1],1,lambda x:None)
    old=n.apply
    def badrow(): old();n.row[2]+=1
    n.apply=badrow
    raises(lambda:p.step(9,127,(),lambda x:None))
    assert not any(f[0]=='append_committed' for f in r.frames)
    checks.append('actual row/past mismatch prevents receipt publication')
    p,n,r,e=setup();raises(lambda:p.prefill([1]*32641,1,lambda x:None))
    assert not r.frames
    checks.append('prompt reserves 128 output positions before any reset or RPC')
    p,n,r,e=setup();p.prefill([1],1,lambda x:None);p.retire();raises(lambda:p.prefill([1],1,lambda x:None))
    checks.append('reused request epoch is rejected')
    # Exercise real binary framing and DraftClient identity/sequence, not a mocked send.
    for invalid in (False, True):
        a,b=socket.socketpair();a.settimeout(2);b.settimeout(2);errors=[]
        def server():
            try:
                hello,body=recv(b);assert hello['nonce']=='N'*32 and not body
                send(b,dict(ready=True,protocol='Kimi-C1-long-r334',C=1,context=32768,
                    block_size=8,proposals=7,prompt_append_max=512,committed_append_max=8,
                    runtime_manifest_SHA256=RUNTIME_SHA,target_Native_SHA256=NATIVE_SHA))
                m,body=recv(b);assert m==dict(cmd='close',seq=0) and not body
                send(b,dict(seq=1 if invalid else 0,closed=True))
            except BaseException as error: errors.append(type(error).__name__)
            finally: b.close()
        t=threading.Thread(target=server);t.start();c=DraftClient(a,'N'*32)
        if invalid: raises(lambda:c.command('close'));assert c.poisoned
        else: c.command('close');assert c.closed
        t.join(3);assert not t.is_alive() and not errors
    checks.append('actual binary hello/close and skipped acknowledgement fail-closed')
    raises(lambda:Publisher(RecordingNative([]),FakeRPC([])))
    checks.append('default OFF')
    return dict(passed=True,checks=checks,count=len(checks),actual_Native_Context=False,
                GPU=False,checkpoint_loaded=False,performance_gain='UNKNOWN')


if __name__=='__main__': print(json.dumps(run(),ensure_ascii=False))
