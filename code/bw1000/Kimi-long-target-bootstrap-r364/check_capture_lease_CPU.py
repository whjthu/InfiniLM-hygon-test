"""Real sockets joined to actual bootstrap; Native/weights remain recording providers."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import socket
import sys
import tempfile
import threading
import time
from unittest import mock

from bootstrap import execute, RealDependencies
from capture_port_lease import CapturePortLease
from check_entry_CPU import Dependencies, Settings, FaultGate, environment

HERE = Path(__file__).resolve().parent


def unused_port():
    with socket.socket() as value:
        value.bind(('127.0.0.1', 0))
        return value.getsockname()[1]


def assert_held(lease):
    if lease.node != 0: return
    assert lease.listener.fileno() == lease.bound_fileno and not lease.listener.get_inheritable()
    assert not lease.listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
    with socket.socket() as other:
        try: other.bind((lease.head, lease.port))
        except OSError: pass
        else: raise AssertionError('Reserved port was not exclusively held')


class Rendezvous:
    def __init__(self): self.cv = threading.Condition(); self.rows = {}; self.error = None
    def collect(self, node, label, value):
        with self.cv:
            bucket = self.rows.setdefault(label, {})
            assert node not in bucket
            bucket[node] = value; self.cv.notify_all()
            end = time.monotonic()+5
            while len(bucket) < 4:
                if self.error: raise RuntimeError('Recording leader failed')
                left = end-time.monotonic()
                if left <= 0: raise TimeoutError('Bounded four-leader CPU rendezvous')
                self.cv.wait(min(.1, left))
            return [bucket[n] for n in range(4)]
    def abort(self):
        with self.cv: self.error = True; self.cv.notify_all()


class Control:
    def __init__(self, node, rendezvous): self.node=node;self.rendezvous=rendezvous;self.closed=False
    def collect(self, label, value): return self.rendezvous.collect(self.node, label, value)
    def close(self): self.closed=True


def one_recording_failure(mode):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'control').mkdir();settings=Settings(environment(root),root)
        settings.capture_port=unused_port();gate=FaultGate(settings)
        deps=Dependencies(root, 'load-fail' if mode=='load' else 'ok');report={}
        if mode=='bind':
            blocker=socket.socket();blocker.bind((settings.head,settings.capture_port))
        else: blocker=None
        original=deps.prepare
        def prepare(s,g):
            value=original(s,g);assert_held(deps.capture_lease)
            if mode=='prepare': raise RuntimeError('Recorded prepare failure after actual bind')
            return value
        deps.prepare=prepare
        try:
            try: execute(settings,gate,deps,report,lambda:None)
            except BaseException: pass
            else: raise AssertionError('Expected bounded failure')
            assert not report['completed'] and deps.capture_closed
            assert deps.created == 0 if mode in ['bind','prepare'] else deps.created==deps.destroyed==1
            if mode=='bind': assert deps.opened==deps.loads==0
            if deps.capture_lease:
                assert deps.capture_lease.closed and deps.capture_lease.listener is None
        finally:
            if blocker:blocker.close()
        with socket.socket()as retry:retry.bind((settings.head,settings.capture_port))
        return dict(mode=mode, actual_socket=True, actual_bootstrap=True, created=deps.created,
                    destroyed=deps.destroyed, weight_payload_opened=False, no_rebind_retry=True)


def four_leader_join(bad_nonce=False):
    # The immutable sender/receiver framing is used without modification.
    publisher=HERE.parent/'Kimi-long-target-publisher-r335'
    if publisher.exists():sys.path.insert(0,str(publisher))
    importlib.import_module('batch_wire_r9128')
    port=unused_port();rendezvous=Rendezvous();outputs={}
    with tempfile.TemporaryDirectory()as tmp:
        root=Path(tmp)
        def leader(node):
            where=root/str(node);where.mkdir();(where/'control').mkdir()
            settings=Settings(environment(where,node),where);settings.capture_port=port
            gate=FaultGate(settings)
            class Joined(Dependencies):
                def prepare(self,s,g):
                    value=super().prepare(s,g);assert_held(self.capture_lease)
                    self.control=Control(node,rendezvous)
                    return value
                def load(self,*args):
                    assert_held(self.capture_lease)
                    return super().load(*args)
                def run_pair(self,g,s,gate,save):
                    self.native=SimpleNamespace(node=node,control=self.control,peer=None,peers={})
                    self.capture_lease.establish(self.native,s.nonce+('wrong'if bad_nonce and node==2 else''),2)
                    result=super().run_pair(g,s,gate,save);self.capture_complete=True
                    return result
            deps=Joined(where);report={};error=None
            try:execute(settings,gate,deps,report,lambda:None)
            except BaseException as exc:error=type(exc).__name__;rendezvous.abort()
            outputs[node]=dict(completed=report.get('completed'),error=error,created=deps.created,
                               destroyed=deps.destroyed,closed=deps.capture_lease.closed,
                               actual_bound_FD=deps.capture_lease.bound_fileno)
        threads=[threading.Thread(target=leader,args=(n,))for n in range(4)]
        for t in threads:t.start()
        for t in threads:t.join(12)
        assert not any(t.is_alive()for t in threads)and len(outputs)==4
    assert all(v['created']==v['destroyed']==1 and v['closed']for v in outputs.values())
    if bad_nonce:assert all(v['completed']is False for v in outputs.values())
    else:
        assert all(v['completed']and v['error']is None for v in outputs.values()),outputs
        # Sender2 closes first; normal receiver0 has no fixed-port TIME_WAIT.
        with socket.socket()as released:released.bind(('127.0.0.1',port))
    return dict(actual_bootstrap=True,actual_four_thread_control=True,actual_wire_handshake=True,
                Native_and_weights='recording only',same_FD_from_reserve_to_listen=True,
                no_early_listen=True,outputs=outputs,bad_nonce=bad_nonce)


def actual_prepare_import_boundary():
    """Exercise actual prepare(), not just its recording provider's reserve hook."""
    with tempfile.TemporaryDirectory()as tmp:
        root=Path(tmp);(root/'control').mkdir();s=Settings(environment(root),root);s.capture_port=unused_port()
        deps=RealDependencies(HERE);gate=FaultGate(s)
        manifest=dict(required_environment={},nodes=[socket.gethostname().split('.')[0]]*4)
        bound=dict(manifest=manifest,modules={},model=root)
        original=__import__
        def imports(name,*args,**kwargs):
            if name=='torch':
                assert_held(deps.capture_lease)
                raise RuntimeError('Expected stop before actual GPU import')
            return original(name,*args,**kwargs)
        with mock.patch('bootstrap.verify_artifacts',return_value=bound),mock.patch('builtins.__import__',imports):
            try:deps.prepare(s,gate)
            except RuntimeError as error:assert str(error)=='Expected stop before actual GPU import'
            else:raise AssertionError('Actual prepare failed to stop at GPU boundary')
        deps.close_capture();assert deps.capture_lease.closed
        with socket.socket()as released:released.bind((s.head,s.capture_port))
    return dict(actual_RealDependencies_prepare=True,source_audit_provider='recording',
                actual_lease_precedes_Torch_Native_weights=True,cleanup_with_native_None=True)


def run():
    return dict(completed=True,passed=True,GPU=0,Native_dlopen=False,actual_Context=False,
                actual_prepare=actual_prepare_import_boundary(),
                failure_cases=[one_recording_failure(m)for m in ['bind','prepare','load']],
                four_leader_normal=four_leader_join(),four_leader_bad_nonce=four_leader_join(True))


if __name__=='__main__':print(json.dumps(run()))
