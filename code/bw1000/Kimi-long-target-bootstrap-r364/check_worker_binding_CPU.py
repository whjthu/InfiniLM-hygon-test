"""Actual new worker source and335 client over real TCP; loader stays recorded."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,socket,sys,tempfile,threading


def run(bound):
    pins=json.loads((Path(__file__).resolve().parent/'bindings.json').read_bytes())
    wp=Path(pins['packages']['Kimi-long-draft-memory-floor-r362b']['path'])
    assert hashlib.sha256((wp/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()=='61ac8b04af464d76b4ce46f0793e6d8a2a8e57f41224d6228c5b826bda6ffd31'
    spec=importlib.util.spec_from_file_location('CPU_actual_worker362b',wp/'worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    sys.path.insert(0,str(wp.parent/'Kimi-long-draft-RPC-CPU-r333'))
    import long_rpc
    wire=bound['modules']['batch_wire_r9128'];publisher=bound['modules']['publisher']
    assert hashlib.sha256(Path(wire.__file__).read_bytes()).hexdigest()==hashlib.sha256((wp.parent/'Kimi-long-draft-RPC-CPU-r333/batch_wire_r9128.py').read_bytes()).hexdigest()
    results=[]
    for bad in [False,True]:
        nonce='CPU-real-wire-source-identity-'+32*'0';expected=worker.expected_hello(nonce)
        assert set(expected)=={'cmd','nonce','protocol','runtime_manifest_SHA256','target_Native_SHA256'}
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1);listener.settimeout(5)
        state={'loader_calls':0,'runtime_closed':False};report={};errors=[]
        def create():
            state['loader_calls']+=1
            class Context:
                def close(self):state['context_closed']=True
            session=long_rpc.DraftSession(Context(),None,None,enabled=True)
            def close():session.close();state['runtime_closed']=True
            return SimpleNamespace(session=session,close=close)
        def serve():
            try:worker.serve_authenticated(listener,expected,create,report,lambda:None)
            except BaseException as e:errors.append(repr(e))
        thread=threading.Thread(target=serve);thread.start()
        connection=socket.create_connection(listener.getsockname(),timeout=5);connection.settimeout(5)
        try:
            try:client=publisher.DraftClient(connection,nonce+('wrong'if bad else''))
            except (EOFError,ConnectionError):assert bad
            else:
                assert not bad;assert client.command('close')['closed']
        finally:connection.close();thread.join(8);assert not thread.is_alive()
        if bad:assert errors and state['loader_calls']==0 and not report.get('authenticated')
        else:assert not errors and state['loader_calls']==1 and state['runtime_closed']and state['context_closed']and report['normal_protocol_close']
        results.append(dict(wrong_nonce=bad,passed=True,actual_worker362b_serve=True,actual_335_DraftClient=True,
                            real_TCP=True,strict_loader_recording_boundary=True,actual_GPU=False))
    assert 'torch'not in sys.modules
    return dict(passed=True,worker_manifest_SHA256=pins['packages']['Kimi-long-draft-memory-floor-r362b']['manifest_SHA256'],
                worker_path=str(wp),wire_hello_has_no_worker_manifest_field=True,worker_identity_pinned_by_source_CLI_marker_PID=True,cases=results)
