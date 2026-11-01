"""Authenticate/load-order/cleanup checks. The loader is a recording boundary."""
import copy
import json
import socket
import sys
import threading
from types import SimpleNamespace
from worker import expected_hello,serve_authenticated
from batch_wire_r9128 import send,recv
from long_rpc import DraftSession


class Listener:
    def __init__(self,socket):self.socket=socket;self.closed=False
    def accept(self):return self.socket,('CPU_socketpair',0)
    def close(self):self.closed=True


def main():
    cases=[]
    hello=expected_hello('0123456789abcdef'*4)
    for mode in ['positive','wrong_nonce','wrong_protocol','wrong_runtime','wrong_target','extra_body',
                 'EOF_before_hello','loader_failure','cleanup_failure']:
        client,server=socket.socketpair();client.settimeout(4);server.settimeout(4)
        listener=Listener(server);state={'loads':0,'closes':0};report={};errors=[]
        def load():
            state['loads']+=1
            if mode=='loader_failure':raise RuntimeError('CPU_injected_loader_failure')
            class Context:
                def close(self):state['context_closed']=True
            session=DraftSession(Context(),None,None,enabled=True)
            def close():
                state['closes']+=1;session.close()
                if mode=='cleanup_failure':raise RuntimeError('CPU_injected_cleanup_failure')
            return SimpleNamespace(session=session,close=close)
        def serve():
            try:serve_authenticated(listener,hello,load,report,lambda:None)
            except BaseException as e:errors.append(repr(e))
        t=threading.Thread(target=serve);t.start()
        if mode=='EOF_before_hello':client.close()
        else:
            m=copy.deepcopy(hello)
            field={'wrong_nonce':'nonce','wrong_protocol':'protocol','wrong_runtime':'runtime_manifest_SHA256','wrong_target':'target_Native_SHA256'}.get(mode)
            if field:m[field]='wrong'
            send(client,m,b'x'if mode=='extra_body'else b'')
            if mode in ['positive','cleanup_failure']:
                ready,body=recv(client);assert ready['ready']and not body
                send(client,{'cmd':'close','seq':0});assert recv(client)[0]['closed']
            else:
                try:recv(client)
                except (EOFError,ConnectionError):pass
                else:raise AssertionError('Rejected hello/loader produced ready')
            client.close()
        t.join(5);assert not t.is_alive()and listener.closed and server.fileno()==-1
        if mode=='positive':assert not errors and state['loads']==state['closes']==1 and report['normal_protocol_close']
        elif mode=='cleanup_failure':assert errors and state['loads']==state['closes']==1 and state['context_closed']
        elif mode=='loader_failure':assert errors and state['loads']==1 and not state['closes']
        else:assert errors and state['loads']==state['closes']==0
        cases.append({'name':mode,'passed':True,'loader_calls':state['loads']})
    assert 'torch'not in sys.modules
    result={'completed':True,'passed':True,'CPU_only':True,'cases':cases,
        'GPU':0,'checkpoint_weights':0,'actual_listener_accept_or_loader_tested':False,
        'actual_socket_wire_and_phase_close':True,'recording_loader_boundary':True}
    print(json.dumps(result));return result


if __name__=='__main__':main()
