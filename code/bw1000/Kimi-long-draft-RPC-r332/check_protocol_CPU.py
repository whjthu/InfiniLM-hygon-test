"""Protocol/actual socket framing checks with recording feature-consumer boundary.

No Torch, real projection, checkpoint, target commit, CUDA or model quality proof.
"""
import copy
import json
import socket
import threading
from long_rpc import DraftSession, serve_connection
from batch_wire_r9128 import send, recv


class Context:
    def __init__(self):
        self.epoch=0; self.phase='closed'; self.length=0; self.prompt_tokens=0; self.appends=[]
    def _check(self,e,r):
        assert e==self.epoch and r==1 and self.phase in ['prompt','decode']
    def begin(self,e,r,p):
        assert e>self.epoch and self.phase in ['closed','retired']
        self.epoch=e; self.phase='prompt'; self.length=0; self.prompt_tokens=p
    def append_prompt(self,e,r,p,part,fence): self._append(part,fence)
    def append_committed(self,e,r,p,part,fence): self._append(part,fence)
    def _append(self,part,fence):
        fence();self.length+=part;self.appends.append(part)
    def seal_prompt(self,e,r):
        assert self.phase=='prompt'and self.length==self.prompt_tokens
        self.phase='decode'
    def query(self,e,r,h,a,K,return_debug):
        assert self.phase=='decode'and return_debug
        return {'mode':'ar_tail','values':None}if self.length>32760 else {'mode':'draft','values':([a]*K,{'finite':True})}
    def retire(self,e,r): self.phase='retired';return True
    def close(self):self.phase='closed'


class Codec:
    raw_dim=40;vocab=163840
    def decode(self,b,n):return n
    def fence(self):pass
    def proposals(self,v,d,K):assert d['finite']and len(v)==K;return v


def rejection(fn):
    try:fn()
    except (ValueError,RuntimeError,AssertionError,KeyError):return
    raise AssertionError('Invalid protocol accepted')


def session():return DraftSession(Context(),None,Codec(),enabled=True)
def packet(s,cmd,**fields):return dict(cmd=cmd,seq=s.sequence,epoch=1,request=1,**fields)
def prompt(s,n=512):
    s.dispatch(packet(s,'begin',prompt_tokens=n),b'')
    s.dispatch(packet(s,'append_prompt',past=0,rows=n,capture_epoch=1),bytes(n*80))
    s.dispatch(packet(s,'seal',past=n),b'')


def main():
    cases=[]
    rejection(lambda:DraftSession(Context(),None,Codec()))
    s=session();prompt(s)
    assert s.dispatch(packet(s,'query',past=512,anchor=7,K=7),b'')['proposals']==[7]*7
    proof=dict(capture_epoch=2,past_before=512,past_after=514,accepted_inputs=2,complete=True)
    s.dispatch(packet(s,'append_committed',past=512,rows=2,capture_epoch=2,target_commit=proof),bytes(160))
    assert s.context.length==514 and s.context.appends==[512,2]
    s.dispatch(packet(s,'retire'),b'');s.dispatch(dict(cmd='close',seq=s.sequence),b'');assert s.closed
    cases.append('prompt512_seal_query_committedprefix_retire_close')
    for name,change,body in [
        ('wrong_seq',{'seq':99},bytes(80)),('bool_seq',{'seq':True},bytes(80)),
        ('wrong_epoch',{'epoch':2},bytes(80)),('wrong_request',{'request':2},bytes(80)),
        ('stale_past',{'past':511},bytes(80)),('stale_capture',{'capture_epoch':1},bytes(80)),
        ('oversize_decode',{'rows':9},bytes(720)),('wrong_bytes',{},bytes(79)),
        ('extra_metadata',{'unused':1},bytes(80)),
        ('incomplete_commit',{'target_commit':dict(proof,complete=False)},bytes(80)),
        ('rejected_suffix',{'target_commit':dict(capture_epoch=2,past_before=512,past_after=513,accepted_inputs=2,complete=True)},bytes(80))]:
        s=session();prompt(s);before=(s.sequence,s.last_capture_epoch,s.context.length,list(s.context.appends))
        m=packet(s,'append_committed',past=512,rows=1,capture_epoch=2,
                 target_commit=dict(capture_epoch=2,past_before=512,past_after=513,accepted_inputs=1,complete=True))
        m.update(change);rejection(lambda:s.dispatch(m,body))
        assert before==(s.sequence,s.last_capture_epoch,s.context.length,s.context.appends)and s.poisoned
        rejection(lambda:s.dispatch(packet(s,'query',past=512,anchor=7,K=7),b''));s.close();cases.append(name)
    s=session();s.dispatch(packet(s,'begin',prompt_tokens=512),b'')
    rejection(lambda:s.dispatch(packet(s,'query',past=1,anchor=7,K=7),b''));s.close();cases.append('query_before_seal')
    s=session();errors=[]
    def foreign():
        try:s.dispatch(packet(s,'begin',prompt_tokens=512),b'')
        except RuntimeError:errors.append(1)
    t=threading.Thread(target=foreign);t.start();t.join();assert errors==[1]and s.sequence==0;s.close();cases.append('foreign_owner')
    # Actual bounded binary transport and server cleanup on clean close and EOF.
    for mode in ['close','EOF']:
        client,server=socket.socketpair();client.settimeout(3);server.settimeout(3);seen={}
        def run():
            x=session();seen['session']=x
            try:serve_connection(server,x)
            except EOFError:seen['EOF']=True
            finally:server.close()
        t=threading.Thread(target=run);t.start()
        send(client,dict(cmd='begin',seq=0,epoch=1,request=1,prompt_tokens=1));answer,body=recv(client)
        assert answer['phase']=='prompt'and not body
        if mode=='close':
            send(client,dict(cmd='append_prompt',seq=1,epoch=1,request=1,past=0,rows=1,capture_epoch=1),bytes(80))
            assert recv(client)[0]['past_after']==1
            send(client,dict(cmd='seal',seq=2,epoch=1,request=1,past=1));assert recv(client)[0]['phase']=='decode'
            send(client,dict(cmd='query',seq=3,epoch=1,request=1,past=1,anchor=7,K=7));assert recv(client)[0]['proposals']==[7]*7
            send(client,dict(cmd='retire',seq=4,epoch=1,request=1));assert recv(client)[0]['removed']
            send(client,dict(cmd='close',seq=5));assert recv(client)[0]['closed']
        client.close();t.join(4);assert not t.is_alive()and seen['session'].closed
        cases.append('socket_'+mode+'_cleanup')
    result={'completed':True,'passed':True,'CPU_only':True,'actual_binary_socketpair':True,
        'cases':cases,'recording_feature_consumer_boundary':True,'Torch_executed':False,
        'real_target_commit_or_GPU_qualified':False,'GPU':0,'Model':0}
    print(json.dumps(result));return result


if __name__=='__main__':main()
