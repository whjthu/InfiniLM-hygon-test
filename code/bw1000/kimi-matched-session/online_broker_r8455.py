"""Real bounded TCP CPU coordinator. Not yet GPU/online performance qualified.

All timestamps in scheduler events are coordinator perf_counter observations.
Handlers only frame/authenticate and enqueue. A single coordinator thread owns
Scheduler. Owner main processes execute adapter outside the broker. Lost replies
or unknown native execution poison the session: NO reconnect/replay/resubmit.
"""
import dataclasses
import json
import os
import queue
import socket
import socketserver
import threading
import time
from online_service_core_r8455 import Scheduler

MAX_FRAME=256*1024


def send_frame(sock, value):
    data=json.dumps(value,allow_nan=False,separators=(',',':')).encode()+b'\n'
    if len(data)>MAX_FRAME:raise ValueError('frame too large')
    sock.sendall(data)


def read_frame(reader):
    raw=reader.readline(MAX_FRAME+1)
    if not raw or len(raw)>MAX_FRAME or not raw.endswith(b'\n'):
        raise ValueError('EOF/incomplete/oversized frame')
    value=json.loads(raw)
    if not isinstance(value,dict):raise ValueError('object frame required')
    return value


class Journal:
    """Append-only event sink, fsync before clearing core buffer.

    Open failure or partial write is failstop. Never retry a native job. Startup
    recovery/resume deliberately absent; use a new session after diagnosis.
    """
    def __init__(self,path):
        self.file=open(path,'x',encoding='utf8')
        self.lock=threading.Lock()
    def __call__(self,events):
        with self.lock:
            for event in events:self.file.write(json.dumps(event,allow_nan=False)+'\n')
            self.file.flush()
            os.fsync(self.file.fileno())
    def close(self):self.file.close()


class Broker:
    def __init__(self, *, nonce, journal, scheduler, host='127.0.0.1',port=0,max_pending=32,max_connections=16):
        assert isinstance(nonce,str) and len(nonce)>=24
        self.nonce,self.journal,self.core=nonce,journal,scheduler
        self.commands=queue.Queue(max_pending)
        self.failure=None
        self.stopped=threading.Event()
        self.owner_connections={}
        self.owner_sessions={}
        self.next_request=0
        self.accepting=True
        self.peers=threading.BoundedSemaphore(max_connections)
        outer=self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(7200) # actual Engine calls may run minutes.
                self.request.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                role=None
                try:
                    hello=read_frame(self.rfile)
                    if hello.get('nonce')!=outer.nonce:raise ValueError('authentication')
                    role=hello.get('role')
                    if role not in ('owner','ingress','observer'):raise ValueError('role')
                    # Stage registration is serialized through coordinator too.
                    result=outer.submit(dict(op='hello',role=role,stage=hello.get('stage')),time.perf_counter(),self)
                    send_frame(self.request,result)
                    if not result['ok']:return
                    while not outer.stopped.is_set():
                        req=read_frame(self.rfile)
                        stamp=time.perf_counter() # retained across queue/journal waiting.
                        req['role']=role
                        result=outer.submit(req,stamp,self)
                        send_frame(self.request,result)
                        if req.get('op')=='bye':return
                except Exception as error:
                    if role=='owner':outer.poison('owner transport lost; execution unknown: '+str(error))
                finally:
                    # Owner normal bye is accepted only with all jobs drained.
                    if role=='owner' and self in outer.owner_connections.values():
                        outer.poison('owner disconnected without verified drained bye')

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads=True
            allow_reuse_address=False
            def process_request(self,request,address):
                if not outer.peers.acquire(False):
                    try:send_frame(request,dict(ok=False,error='connection_limit'))
                    except OSError:pass
                    request.close();return
                try:super().process_request(request,address)
                except BaseException:
                    outer.peers.release();raise
            def process_request_thread(self,request,address):
                try:super().process_request_thread(request,address)
                finally:outer.peers.release()
        self.server=Server((host,port),Handler)
        self.address=self.server.server_address
        self.coordinator=threading.Thread(target=self.run,name='sole-scheduler-owner',daemon=True)
        self.transport=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.coordinator.start();self.transport.start()

    def poison(self,reason):
        # Sticky session failure; do not touch Scheduler from transport threads.
        if self.failure is None:self.failure=str(reason)[:1024]

    def submit(self,req,arrival,connection):
        reply=queue.Queue(1)
        item=(req,arrival,connection,reply)
        try:self.commands.put_nowait(item)
        except queue.Full:
            # This is an explicit transport rejection, returned with original
            # arrival. No request was admitted or native work submitted.
            self.journal([dict(kind='ingress_rejection',t=arrival,reason='coordinator_queue_full',role=req.get('role'))])
            if req.get('role')=='owner':self.poison('owner command queue overflow; no replay')
            return dict(ok=False,error='coordinator_queue_full',arrival=arrival)
        try:return reply.get(timeout=10)
        except queue.Empty:
            self.poison('command response unknown; no replay')
            return dict(ok=False,error='unknown_response_no_replay',arrival=arrival)

    def flush(self):
        self.core.drain_events(self.journal)

    def run(self):
        while not self.stopped.is_set():
            try:req,arrival,conn,reply=self.commands.get(timeout=.1)
            except queue.Empty:continue
            try:
                if self.failure:raise RuntimeError(self.failure)
                self.flush() # failed persistence poisons; command stays unapplied.
                result=self.handle(req,arrival,conn)
                self.flush() # durability before successful control response.
                reply.put(dict(ok=True,**result))
            except BaseException as error:
                self.poison(type(error).__name__+': '+str(error))
                reply.put(dict(ok=False,error=self.failure))

    def handle(self,req,arrival,conn):
        op,role=req['op'],req['role'];now=time.perf_counter()
        if op=='hello':
            if role=='owner':
                stage=req['stage'];assert type(stage)is int and 0<=stage<self.core.stages
                assert stage not in self.owner_sessions,'owner reconnect forbidden'
                self.owner_sessions[stage]=conn;self.owner_connections[stage]=conn
                self.dispatch_ready()
            return dict(hello=True)
        if role=='owner':
            stage=next(k for k,v in self.owner_connections.items() if v is conn)
            if op=='next':
                if len(self.owner_connections)!=self.core.stages:return dict(wait=True)
                job=self.core.take(stage,now)
                return dict(wait=job is None,job=None if job is None else dataclasses.asdict(job))
            if op=='finish':
                self.core.finish(stage,req['seq'],now,ids=req.get('ids'))
                self.dispatch_ready()
                return dict(acked=req['seq'])
            if op=='bye':
                assert self.core.can_reset_epoch() and not self.core.queue,'bye before complete world drain'
                del self.owner_connections[stage]
                return dict(closed=True)
            raise ValueError('invalid owner op')
        if role=='ingress':
            assert op=='arrive' and self.accepting
            prompt,limit=req['prompt'],req['limit']
            assert isinstance(prompt,list) and 0<len(prompt)<=1024
            key=self.next_request;self.next_request+=1
            accepted=self.core.arrive(key,prompt,limit,now,arrival_time=arrival)
            self.dispatch_ready()
            return dict(request=key,accepted=accepted,arrival=arrival)
        assert role=='observer'
        if op=='status':
            return dict(jobs=len(self.core.jobs),queued=len(self.core.queue),ready=len(self.core.ready),
                        busy=self.core.busy[:],cursors=self.core.cursor[:],next_job=self.core.next_job,
                        can_reset=self.core.can_reset_epoch(),failure=self.failure)
        raise ValueError('invalid observer op')

    def dispatch_ready(self):
        if len(self.owner_connections)!=self.core.stages:return
        while True:
            self.flush()
            if self.core.dispatch(time.perf_counter()) is None:break

    def close(self):
        self.stopped.set();self.server.shutdown();self.server.server_close()
        self.transport.join(3);self.coordinator.join(3)


class Client:
    def __init__(self,address,nonce,role,stage=None):
        self.sock=socket.create_connection(address,timeout=10)
        self.sock.settimeout(10);self.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.reader=self.sock.makefile('rb')
        self.call(dict(op='hello',nonce=nonce,role=role,stage=stage))
    def call(self,req):
        send_frame(self.sock,req);result=read_frame(self.reader)
        if not result['ok']:raise RuntimeError(result['error'])
        return result
    def close(self):
        self.reader.close();self.sock.close()


def owner_loop(address,nonce,stage,adapter,stop_event,receipt_queue=None):
    """Run in owner PROCESS MAIN THREAD. No socket handler owns an Engine.

    adapter.forward(job) synchronous; returns IDs only on emitter. Never retry
    forward or finish after any transport/native exception. Adapter.close must
    own appropriate actual owner drain/destroy; native timeout needs external
    Slurm watchdog/postKFD and is not solved by Python finally.
    """
    client=Client(address,nonce,'owner',stage)
    close_attempted=False
    try:
        while not stop_event.is_set():
            response=client.call(dict(op='next'))
            if response['wait']:
                time.sleep(.002);continue
            job=response['job']
            ids=adapter.forward(job)
            client.call(dict(op='finish',seq=job['seq'],ids=ids))
        close_attempted=True
        adapter.close()
        client.call(dict(op='bye'))
        if receipt_queue is not None:receipt_queue.put(dict(stage=stage,normal=True))
    except BaseException as error:
        if receipt_queue is not None:receipt_queue.put(dict(stage=stage,normal=False,error=repr(error)))
        raise
    finally:
        try:
            if not close_attempted:adapter.close()
        finally:client.close()
