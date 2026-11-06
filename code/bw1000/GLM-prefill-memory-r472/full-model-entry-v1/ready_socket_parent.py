"""One-use verified scalar-ready relay; no Native/request/tensor transport.

Reuses the PD445 early-listener/one-connection pattern and strict JSON framing
idea. PD role constants, ticket/state transfer and long-lived control are absent.
This removes stage1 directory-polling dependence; it is not a proven NFS fix.
"""
import hashlib
import hmac
import ipaddress
import json
import math
import secrets
import socket
import struct
import time

PORT=43844
MAX_FRAME=256*1024


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def no_duplicates(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate ready JSON key')
        result[key]=value
    return result


def decode(raw):
    def nonfinite(value):raise ValueError('Nonfinite ready JSON')
    return json.loads(raw,object_pairs_hook=no_duplicates,parse_constant=nonfinite)


def addresses(config):
    # Freeze numeric addresses in the concrete CPU/GPU claim. No DNS lookup can
    # silently outlive this channel's timeout or select a different interface.
    ips=config['ready_metadata_IPs']
    if type(ips)is not list or len(ips)!=2:raise ValueError('Two fixed ready IPs required')
    for ip in ips:
        if type(ip)is not str or str(ipaddress.IPv4Address(ip))!=ip:
            raise ValueError('Ready relay requires canonical numeric IPv4')
    return ips


def check(run,cancelled,deadline):
    if cancelled():raise RuntimeError('Owned ready relay cancelled')
    if run.stop_requested():raise RuntimeError('STOP/FAILED before verified ready relay')
    if time.monotonic()>=deadline:raise TimeoutError('Original ready relay deadline expired')


def deadline_for(run,seconds=None):
    remaining=min(run.config['gpu_start_epoch']+2700-time.time(),
                  run.config['gpu_start_epoch']+run.config['service_seconds']-120-time.time())
    if seconds is not None:remaining=min(remaining,seconds)
    if remaining<=0:raise TimeoutError('Original startup deadline exhausted')
    return time.monotonic()+remaining


def send_json(sock,value,run,cancelled,deadline):
    raw=canonical(value)
    if not 0<len(raw)<=MAX_FRAME:raise ValueError('Ready frame exceeds fixed256KiB')
    pending=memoryview(struct.pack('!I',len(raw))+raw)
    while pending:
        check(run,cancelled,deadline);sock.settimeout(min(.25,max(.001,deadline-time.monotonic())))
        try:count=sock.send(pending)
        except socket.timeout:continue
        if count<=0:raise EOFError('Ready peer closed during send')
        pending=pending[count:]


def receive_exact(sock,count,run,cancelled,deadline):
    data=bytearray()
    while len(data)<count:
        check(run,cancelled,deadline);sock.settimeout(min(.25,max(.001,deadline-time.monotonic())))
        try:part=sock.recv(count-len(data))
        except socket.timeout:continue
        if not part:raise EOFError('Truncated ready relay frame')
        data.extend(part)
    return bytes(data)


def receive_json(sock,run,cancelled,deadline):
    size,=struct.unpack('!I',receive_exact(sock,4,run,cancelled,deadline))
    if not 0<size<=MAX_FRAME:raise ValueError('Ready frame exceeds fixed256KiB')
    return decode(receive_exact(sock,size,run,cancelled,deadline))


class ReadyReceiver:
    def __init__(self,run,cancelled=lambda:False):
        if run.config['stage']!=1:raise ValueError('Only stage1 receives ready proof')
        self.run,self.cancelled=run,cancelled
        self.ips=addresses(run.config);self.listener=self.connection=None;self.used=False
        self.cleanup_errors=[]
        self.stage0_all_ready_return_unix=None

    def __enter__(self):
        try:
            self.listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            self.listener.set_inheritable(False)
            self.listener.bind((self.ips[1],PORT));self.listener.listen(1)
            return self
        except BaseException:
            self.close()
            raise

    def receive(self):
        if self.used or self.listener is None:raise RuntimeError('Ready receiver is one-use')
        self.used=True;limit=deadline_for(self.run)
        while True:
            check(self.run,self.cancelled,limit);self.listener.settimeout(min(.25,max(.001,limit-time.monotonic())))
            try:self.connection,peer=self.listener.accept();break
            except socket.timeout:continue
        self.listener.close();self.listener=None
        if peer[0]!=self.ips[0]:raise ValueError('Ready sender IP differs from fixedstage0')
        limit=min(limit,deadline_for(self.run,120));nonce=secrets.token_hex(32)
        identity=self.run.identity()
        send_json(self.connection,dict(kind='ready-challenge-v1',nonce=nonce,identity=identity),
                  self.run,self.cancelled,limit)
        value=receive_json(self.connection,self.run,self.cancelled,limit)
        if type(value)is not dict or set(value)!={'kind','nonce','identity','records','stage0_all_ready_return_unix'}:
            raise ValueError('Ready response fields differ')
        if value['kind']!='ready-records-v1'or value['identity']!=identity:
            raise ValueError('Foreign ready response identity')
        if type(value['nonce'])is not str or not hmac.compare_digest(value['nonce'],nonce):
            raise ValueError('Foreign ready challenge nonce')
        first=value['stage0_all_ready_return_unix']
        if type(first)not in (int,float)or not math.isfinite(first)or not self.run.config['gpu_start_epoch']<=first<self.run.config['gpu_start_epoch']+2700:
            raise ValueError('Invalid fixed stage0 all_ready return timestamp')
        if time.time()>=first+120:raise TimeoutError('Additional stage0-return120 guard expired')
        limit=min(limit,time.monotonic()+max(0,first+120-time.time()))
        records=self.run.validate_ready_records(value['records'])
        self.stage0_all_ready_return_unix=first
        check(self.run,self.cancelled,limit)
        digest=hashlib.sha256(canonical(records)).hexdigest()
        send_json(self.connection,dict(kind='ready-verified-v1',nonce=nonce,identity=identity,
                  records_SHA256=digest,stage0_all_ready_return_unix=first),self.run,self.cancelled,limit)
        # Sender actively closes after ACK. Passive receiver close avoids
        # leaving the fixed listening port in a new TIME_WAIT lifecycle.
        eof_limit=min(limit,time.monotonic()+5)
        while True:
            check(self.run,self.cancelled,eof_limit);self.connection.settimeout(min(.25,max(.001,eof_limit-time.monotonic())))
            try:extra=self.connection.recv(1);break
            except socket.timeout:continue
        if extra:raise ValueError('Extra data after single ready response')
        check(self.run,self.cancelled,limit)
        return records

    def close(self):
        for name in ['connection','listener']:
            obj=getattr(self,name,None)
            if obj is not None:
                try:obj.close()
                except BaseException as error:self.cleanup_errors.append((name,repr(error)))
                finally:setattr(self,name,None)

    def __exit__(self,kind,error,tb):
        self.close()
        if self.cleanup_errors:
            if error is not None:
                if hasattr(error,'add_note'):error.add_note('Ready socket cleanup: '+repr(self.cleanup_errors))
            else:raise RuntimeError('Ready socket cleanup failed: '+repr(self.cleanup_errors))
        return False


def send_verified_ready(run,records,stage0_all_ready_return_unix,cancelled=lambda:False):
    if run.config['stage']!=0:raise ValueError('Only actualstage0 relays ready records')
    # Exact same validator before send and after receipt. No arbitrary READY bit.
    records=run.validate_ready_records(records);ips=addresses(run.config)
    if type(stage0_all_ready_return_unix)not in(int,float)or not math.isfinite(stage0_all_ready_return_unix):
        raise ValueError('Fixed stage0 all_ready return time required')
    remaining=stage0_all_ready_return_unix+120-time.time()
    if remaining<=0:raise TimeoutError('Additional stage0-return120 guard expired before connect')
    limit=min(deadline_for(run,120),time.monotonic()+remaining)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM)as sock:
        sock.set_inheritable(False);sock.bind((ips[0],0))
        check(run,cancelled,limit);sock.settimeout(min(15,max(.001,limit-time.monotonic())))
        sock.connect((ips[1],PORT))  # One connect attempt, no restart/retry.
        hello=receive_json(sock,run,cancelled,limit)
        if type(hello)is not dict or set(hello)!={'kind','nonce','identity'}or hello['kind']!='ready-challenge-v1'or hello['identity']!=run.identity():
            raise ValueError('Foreign ready receiver challenge')
        nonce=hello['nonce']
        if type(nonce)is not str or len(nonce)!=64 or any(c not in'0123456789abcdef'for c in nonce):
            raise ValueError('Malformed ready challenge nonce')
        send_json(sock,dict(kind='ready-records-v1',nonce=nonce,identity=run.identity(),records=records,
                           stage0_all_ready_return_unix=stage0_all_ready_return_unix),
                  run,cancelled,limit)
        ack=receive_json(sock,run,cancelled,limit)
        if ack!=dict(kind='ready-verified-v1',nonce=nonce,identity=run.identity(),
                     records_SHA256=hashlib.sha256(canonical(records)).hexdigest(),
                     stage0_all_ready_return_unix=stage0_all_ready_return_unix):
            raise ValueError('Ready verification ACK differs')
