"""Bounded binary frames for isolated Kimi trained-draft experiments."""
import hashlib,json,socket,struct
MAGIC=b'K484';MAX_HEADER=262144;MAX_BODY=6*8192*7168*2
def receive_exact(sock,n):
    result=bytearray()
    while len(result)<n:
        part=sock.recv(min(n-len(result),1048576))
        if not part:raise EOFError('truncated draft RPC frame')
        result.extend(part)
    return bytes(result)
def send(sock,header,body=b''):
    assert isinstance(body,bytes)and len(body)<=MAX_BODY
    meta=dict(header);meta['body_sha256']=hashlib.sha256(body).hexdigest()
    raw=json.dumps(meta,separators=(',',':')).encode();assert len(raw)<=MAX_HEADER
    sock.sendall(struct.pack('!4sIQ',MAGIC,len(raw),len(body))+raw+body)
def receive(sock):
    magic,h,n=struct.unpack('!4sIQ',receive_exact(sock,16))
    if magic!=MAGIC or not 0<h<=MAX_HEADER or n>MAX_BODY:raise ValueError('invalid frame bounds')
    meta=json.loads(receive_exact(sock,h));body=receive_exact(sock,n)
    if hashlib.sha256(body).hexdigest()!=meta.pop('body_sha256'):raise ValueError('body SHA mismatch')
    return meta,body
def tune(sock):
    sock.settimeout(300);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);return sock

