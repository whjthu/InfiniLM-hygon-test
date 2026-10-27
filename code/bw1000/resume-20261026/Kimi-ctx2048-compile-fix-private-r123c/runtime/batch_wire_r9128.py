"""Bounded JSON metadata + binary tensor transport; never pickle or NFS."""
import socket,struct,json,time
MAX_HEADER=65536;MAX_BODY=128<<20

def recv_exact(sock,n):
    buf=bytearray(n);view=memoryview(buf);pos=0
    while pos<n:
        k=sock.recv_into(view[pos:])
        if not k:raise EOFError('closed frame')
        pos+=k
    return bytes(buf)
def send(sock,meta,body=b''):
    head=json.dumps(meta,separators=(',',':'),allow_nan=False).encode()
    if len(head)>MAX_HEADER or len(body)>MAX_BODY:raise ValueError('frame bound')
    sock.sendall(struct.pack('!II',len(head),len(body))+head)
    if body:sock.sendall(body)
def recv(sock):
    a,b=struct.unpack('!II',recv_exact(sock,8))
    if not 1<=a<=MAX_HEADER or b>MAX_BODY:raise ValueError('frame bound')
    meta=json.loads(recv_exact(sock,a));body=recv_exact(sock,b)
    if not isinstance(meta,dict):raise ValueError('object metadata required')
    return meta,body
def connect(host,port,timeout=60):
    end=time.monotonic()+timeout;last=None
    while time.monotonic()<end:
        try:
            s=socket.create_connection((host,port),timeout=5);s.settimeout(120);s.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);return s
        except OSError as e:last=e;time.sleep(.2)
    raise TimeoutError(str(last))
def listen(host,port):
    s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((host,port));s.listen(1);s.settimeout(1800);return s
