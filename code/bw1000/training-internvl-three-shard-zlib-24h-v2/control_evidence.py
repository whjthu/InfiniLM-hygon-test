"""Persist bounded private control/query bytes before decoding or checking them."""
import hashlib,os
from common import need,save

def read_control_bytes(fd,cap=65536):
    raw=bytearray()
    while len(raw)<=cap:
        part=os.read(fd,1)
        if not part:break
        raw.extend(part)
        if part==b'\n':break
    return bytes(raw)

def persist_bytes(root,name,raw,cap,context):
    need(type(raw)is bytes and '/' not in name and '\\' not in name,'EXACT_PRIVATE_CONTROL_MEMBER')
    stored=raw[:cap];path=root/name
    with path.open('xb') as f:f.write(stored);f.flush();os.fsync(f.fileno())
    pin=dict(path=name,bytes=len(stored),SHA256=hashlib.sha256(stored).hexdigest(),observed_bytes=len(raw),observed_SHA256=hashlib.sha256(raw).hexdigest(),truncated=len(raw)>cap)
    save(root/(name+'.json'),dict(context,**pin,private_unvalidated_bytes=True))
    return pin
