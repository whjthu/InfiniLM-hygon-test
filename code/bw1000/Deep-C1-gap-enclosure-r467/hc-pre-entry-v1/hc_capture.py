from pathlib import Path
import hashlib,time
from hc_consume import consume

def capture(read,window,destination):
 p=window['scalar_packets']['owner'];owner=Path(p['path']).read_bytes();assert hashlib.sha256(owner).hexdigest()==p['SHA256'];start=time.perf_counter();raw=read();read_end=time.perf_counter();raw=bytes(raw)if raw else None
 # Persist before validation. Null/incomplete/foreign packet stops originaldriver.
 if raw is not None:
  with destination.open('xb')as f:f.write(raw)
 q=consume(raw,owner);q.update(read_wall_s=read_end-start,validation_and_IO_wall_s=time.perf_counter()-read_end,packet=dict(path=str(destination),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()));return q
