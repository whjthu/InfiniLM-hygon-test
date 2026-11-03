"""Parent reads independent CPU bytes after the real whole-Native return."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import struct

POINTS=('L2exit','L3entry','L3attn_norm','L3attn_value','L3attn_residual','L3ffn_collapsed','L3ffn_norm')
H=lambda raw:hashlib.sha256(raw).hexdigest()


class Capture:
 def __init__(self,lib,budget):
  self.lib,self.budget,self.tag=lib,budget,None
  specs=[('arm',[C.c_int,C.c_int,C.c_uint64],C.c_int),('seal',[],C.c_int),('abandon',[],C.c_int),('clear',[],C.c_int),
   ('count',[C.c_int],C.c_uint64),('info',[C.c_int,C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t],C.c_int),
   ('copy',[C.c_int,C.c_size_t,C.c_void_p,C.c_size_t],C.c_int)]
  for name,args,result in specs:
   f=getattr(lib,'deep_boundary_'+name+'_r447');f.argtypes=args;f.restype=result
 def check(self,rc):
  if rc:raise RuntimeError('r447 typed save rc='+str(rc))
 def quiescence(self,control):
  if not control.active or not control.quiescent or control.poisoned:raise RuntimeError('Require actual original whole-Native completion/reset, not HostReturned')
 def arm(self,control,p,o,f):
  self.quiescence(control)
  if p not in (0,1)or o not in (82,83)or f!=9+o:raise ValueError('Frozen OFF/OFF window')
  self.check(self.lib.deep_boundary_arm_r447(p,o,f));self.tag=(p,o,f)
 def save(self,control,folder,*,partial=False):
  if partial:
   if not control.quiescent:raise RuntimeError('Partial CPU save requires original completion/reset drain')
  else:self.quiescence(control)
  if self.tag is None:raise RuntimeError('No actual armed window')
  self.check(self.lib.deep_boundary_abandon_r447()if partial else self.lib.deep_boundary_seal_r447())
  folder=Path(folder);folder.mkdir(exist_ok=False);owners=[];nonfinite=[]
  for owner in range(8):
   count=int(self.lib.deep_boundary_count_r447(owner))
   if not 0<=count<=7 or(not partial and count!=7):raise RuntimeError('Actual 7point owner coverage differs')
   data=bytearray();records=[];seen=set();resources=set()
   for ordinal in range(count):
    buf=(C.c_uint64*15)();self.check(self.lib.deep_boundary_info_r447(owner,ordinal,buf,15));v=list(buf)
    p,o,f,actual_owner,layer,point,code,ndim,d0,d1,d2,n,runtime,stream,thread=v
    residual=point in (0,1,4);shape=[d0,d1,d2]if ndim==3 else[d0,d1]
    if(p,o,f)!=self.tag or actual_owner!=owner or point not in range(7)or point in seen or layer!=(2 if point==0 else 3)or code!=1 or shape!=([1,4,4096]if residual else[1,4096])or n!=(32768 if residual else 8192):raise RuntimeError('Actual typed boundary identity/shape differs')
    if not runtime or not thread:raise RuntimeError('Missing actual owner identity')
    seen.add(point);resources.add((runtime,stream,thread));rawbuf=C.create_string_buffer(n);self.check(self.lib.deep_boundary_copy_r447(owner,ordinal,rawbuf,n));raw=rawbuf.raw
    finite=all(math.isfinite(struct.unpack('<f',struct.pack('<I',h[0]<<16))[0])for h in struct.iter_unpack('<H',raw))
    if not finite:nonfinite.append(dict(owner=owner,point=POINTS[point]))
    self.budget.charge(n);records.append(dict(header=v,name=POINTS[point],shape=shape,dtype='BF16',offset=len(data),bytes=n,SHA256=H(raw),finite=finite));data.extend(raw)
   if count and len(resources)!=1:raise RuntimeError('Runtime/stream/thread identity changed within owner')
   by={r['name']:data[r['offset']:r['offset']+r['bytes']]for r in records}
   continuity=by.get('L2exit')==by.get('L3entry')if'L2exit'in by and'L3entry'in by else None
   path=folder/('owner%d.bin'%owner)
   with path.open('xb')as out:out.write(data)
   meta=dict(owner=owner,tag=list(self.tag),partial=partial,records=records,L2exit_equals_L3entry=continuity,CPU_bytes_only=True,actual_whole_Native_forward_completed_before_read=not partial,whole_engine_quiescence_before_read=True,binary=path.name,bytes=len(data),SHA256=H(data))
   raw=(json.dumps(meta,indent=2)+'\n').encode();self.budget.charge(len(raw));(folder/('owner%d.json'%owner)).write_bytes(raw)
   owners.append(dict(owner=owner,records=count,raw_bytes=len(data),SHA256=H(data),metadata_SHA256=H(raw),L2exit_equals_L3entry=continuity))
  self.check(self.lib.deep_boundary_clear_r447());tag=self.tag;self.tag=None
  return dict(tag=list(tag),owners=owners,nonfinite=nonfinite,partial=partial,raw_budget_used=self.budget.used)
