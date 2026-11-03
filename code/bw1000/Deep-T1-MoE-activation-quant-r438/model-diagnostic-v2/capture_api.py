"""Typed CPU-byte-only consumer. Caller must own actual Native quiescence."""
import ctypes as C,hashlib,json,math,struct
from pathlib import Path
DTYPES={1:(2,'BF16'),2:(1,'I8'),3:(4,'F32'),4:(8,'I64')}
ROLE_FIELDS={0:['hidden','route_ids','route_weights','original_GU','snapshot_GU','reference_activation','reference_I8','reference_scale','candidate_activation','candidate_I8','candidate_scale'],1:['original_GU','snapshot_GU','reference_activation','reference_I8','reference_scale','candidate_activation','candidate_I8','candidate_scale']}
HASH=lambda raw:hashlib.sha256(raw).hexdigest()

class Budget:
 def __init__(self,limit=128*1024**2):self.limit=limit;self.used=0
 def charge(self,count):
  if type(count)is not int or count<0 or self.used+count>self.limit:raise RuntimeError('Frozen total raw storage bound exceeded')
  self.used+=count

def bind(lib):
 specs=[('arm',[C.c_int,C.c_int,C.c_uint64],C.c_int),('seal',[],C.c_int),('clear',[],C.c_int),('abandon',[],C.c_int),('count',[C.c_int],C.c_uint64),('record',[C.c_int,C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t],C.c_int),('tensor',[C.c_int,C.c_size_t,C.c_size_t,C.POINTER(C.c_uint64),C.c_size_t],C.c_int),('copy',[C.c_int,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t],C.c_int)]
 for name,args,result in specs:
  f=getattr(lib,'deep_model_capture_'+name+'_r438v2');f.argtypes=args;f.restype=result
 return lib

def finite(raw,code):
 if code==1:
  values=[struct.unpack('<f',struct.pack('<I',v[0]<<16))[0]for v in struct.iter_unpack('<H',raw)]
 elif code==3:values=[v[0]for v in struct.iter_unpack('<f',raw)]
 else:return True
 return all(math.isfinite(v)for v in values)

class Capture:
 def __init__(self,lib,budget):self.lib=bind(lib);self.budget=budget;self.tag=None
 def check(self,rc):
  if rc:raise RuntimeError('Typed capture return '+str(rc))
 def arm(self,mode_control,pass_index,output_index,forward_1based):
  if not mode_control.active or not mode_control.quiescent or mode_control.poisoned:raise RuntimeError('Capture arm requires original whole Native boundary')
  if pass_index not in range(4)or output_index not in [82,83]or forward_1based!=9+output_index:raise ValueError('Frozen capture geometry differs')
  self.check(self.lib.deep_model_capture_arm_r438v2(pass_index,output_index,forward_1based));self.tag=(pass_index,output_index,forward_1based)
 def collect(self,mode_control,folder):
  if self.tag is None or not mode_control.active or not mode_control.quiescent or mode_control.poisoned:raise RuntimeError('CPU records require completed original whole Native forward')
  self.check(self.lib.deep_model_capture_seal_r438v2());folder=Path(folder);folder.mkdir(exist_ok=False);summaries=[];differences=[];anomalies=[]
  for owner in range(8):
   count=int(self.lib.deep_model_capture_count_r438v2(owner))
   if count!=86:raise RuntimeError('Actual43layer x2role capture coverage differs')
   packed=bytearray();records=[];seen=set();resources=set()
   for index in range(count):
    v=(C.c_uint64*12)();self.check(self.lib.deep_model_capture_record_r438v2(owner,index,v,12));header=list(v)
    p,o,f,layer,actual_owner,role,fields,byte_count,runtime,stream,thread,source_GU=header
    if (p,o,f)!=self.tag or actual_owner!=owner or layer>=43 or role not in[0,1]or fields!=len(ROLE_FIELDS[role])or(layer,role)in seen:raise RuntimeError('Record identity/ordinal/role/field boundary differs')
    seen.add((layer,role));resources.add((runtime,stream,thread));tensors={};raws={};total=0
    for field,name in enumerate(ROLE_FIELDS[role]):
     info=(C.c_uint64*5)();self.check(self.lib.deep_model_capture_tensor_r438v2(owner,index,field,info,5));code,a,b,width,n=list(info)
     if code not in DTYPES or width!=DTYPES[code][0]or n!=a*b*width or n>8192:raise RuntimeError('Typed dtype/shape/rawbytes differ')
     shape=(a,b);rows=6 if role==0 else 1
     expected=(1,(1,4096))if name=='hidden'else(4,(1,6))if name=='route_ids'else(3,(1,6))if name=='route_weights'else(1,(rows,512))if name.endswith('GU')else(1,(rows,256))if name.endswith('activation')else(2,(rows,256))if name.endswith('I8')else(3,(rows,1))
     if (code,shape)!=expected:raise RuntimeError('Actual field dtype/shape outside frozen contract:'+name)
     data=C.create_string_buffer(n);self.check(self.lib.deep_model_capture_copy_r438v2(owner,index,field,data,n));raw=data.raw
     offset=len(packed);self.budget.charge(n);packed.extend(raw);total+=n;raws[name]=raw
     tensors[name]=dict(dtype=DTYPES[code][1],dtype_code=code,shape=list(shape),width=width,bytes=n,offset=offset,SHA256=HASH(raw))
     if not finite(raw,code):anomalies.append(dict(owner=owner,layer=layer,role=role,field=name,kind='NONFINITE_OBSERVED'))
    if total!=byte_count:raise RuntimeError('Record payload byte count differs')
    if raws['original_GU']!=raws['snapshot_GU']:anomalies.append(dict(owner=owner,layer=layer,role=role,kind='SAME_GU_SNAPSHOT_DIFFERS'))
    if role==0:
     routes=[v[0]for v in struct.iter_unpack('<q',raws['route_ids'])]
     if any(v<0 or v>=256 for v in routes):anomalies.append(dict(owner=owner,layer=layer,role=role,kind='OUT_OF_RANGE_ORIGINAL_ROUTE'))
    differing=[name for name in ['activation','I8','scale']if raws['reference_'+name]!=raws['candidate_'+name]]
    if differing:differences.append(dict(owner=owner,layer=layer,role=role,fields=differing))
    records.append(dict(record=index,header=header,tensors=tensors,same_actual_GU=raws['original_GU']==raws['snapshot_GU'],shadow_differing_fields=differing))
   if len(seen)!=86 or len(resources)!=1:raise RuntimeError('Layer role coverage or Runtime/stream/owner-thread differs')
   path=folder/('owner%d.bin'%owner)
   with path.open('xb')as out:out.write(packed)
   metadata=dict(owner=owner,tag=list(self.tag),records=records,binary=path.name,bytes=len(packed),SHA256=HASH(packed),CPU_bytes_only=True,Native_forward_completed_before_parent_read=True)
   raw=(json.dumps(metadata,indent=2)+'\n').encode();self.budget.charge(len(raw));(folder/('owner%d.json'%owner)).write_bytes(raw)
   summaries.append(dict(owner=owner,records=86,raw_bytes=len(packed),SHA256=HASH(packed),metadata_SHA256=HASH(raw)))
  self.check(self.lib.deep_model_capture_clear_r438v2());tag=self.tag;self.tag=None
  return dict(tag=list(tag),owners=summaries,same_GU_shadow_differences=differences,anomalies=anomalies,raw_budget_used=self.budget.used,owner_provided_typed_records_copied=True)
 def abort_after_quiescence(self,mode_control):
  if not mode_control.quiescent:raise RuntimeError('Cannot abandon records without actual reset/forward quiescence')
  self.check(self.lib.deep_model_capture_abandon_r438v2());self.tag=None
 def save_partial(self,folder):
  """Preserve already completed CPU bytes on failure; never reinterpret dtype."""
  folder=Path(folder);folder.mkdir(exist_ok=False);result=[]
  for owner in range(8):
   count=int(self.lib.deep_model_capture_count_r438v2(owner));meta=dict(owner=owner,partial=True,qualified=False,records=[],errors=[]);packed=bytearray()
   try:
    if not 0<=count<=86:raise RuntimeError('Partial CPU record count unknown')
    for index in range(count):
     v=(C.c_uint64*12)();self.check(self.lib.deep_model_capture_record_r438v2(owner,index,v,12));row=dict(header=list(v),fields=[]);meta['records'].append(row)
     if v[6]>11:raise RuntimeError('Partial field bound')
     for field in range(v[6]):
      info=(C.c_uint64*5)();self.check(self.lib.deep_model_capture_tensor_r438v2(owner,index,field,info,5));n=int(info[4])
      if not 0<n<=8192:raise RuntimeError('Partial raw byte bound')
      data=C.create_string_buffer(n);self.check(self.lib.deep_model_capture_copy_r438v2(owner,index,field,data,n));raw=data.raw;self.budget.charge(n)
      row['fields'].append(dict(info=list(info),offset=len(packed),bytes=n,SHA256=HASH(raw)));packed.extend(raw)
   except BaseException as error:meta['errors'].append(repr(error))
   path=folder/('owner%d.bin'%owner)
   with path.open('xb')as f:f.write(packed)
   meta.update(bytes=len(packed),SHA256=HASH(packed),CPU_bytes_only=True);raw=(json.dumps(meta,indent=2)+'\n').encode();self.budget.charge(len(raw));(folder/('owner%d.json'%owner)).write_bytes(raw);result.append(dict(owner=owner,records=len(meta['records']),bytes=len(packed),errors=meta['errors']))
  return result
