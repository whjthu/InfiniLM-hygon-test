"""Actual typed snapshot consumer, no Torch/HTTP/Native import side effects."""
from pathlib import Path
import ctypes,hashlib,json,math,re
def native_error(lib):
 fn=getattr(lib,'glm_attention_error_r398',None)
 value=fn()if fn else None
 return value.decode(errors='replace')if isinstance(value,bytes)else 'error getter unavailable'
def validate_descriptor(q,mode,detail):
 assert q['quiesced']is True and q['mode']==mode and q['detail']==detail and type(q['ticket'])is int and q['ticket']in [1,2]
 assert type(q['query_CAPI_completed'])is bool and isinstance(q['slots'],list)and 0<len(q['slots'])<=24
 names=set();keys=set();total=0;sizes={'U8':1,'BF16':2,'F32':4}
 for slot in q['slots']:
  name=slot['name'];key=slot['key'];shape=slot['shape'];dtype=slot['dtype'];n=slot['nbytes']
  assert isinstance(name,str)and re.fullmatch('[A-Za-z0-9_.-]+',name)and name not in names;names.add(name)
  assert type(key)is int and key>=0 and key not in keys;keys.add(key)
  assert isinstance(shape,list)and 0<len(shape)<=3 and all(type(v)is int and v>0 for v in shape)
  assert dtype in sizes and type(n)is int and n==math.prod(shape)*sizes[dtype]and n<=32*1024**2
  total+=n
 assert keys==set(range(len(q['slots'])))and total==q['total_bytes']and total<=32*1024**2
 return q
def read_snapshot(lib,ctx,directory,prefix,*,mode,detail):
 """Call after query (even failed/partial), before diagnostic numerical vote.

 The owner supplies only quiesced written tensors. This never treats snapshot
 admission or a successful copy as numerical/Native-query qualification.
 Always clear owned snapshot; preserve a prior copy/file error if clear fails.
 """
 assert re.fullmatch('[A-Za-z0-9_.-]+',prefix);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 report_path=directory/(prefix+'.json');report=dict(completed=False,collection_passed=False,files=[],mode=mode,detail=detail,query_qualification=False);primary=None;owned=False
 def save():
  tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(report_path)
 try:
  with report_path.open('x')as f:f.write(json.dumps(report))
  owned=True;raw=lib.glm_attention_diag_desc_r433(ctx)
  if not raw:raise RuntimeError('Native snapshot unavailable; no valid output claimed: '+native_error(lib))
  desc=validate_descriptor(json.loads(raw),mode,detail);report['descriptor']=desc;save()
  for slot in desc['slots']:
   host=ctypes.create_string_buffer(slot['nbytes']);rc=lib.glm_attention_diag_read_r433(ctx,slot['key'],ctypes.addressof(host),slot['nbytes'])
   if rc:raise RuntimeError('Native diagnostic copy failed at '+slot['name']+': '+native_error(lib))
   data=host.raw;assert len(data)==slot['nbytes'];path=directory/(prefix+'-'+slot['name']+'.bin')
   with path.open('xb')as f:f.write(data)
   report['files'].append(dict(**slot,path=str(path),SHA256=hashlib.sha256(data).hexdigest()));save()
  report['collection_passed']=True
 except BaseException as error:primary=error;report['error']=repr(error)
 finally:
  try:
   rc=lib.glm_attention_diag_clear_r433(ctx);report['clear_RC']=rc
   if rc:raise RuntimeError('Native diagnostic clear failed: '+native_error(lib))
  except BaseException as error:
   report['clear_error']=repr(error)
   if primary is None:primary=error
  report['completed']=True
  if owned:
   try:save()
   except BaseException as error:
    if primary is None:primary=error
 if primary is not None:raise primary
 return report
