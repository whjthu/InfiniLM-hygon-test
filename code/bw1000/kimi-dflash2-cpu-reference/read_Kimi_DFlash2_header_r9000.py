import pathlib,urllib.request,json,hashlib,struct
r=pathlib.Path.cwd()/'Kimi-DFlash2-CPU-private-r9000';r.mkdir(exist_ok=True);url='https://huggingface.co/lightseekorg/kimi-k3-dflash2/resolve/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors';calls=[]
def part(begin,end):
 req=urllib.request.Request(url,headers={'Range':f'bytes={begin}-{end}','Accept-Encoding':'identity'})
 with urllib.request.urlopen(req,timeout=35)as response:
  info={'status':response.status,'content_range':response.headers.get('Content-Range'),'content_length':response.headers.get('Content-Length'),'requested':[begin,end],'URL':response.url};calls.append(info)
  # Never read a body if the server ignores the requested byte range.
  assert response.status==206,'STOP_SERVER_IGNORED_RANGE_NO_BODY_READ'
  assert response.headers.get('Content-Range','').startswith(f'bytes {begin}-{end}/'),'STOP_WRONG_RANGE_NO_BODY_READ'
  assert int(response.headers.get('Content-Length',str(end-begin+1)))==end-begin+1
  raw=response.read(end-begin+2);assert len(raw)==end-begin+1
  return raw
d={'CPU_only':True,'weight_payload_bytes':0,'calls':calls,'completed':False}
try:
 first=part(0,7);n=struct.unpack('<Q',first)[0];assert 0<n<=1024**2-8,('STOP_HEADER_TOO_LARGE',n)
 raw=part(8,7+n);header=json.loads(raw);assert isinstance(header,dict)
 (r/'actual-safetensors-header.json').write_bytes(raw);d.update(completed=True,header_bytes=n,downloaded_bytes=8+n,header_SHA256=hashlib.sha256(raw).hexdigest(),tensors={k:v for k,v in header.items()if k!='__metadata__'},parameter_count=sum(__import__('math').prod(v['shape'])for k,v in header.items()if k!='__metadata__'))
except BaseException as e:d['error']=repr(e)
(r/'header-receipt.json').write_bytes((json.dumps(d,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in d.items()if k!='tensors'},indent=2))
