import sys,json,time,socket,ssl,http.client,urllib.parse,hashlib,pathlib,os,fcntl
start=time.monotonic();deadline=start+1160
job=json.load(sys.stdin);url=urllib.parse.urlsplit(job['url']);assert url.scheme=='https' and url.hostname==job['host']
base=pathlib.Path.home()/'hygon-inference-refactor-20261014';folder=base/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769'
part=folder/'model.safetensors.r9004.partial';final=folder/'model.safetensors';receipt=base/'results/DFlash2-CDN-detached-r9018.json'
assert not receipt.exists()
size=7385765464;want='1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2';count=0;h=hashlib.sha256();lock=None
d={'completed':False,'step':'login-PID-'+str(os.getpid()),'PID':os.getpid(),'execution_scope':'login Python network and file I/O only','node':socket.gethostname(),'CPU_only':True,'path':str(part),'expected_size':size,'expected_SHA256':want,'segments':[]}
def emit(x):print(json.dumps(x),flush=True)
emit({'detached':True,'started':True,'step':d['step'],'node':d['node'],'PID':os.getpid(),'scope':'login Python'})
class FixedIP(http.client.HTTPSConnection):
 def __init__(self,host,ip,ctx):super().__init__(host,timeout=8,context=ctx);self.ip=ip
 def connect(self):
  d['phase']='TCP connect';sock=socket.create_connection((self.ip,443),min(8,max(0.1,deadline-time.monotonic())))
  d['phase']='TLS handshake';self.sock=self._context.wrap_socket(sock,server_hostname=self.host)
try:
 d['phase']='prefix admission'
 assert not final.exists() and part.stat().st_size==1500624266
 lock=open(folder/'model.safetensors.r9015.lock','a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with part.open('rb') as f:
  while True:
   b=f.read(4<<20)
   if not b:break
   h.update(b);count+=len(b)
 assert count==1500624266 and h.hexdigest()=='de8f04c3b1b76c6ed55ad1b32145059867a9793d6c71ba7bb40c108cdeaf5099'
 import certifi
 ctx=ssl.create_default_context(cafile=certifi.where());d['CA_bundle']=certifi.where();d['initial_bytes']=count
 # One full-object signed redirect, fixed resolved IP, sequential bounded ranges; no retry.
 with part.open('ab',buffering=0) as f:
  while count<size:
   if time.monotonic()>=deadline:raise TimeoutError('inner deadline')
   offset=count;end=min(size-1,offset+(64<<20)-1);t=time.monotonic();c=None
   try:
    c=FixedIP(job['host'],job['ips'][0],ctx)
    d['phase']='HTTP request';c.request('GET',url.path+('?' +url.query if url.query else ''),headers={'Host':job['host'],'Range':f'bytes={offset}-{end}','Connection':'close'})
    d['phase']='HTTP response headers';res=c.getresponse();d['phase']='Range header validation';expected=f'bytes {offset}-{end}/{size}'
    assert res.status==206 and res.getheader('Content-Range')==expected
    length=res.getheader('Content-Length');assert length is None or int(length)==end-offset+1
    seg=hashlib.sha256();readstart=time.monotonic()
    while count<=end:
     if time.monotonic()>=deadline:raise TimeoutError('inner deadline')
     d['phase']='HTTP body';b=res.read(min(4<<20,end-count+1))
     if not b:raise EOFError('short range')
     d['phase']='partial write/hash';f.write(b);h.update(b);seg.update(b);count+=len(b)
    assert count==end+1
    row={'offset':offset,'bytes':count-offset,'status':206,'content_range':expected,'body_wall_s':time.monotonic()-readstart,'wall_s':time.monotonic()-t,'SHA256':seg.hexdigest()}
    d['segments'].append(row);d['phase']='progress log write';emit({'step':d['step'],'bytes':count,'elapsed_s':time.monotonic()-start,'segment':row})
   finally:
    if c is not None:c.close()
  os.fsync(f.fileno())
 assert count==size and part.stat().st_size==size and h.hexdigest()==want
 part.replace(final);d.update(completed=True,path=str(final))
except BaseException as e:
 d['failure_phase']=d.get('phase','prefix admission');d['error_type']=type(e).__name__
 if isinstance(e,ssl.SSLCertVerificationError):d.update(verify_code=e.verify_code,verify_message=e.verify_message)
finally:
 d.update(bytes=count,prefix_SHA256=h.hexdigest(),actual_file_bytes=final.stat().st_size if final.exists() else part.stat().st_size if part.exists() else None,elapsed_s=time.monotonic()-start)
 if lock is not None:lock.close()
 receipt.write_text(json.dumps(d,indent=2)+'\n');emit({'terminal':d})
