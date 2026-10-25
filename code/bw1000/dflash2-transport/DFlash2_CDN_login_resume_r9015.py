"""One bounded login network/file I/O CDN transfer; signed redirect only in memory."""
import datetime, hashlib, json, pathlib, shlex, socket, subprocess, time, urllib.parse, urllib.request
r=pathlib.Path(__file__).resolve().parent;out=r/'DFlash2-CDN-login-resume-r9015.json';assert not out.exists()
rev='e77935fb4804e17eb55085bffd045eae1d779769'
public=f'https://huggingface.co/lightseekorg/kimi-k3-dflash2/resolve/{rev}/model.safetensors'
d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=rev,source_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),signed_URL_persisted=False,local_payload_files=False,observations=[])
remote=r'''import sys,json,time,socket,ssl,http.client,urllib.parse,hashlib,pathlib,os,fcntl
start=time.monotonic();deadline=start+1160
job=json.load(sys.stdin);url=urllib.parse.urlsplit(job['url']);assert url.scheme=='https' and url.hostname==job['host']
base=pathlib.Path.home()/'hygon-inference-refactor-20261014';folder=base/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769'
part=folder/'model.safetensors.r9004.partial';final=folder/'model.safetensors';receipt=base/'results/DFlash2-CDN-login-resume-r9015.json'
assert not receipt.exists()
size=7385765464;want='1f2a5376315ce5d235303d48d975f78a3dcb19b906e820c1d6447e5fa66e52c2';count=0;h=hashlib.sha256();lock=None
d={'completed':False,'step':'login-PID-'+str(os.getpid()),'PID':os.getpid(),'execution_scope':'login Python network and file I/O only','node':socket.gethostname(),'CPU_only':True,'path':str(part),'expected_size':size,'expected_SHA256':want,'segments':[]}
def emit(x):print(json.dumps(x),flush=True)
emit({'started':True,'step':d['step'],'node':d['node'],'PID':os.getpid(),'scope':'login Python'})
class FixedIP(http.client.HTTPSConnection):
 def __init__(self,host,ip,ctx):super().__init__(host,timeout=8,context=ctx);self.ip=ip
 def connect(self):
  d['phase']='TCP connect';sock=socket.create_connection((self.ip,443),min(8,max(0.1,deadline-time.monotonic())))
  d['phase']='TLS handshake';self.sock=self._context.wrap_socket(sock,server_hostname=self.host)
try:
 d['phase']='prefix admission'
 assert not final.exists() and part.stat().st_size==158446986
 lock=open(folder/'model.safetensors.r9015.lock','a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with part.open('rb') as f:
  while True:
   b=f.read(4<<20)
   if not b:break
   h.update(b);count+=len(b)
 assert count==158446986 and h.hexdigest()=='ce78207a01d46bb50c4bd392f073ce0d2bfd2b2d5326bd534e7a867b4e41fbde'
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
    d['segments'].append(row);emit({'step':d['step'],'bytes':count,'elapsed_s':time.monotonic()-start,'segment':row})
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
'''
p=None
try:
 with urllib.request.urlopen(urllib.request.Request(public,method='HEAD'),timeout=30) as res:
  url=res.geturl();u=urllib.parse.urlsplit(url);assert u.scheme=='https' and u.hostname!='huggingface.co'
  d['redirect']={'status':res.status,'host':u.hostname,'method':'HEAD with no range'}
 ips=[]
 for row in socket.getaddrinfo(u.hostname,443,socket.AF_INET,socket.SOCK_STREAM):
  if row[4][0] not in ips:ips.append(row[4][0])
  if len(ips)==2:break
 assert ips
 command='python3 -c '+shlex.quote(remote)
 args=['ssh','-T','-p','65032','-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile='+str(r/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn',command]
 p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 p.stdin.write(json.dumps({'url':url,'host':u.hostname,'ips':ips}).encode());p.stdin.close();p.stdin=None
 for line in p.stdout:
  item=json.loads(line);d['observations'].append(item);out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(item),flush=True)
 p.wait(timeout=30);d['SSH_exit']=p.returncode
 stderr=p.stderr.read().decode(errors='replace')
 d['SSH_stderr_bytes']=len(stderr.encode())
 classes=[];safe=[]
 for line in stderr.splitlines():
  if any(token in line.lower() for token in ['connection reset','connection closed','connection timed out','connection refused','permission denied','host key verification failed','kex_exchange_identification','banner exchange','ssh_exchange_identification','network is unreachable','no route to host','broken pipe']):
   classes.append(line.split(':',1)[0][:100])
   if 'http' not in line.lower() and '?' not in line and 'url' not in line.lower():safe.append(line[:400])
 d['SSH_error_classes']=classes;d['SSH_diagnostic_lines']=safe
 if stderr and not classes:d['SSH_unclassified_stderr']=True

except BaseException as e:d['error_type']=type(e).__name__
finally:
 out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'local_terminal':True,'SSH_exit':d.get('SSH_exit'),'error_type':d.get('error_type')}),flush=True)
