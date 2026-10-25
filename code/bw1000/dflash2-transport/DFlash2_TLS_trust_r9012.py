"""A single pinned checkpoint CDN redirect diagnosis; signed URL stays in RAM."""
import datetime, hashlib, json, pathlib, shlex, socket, subprocess, time, urllib.parse, urllib.request
r=pathlib.Path(__file__).resolve().parent
out=r/'DFlash2-TLS-trust-r9012.json';assert not out.exists()
rev='e77935fb4804e17eb55085bffd045eae1d779769'
public=f'https://huggingface.co/lightseekorg/kimi-k3-dflash2/resolve/{rev}/model.safetensors'
n=1<<20
d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=rev,
       source_SHA256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
       signed_URL_persisted=False, local_payload_files=False, remote_model_files_touched=False)
remote=r'''import sys,json,time,socket,ssl,http.client,urllib.parse,hashlib,pathlib,importlib.util
job=json.load(sys.stdin);url=urllib.parse.urlsplit(job['url']);assert url.scheme=='https' and url.hostname==job['host']
paths=ssl.get_default_verify_paths();d={'default_verify_paths':paths._asdict(),'bundles':{},'attempts':[]}
for package in ['certifi','requests']:
 try:
  module=__import__(package)
  file=module.where() if package=='certifi' else module.certs.where()
  p=pathlib.Path(file);d['bundles'][package]={'path':file,'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None}
 except Exception as e:d['bundles'][package]={'error_type':type(e).__name__}
if paths.cafile:
 p=pathlib.Path(paths.cafile);d['bundles']['ssl_default']={'path':paths.cafile,'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None}
class FixedIP(http.client.HTTPSConnection):
 def __init__(self,host,ip,context):super().__init__(host,timeout=8,context=context);self.ip=ip
 def connect(self):
  sock=socket.create_connection((self.ip,443),8)
  self.sock=self._context.wrap_socket(sock,server_hostname=self.host)
def attempt(context,label):
 t=time.monotonic();c=None;a={'label':label,'body_bytes':0}
 try:
  c=FixedIP(job['host'],job['ips'][0],context)
  c.request('GET',url.path+('?' +url.query if url.query else ''),headers={'Host':job['host'],'Range':'bytes=0-1048575','Connection':'close'})
  res=c.getresponse();a.update(status=res.status,content_range=res.getheader('Content-Range'),headers_wall_s=time.monotonic()-t)
  start=time.monotonic();h=hashlib.sha256();left=1048576
  while left:
   b=res.read(min(left,65536))
   if not b:break
   a['body_bytes']+=len(b);left-=len(b);h.update(b)
  a.update(body_read_wall_s=time.monotonic()-start,body_SHA256=h.hexdigest())
 except ssl.SSLCertVerificationError as e:a.update(error_type=type(e).__name__,verify_code=e.verify_code,verify_message=e.verify_message)
 except BaseException as e:a['error_type']=type(e).__name__
 finally:
  if c is not None:c.close()
  a['wall_s']=time.monotonic()-t;d['attempts'].append(a)
 return a
first=attempt(ssl.create_default_context(),'existing default trust')
if first.get('error_type')=='SSLCertVerificationError':
 selected=None
 for name in ['certifi','requests']:
  bundle=d['bundles'].get(name,{})
  if bundle.get('exists') and bundle['path']!=paths.cafile:selected=bundle['path'];break
 if selected:attempt(ssl.create_default_context(cafile=selected),'explicit existing trusted bundle '+selected)
 else:d['explicit_bundle_retry']='not performed: no distinct existing certifi/requests bundle'
print(json.dumps(d))
'''
try:
 t=time.monotonic()
 with urllib.request.urlopen(urllib.request.Request(public,headers={'Range':'bytes=0-1048575'}),timeout=30) as response:
  final=response.geturl();u=urllib.parse.urlsplit(final)
  assert u.scheme=='https' and u.hostname and u.hostname!='huggingface.co'
  local_body=response.read(1)
  d['local_redirect']=dict(status=response.status,body_bytes=len(local_body),wall_s=time.monotonic()-t,
                           content_range=response.headers.get('Content-Range'),CDN_hostname=u.hostname)
 ips=[]
 for row in socket.getaddrinfo(u.hostname,443,socket.AF_INET,socket.SOCK_STREAM):
  ip=row[4][0]
  if ip not in ips:ips.append(ip)
  if len(ips)==2:break
 assert ips
 d['resolved_IPv4_count']=len(ips)
 payload=json.dumps(dict(url=final,host=u.hostname,ips=ips)).encode()
 args=['ssh','-T','-p','65032','-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),
       '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
       '-o','UserKnownHostsFile='+str(r/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn',
       'python3 -c '+shlex.quote(remote)]
 p=subprocess.run(args,input=payload,capture_output=True,timeout=75)
 d['SSH_exit']=p.returncode
 if p.returncode==0:d['remote']=json.loads(p.stdout)
 else:d['remote_error']='nonzero SSH exit; stderr intentionally not persisted'
except BaseException as e:d['error_type']=type(e).__name__
out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
print(json.dumps(d))
