"""One detached CDN resume. Signed redirect only in RAM and child stdin PIPE."""
import datetime,hashlib,json,pathlib,shlex,socket,subprocess,urllib.parse,urllib.request
r=pathlib.Path(__file__).resolve().parent;out=r/'DFlash2-detached-launch-r9018.json';assert not out.exists()
source=(r/'DFlash2_CDN_detached_worker_r9018.py').read_text(encoding='utf8')
public='https://huggingface.co/lightseekorg/kimi-k3-dflash2/resolve/e77935fb4804e17eb55085bffd045eae1d779769/model.safetensors'
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_SHA256':hashlib.sha256(source.encode()).hexdigest(),'signed_URL_persisted':False}
outer=r'''import json,os,pathlib,subprocess,sys,time,hashlib,fcntl
payload=json.load(sys.stdin);base=pathlib.Path.home()/'hygon-inference-refactor-20261014'
old=json.loads((base/'results/DFlash2-CDN-login-resume-r9015.json').read_text())
assert old['completed']==False and old['error_type']=='BrokenPipeError' and old['bytes']==1500624266
assert not pathlib.Path('/proc/3706777').exists()
folder=base/'models/kimi-k3-dflash2/e77935fb4804e17eb55085bffd045eae1d779769'
assert (folder/'model.safetensors.r9004.partial').stat().st_size==1500624266 and not (folder/'model.safetensors').exists()
receipt=base/'results/DFlash2-CDN-detached-r9018.json';log=base/'results/DFlash2-CDN-detached-r9018.log'
assert not receipt.exists() and not log.exists()
assert hashlib.sha256(payload['source'].encode()).hexdigest()==payload['source_SHA256']
# Verify no existing writer before launch; worker independently retains same lock.
with (folder/'model.safetensors.r9015.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
with log.open('xb',buffering=0) as stream:
 p=subprocess.Popen([sys.executable,'-c',payload['source']],stdin=subprocess.PIPE,stdout=stream,stderr=stream,start_new_session=True,close_fds=True)
 p.stdin.write(json.dumps(payload['job']).encode());p.stdin.close()
 time.sleep(0.5)
 print(json.dumps({'launched_PID':p.pid,'alive':p.poll() is None,'node':__import__('socket').gethostname(),'progress_log':str(log),'receipt':str(receipt),'detached_session':True,'initial_progress_lines':log.read_text().splitlines()[:1]}),flush=True)
'''
try:
 with urllib.request.urlopen(urllib.request.Request(public,method='HEAD'),timeout=30) as res:
  final=res.geturl();u=urllib.parse.urlsplit(final);assert u.scheme=='https' and u.hostname!='huggingface.co'
 ips=[]
 for row in socket.getaddrinfo(u.hostname,443,socket.AF_INET,socket.SOCK_STREAM):
  if row[4][0] not in ips:ips.append(row[4][0])
  if len(ips)==2:break
 assert ips
 payload=json.dumps({'source':source,'source_SHA256':d['worker_SHA256'],'job':{'url':final,'host':u.hostname,'ips':ips}}).encode()
 args=['ssh','-T','-p','65032','-i',str(pathlib.Path.home()/'.ssh/hygon-261012.txt'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','UserKnownHostsFile='+str(r/'known_hosts'),'aclsylqidf@zzeshell.scnet.cn','python3 -c '+shlex.quote(outer)]
 p=subprocess.run(args,input=payload,capture_output=True,timeout=60);d['SSH_exit']=p.returncode
 if p.returncode==0:d['launch']=json.loads(p.stdout)
 else:
  d['stderr_bytes']=len(p.stderr)
  # Outer source contains no URL literal or credential value; do not persist arbitrary stderr.
  safe=[line for line in p.stderr.decode(errors='replace').splitlines() if any(t in line.lower() for t in ['connection reset','connection closed','permission denied','timed out','assertionerror','blockingioerror']) and not any(t in line.lower() for t in ['http','?','url'])]
  d['SSH_classified_errors']=safe
except Exception as e:d['error_type']=type(e).__name__
out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8');print(json.dumps(d))
