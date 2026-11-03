"""Two finite stdio children; pipe-reader threads perform no Runtime calls."""
from pathlib import Path
import json,os,queue,subprocess,threading,time

MAX_FRAME=262144

def plan():
 out=[dict(seq=0,role='baseline',op='check',width=0,warm=True,index=0),dict(seq=1,role='candidate',op='check',width=0,warm=True,index=1)]
 for width in [96,6]:
  for i,mode in enumerate(['OFF','ON']+['OFF','ON','ON','OFF']*4):out.append(dict(seq=len(out),role='candidate'if mode=='ON'else'baseline',op='bench',width=width,warm=i<2,index=i))
 assert len(out)==38 and all(sum(v['role']==r for v in out)==19 for r in ['baseline','candidate'])
 return out

def command(item,deadline):
 return {k:item[k]for k in ['seq','role','op','width']}|dict(protocol=1,deadline_unix=deadline)

class Peer:
 def __init__(self,exe,role,directory,deadline,env,popen=None):
  self.role=role;self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
  self.stderr_path=self.directory/'pipe.stderr';self.stdout_path=self.directory/'pipe.stdout'
  self.stderr=self.stderr_path.open('xb');self.stdout=self.stdout_path.open('xb');self.events=queue.Queue(maxsize=4);self.stop_reading=threading.Event();self.outstanding=False;self.owner=threading.get_ident();self.deadline=deadline;self.started_unix=time.time();self.exchanges=0
  self.argv=[str(exe),role,str(self.directory),str(deadline)];childenv=dict(env);childenv['INFINILM_CPU_ROW_COPY_R436']='1'if role=='candidate'else'0'
  self.p=(popen or subprocess.Popen)(self.argv,env=childenv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0)
  self.reader=threading.Thread(target=self._read,daemon=True,name='r439-'+role+'-stdio');self.reader.start()
 def _put(self,item):
  while not self.stop_reading.is_set():
   try:self.events.put(item,timeout=.1);return
   except queue.Full:continue
 def _read(self):
  try:
   while not self.stop_reading.is_set():
    b=self.p.stdout.readline(MAX_FRAME+1)
    if not b:self._put(('EOF',None));return
    self.stdout.write(b);self.stdout.flush()
    if len(b)>MAX_FRAME or not b.endswith(b'\n'):self._put(('error','stdout frame exceeded bound or missing newline'));return
    self._put(('line',b))
  except BaseException as e:self._put(('error',repr(e)))
 def receive(self,limit,on_wait=None):
  assert threading.get_ident()==self.owner
  until=min(limit,self.deadline)
  while True:
   if on_wait:on_wait(self)
   left=until-time.time()
   if left<=0:raise TimeoutError('Original fixed child response deadline')
   try:kind,b=self.events.get(timeout=min(.5,left))
   except queue.Empty:continue
   if kind=='EOF':raise EOFError('Fixed child stdout EOF, rc='+str(self.p.poll()))
   if kind=='error':raise RuntimeError(b)
   self.outstanding=False;self.exchanges+=1
   return json.loads(b.decode('utf8'))
 def send(self,item):
  assert threading.get_ident()==self.owner and not self.outstanding and self.p.poll()is None
  assert item['role']==self.role and item['protocol']==1 and item['deadline_unix']==self.deadline
  b=(json.dumps(item,separators=(',',':'),allow_nan=False)+'\n').encode();assert len(b)<=4096
  assert self.p.stdin.write(b)==len(b),'Partial command write';self.p.stdin.flush();self.outstanding=True
 def close_input(self):
  if self.p.stdin and not self.p.stdin.closed:
   try:self.p.stdin.close()
   except(BrokenPipeError,OSError):pass
 def finish(self,until):
  self.close_input();forced=False
  try:self.p.wait(timeout=max(.001,until-time.time()))
  except subprocess.TimeoutExpired:
   forced=True;self.p.kill()
   try:self.p.wait(timeout=2)
   except subprocess.TimeoutExpired:pass
  self.stop_reading.set();self.reader.join(timeout=2)
  if not self.reader.is_alive():self.stdout.close();self.p.stdout.close()
  self.stderr.close()
  return dict(PID=self.p.pid,rc=self.p.poll(),forced_kill=forced,reader_stopped=not self.reader.is_alive(),process_residency_s=time.time()-self.started_unix,stdout=str(self.stdout_path),stderr=str(self.stderr_path))

def startup_observer(peer,elapsed):
 """Bounded owned-PID metadata; no file contents or device API."""
 proc=Path('/proc')/str(peer.p.pid);v=dict(PID=peer.p.pid,elapsed_since_spawn_s=elapsed)
 for name,limit in [('stat',8192),('io',8192),('wchan',256)]:
  try:
   with(proc/name).open('rb')as f:v[name]=f.read(limit).decode(errors='replace')
  except OSError as e:v[name+'_error']=repr(e)
 try:
  with(proc/'maps').open('rb')as f:b=f.read(2000000);v['maps_truncated']=bool(f.read(1))
  v['maps_paths']=sorted({s.split(None,5)[5]for s in b.decode(errors='replace').splitlines()if len(s.split(None,5))==6 and s.split(None,5)[5].startswith('/')})
 except OSError as e:v['maps_error']=repr(e)
 v['fd_paths']=[]
 try:
  for p in sorted((proc/'fd').iterdir(),key=lambda p:int(p.name))[:64]:
   try:
    item=dict(fd=int(p.name),target=os.readlink(p))
    with(proc/'fdinfo'/p.name).open('rb')as f:item['fdinfo']=f.read(1024).decode(errors='replace')
    v['fd_paths'].append(item)
   except OSError as e:v['fd_paths'].append(dict(fd=p.name,error=repr(e)))
 except OSError as e:v['fd_error']=repr(e)
 return v
