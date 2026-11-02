"""Private production entry. No launcher; legacy OFF, source-bound ON."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,math,os,signal,sys,time
from policy import OFF,ON,OLD_WORKER_SHA256,OLD_MANIFEST_SHA256,checked,expected_hello
from runtime_factory import RuntimeFactory
from entry import serve_authenticated

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_old(root):
 directory=Path(root)/'incoming/Kimi-long-draft-memory-floor-r362b'
 assert sha(directory/'worker.py')==OLD_WORKER_SHA256 and sha(directory/'SOURCE_MANIFEST.json')==OLD_MANIFEST_SHA256
 sys.path.insert(0,str(directory))
 spec=importlib.util.spec_from_file_location('strict_worker_r362b_for388',directory/'worker.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 return old,directory

def verify(directory,root,expected):
 directory=Path(directory);root=Path(root)
 assert sha(directory/'SOURCE_MANIFEST.json')==expected
 m=json.loads((directory/'SOURCE_MANIFEST.json').read_bytes())
 assert all(sha(directory/n)==h for n,h in m['files_SHA256'].items())
 assert all(sha(Path(p))==h for p,h in m['dependencies_SHA256'].items())
 old,where=load_old(root);runtime,incoming=old.verify_sources(where,root,OLD_MANIFEST_SHA256)
 return old,runtime,incoming

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--context-policy',choices=[OFF,ON],default=OFF)
 p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();mode=checked(args.context_policy)
 if os.environ.get('INFINILM_KIMI_LONG_RPC_R334')!='1':raise RuntimeError('Long worker default OFF')
 left=args.deadline_unix-time.time()
 if not math.isfinite(left)or not 30<left<=29400 or not 1<=args.port<=65535:raise ValueError('Concrete bounded deadline/port required')
 root=Path('/refactor');old,runtime,incoming=verify(Path(__file__).resolve().parent,root,args.manifest_sha)
 sys.path.insert(0,str(runtime));from batch_wire_r9128 import listen
 hello=expected_hello(old,os.environ['KIMI_DRAFT_NONCE_R9046'],mode,args.manifest_sha)
 if args.output.exists():raise ValueError('Do not overwrite a worker result')
 report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=False,passed=False,actual_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],authenticated=False,ready=False,normal_protocol_close=False,GPU_count=1,source_manifest_SHA256=args.manifest_sha,runtime_manifest_SHA256=old.RUNTIME_SHA,nonce_SHA256=hashlib.sha256(hello['nonce'].encode()).hexdigest(),context_policy_requested=mode,no_retry_reload=True)
 def save():
  t=args.output.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(args.output)
 def expired(*_):raise TimeoutError('Original worker absolute deadline exceeded')
 left=args.deadline_unix-time.time()
 if left<=30:raise TimeoutError('Original budget exhausted during source verification')
 signal.signal(signal.SIGALRM,expired);signal.alarm(int(left))
 factory=RuntimeFactory(old,root,runtime,incoming,report,mode)
 class BoundedFactory:
  def __call__(self):
   if args.deadline_unix-time.time()<=30:raise TimeoutError('Original budget exhausted before checkpoint loading')
   return factory()
  def close_scope(self):factory.close_scope()
 listener=None;save()
 try:
  listener=listen(args.host,args.port);listener.settimeout(min(left,120))
  serve_authenticated(old,listener,hello,BoundedFactory(),report,save,mode)
  report['passed']=report['normal_protocol_close']
 except BaseException as e:report['error']=repr(e);raise
 finally:
  if listener is not None:listener.close()
  report['completed']=True;save()
 return 0
if __name__=='__main__':raise SystemExit(main())
