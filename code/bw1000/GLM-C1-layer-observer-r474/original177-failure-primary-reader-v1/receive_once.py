"""Install only this small reader in a new private namespace; no original writes."""
import base64,hashlib,json,math,os,resource,signal,sys,time
from pathlib import Path
need=lambda ok:None if ok else (_ for _ in ()).throw(ValueError('READER_BOOTSTRAP_BOUNDARY'))
need(math.isfinite(EPOCH) and 0<EPOCH<=time.time()<EPOCH+175)
affinity=[c for c in sorted(os.sched_getaffinity(0)) if c not in (0,1)][:2];need(len(affinity)==2);os.sched_setaffinity(0,affinity)
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));os.umask(0o077)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('READER_REMOTE180')));signal.setitimer(signal.ITIMER_REAL,max(.001,EPOCH+180-time.time()))
W=Path(NEW_ROOT);need(W==Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/GLM474-original177-failure-primary-reader-v1') and not W.exists());W.mkdir(mode=0o700);S=W/'src';S.mkdir(mode=0o700)
for name,pin in FILES.items():
    need(Path(name).name==name);raw=base64.b64decode(pin['b64'],validate=True);need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256'])
    with (S/name).open('xb') as f:f.write(raw)
need(hashlib.sha256((S/'SOURCE_FILES.json').read_bytes()).hexdigest()==SOURCE_SHA and hashlib.sha256((S/'CLAIM.json').read_bytes()).hexdigest()==CLAIM_SHA)
claim=json.loads((S/'CLAIM.json').read_bytes());need(claim['ROOT_GO'] is True and claim['max_invocations']==1 and claim['primary_files']==35 and not claim['Model_GO'] and not claim['GPU_GO'])
with (W/'intent.json').open('x') as f:json.dump(dict(epoch=EPOCH,deadline=EPOCH+180,request_id=REQUEST,publication=PUBLICATION,source_manifest_SHA256=SOURCE_SHA,claim_SHA256=CLAIM_SHA,CPU_affinity=affinity,RLIMIT_AS_bytes=3*1024**3,AS_is_not_RSS=True),f)
sys.path.insert(0,str(S));sys.argv=ARGS
exec(compile((S/'reader.py').read_bytes(),str(S/'reader.py'),'exec'),{'__name__':'__main__','__file__':str(S/'reader.py')})
