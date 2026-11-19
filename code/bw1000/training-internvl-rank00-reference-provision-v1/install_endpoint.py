"""One published source-only installation into a new role namespace, then exact entry."""
import base64,hashlib,json,math,os,signal,sys,time
from pathlib import Path,PurePosixPath

def need(v,w):
    if not v:raise ValueError(w)

def main():
    epoch=CONTEXT['epoch'];need(math.isfinite(epoch) and 0<epoch<=time.time()<epoch+75 and CONTEXT['deadline']==epoch+900,'ORIGINAL_EARLY900_EPOCH')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+895-time.time()))
    decoded={n:base64.b64decode(b,validate=True) for n,b in FILES.items()};need(sum(map(len,decoded.values()))<=196608,'SMALL_SOURCE_PACKAGE')
    mf=decoded['SOURCE_FILES.json'];need(hashlib.sha256(mf).hexdigest()==CONTEXT['source_manifest_SHA256'],'PUBLISHED_MF');pins=json.loads(mf)['files'];need(set(decoded)==set(pins)|{'SOURCE_FILES.json'},'EXACT_PACKAGE')
    for n,p in pins.items():
        q=PurePosixPath(n);need(not q.is_absolute() and '..' not in q.parts and n==q.as_posix(),'RELATIVE_SOURCE_MEMBER');need(len(decoded[n])==p['bytes'] and hashlib.sha256(decoded[n]).hexdigest()==p['SHA256'],'EXACT_SOURCE_PIN')
    plan=json.loads(decoded['PLAN.json']);claim=json.loads(decoded['CLAIM.json']);need(claim['root_GO'] is claim['provision_GO'] is plan['root_GO'] is plan['provision_GO'] is True and claim['max_invocations']==1 and claim['GPU_GO'] is claim['Model_GO'] is claim['training_GO'] is False,'ROOT_PUBLISHED_TWO_FILE_SCOPE')
    need(ROLE in ('source','receiver'),'EXACT_ROLE');root=Path(plan['source_namespace' if ROLE=='source' else 'target_namespace']);need(root.is_absolute() and root.parent.is_dir() and not root.exists(),'EXCLUSIVE_NEW_NAMESPACE')
    for p in [*reversed(root.parents),root]:need(not p.is_symlink(),'NAMESPACE_SYMLINK')
    os.umask(0o077);root.mkdir(mode=0o700);source=root/'source';source.mkdir()
    for n,b in decoded.items():
        p=source/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    sys.path.insert(0,str(source));from common import save,own
    save(root/'install-intent.json',dict(CONTEXT,role=ROLE,installer=own(),source_members=len(pins)))
    from verify_package import verify
    verify(source)
    if ROLE=='source':
        from endpoint import main as endpoint
        return endpoint('source',CONTEXT,REFERENCE_B64)
    from target_host import main as target
    return target(CONTEXT)

if __name__=='__main__':raise SystemExit(main())
