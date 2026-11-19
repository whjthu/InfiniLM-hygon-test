"""Only a root-published new CPU scope with actual fixed-original3PASS can install/run."""
import base64,hashlib,json,math,os,signal,sys,time
from pathlib import Path,PurePosixPath

def need(v,w):
    if not v:raise ValueError(w)

def main():
    epoch=CONTEXT['epoch'];need(math.isfinite(epoch) and 0<epoch<=time.time()<epoch+90 and CONTEXT['deadline']==epoch+900,'ORIGINAL900_INSTALL_DEADLINE');signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+865-time.time()))
    decoded={n:base64.b64decode(b,validate=True) for n,b in FILES.items()};need(sum(map(len,decoded.values()))<=196608,'SOURCE_BUNDLE_BOUND');mf=decoded['SOURCE_FILES.json'];need(hashlib.sha256(mf).hexdigest()==CONTEXT['source_manifest_SHA256'],'EXACT_PUBLISHED_MF');pins=json.loads(mf)['files'];need(set(decoded)==set(pins)|{'SOURCE_FILES.json'},'EXACT_SOURCE_MEMBERS')
    for n,p in pins.items():
        q=PurePosixPath(n);need(not q.is_absolute() and '..' not in q.parts and q.as_posix()==n,'SAFE_SOURCE_MEMBER');need(len(decoded[n])==p['bytes'] and hashlib.sha256(decoded[n]).hexdigest()==p['SHA256'],'EXACT_SOURCE_PIN')
    plan=json.loads(decoded['PLAN.json']);claim=json.loads(decoded['CLAIM.json']);need(all(plan[k] is claim[k] is True for k in ['root_GO','CPU_GO','assembly_GO','READ_GO']) and all(plan[k] is claim[k] is False for k in ['GPU_GO','Model_GO','training_GO']),'EXACT_NEW_CPU_ADOPTION')
    # Validate immutable original identity and PASS snapshot before creating even this new namespace.
    dep=json.loads(decoded['DEPENDENCIES.json']);need(SNAPSHOT['original_context']==dep['original_context'] and SNAPSHOT['original_passed'] is SNAPSHOT['original_independent_passed'] is True and SNAPSHOT['verified_files']==3 and SNAPSHOT['verified_raw_bytes']==12150388664,'FIXED_ORIGINAL_PASS_ONLY_NO_SUBSTITUTION')
    raw=(json.dumps(SNAPSHOT,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode();need(len(raw)<=32768 and hashlib.sha256(raw).hexdigest()==CONTEXT['dependency_snapshot_SHA256'],'EXACT_BOUNDED_PASS_SNAPSHOT')
    root=Path(plan['namespace']);need(os.getuid()==154459 and root.is_absolute() and root.parent.is_dir() and not root.exists(),'EXCLUSIVE_NEW_CPU_NAMESPACE')
    for q in [*reversed(root.parents),root]:need(not q.is_symlink(),'NO_NAMESPACE_SYMLINK')
    os.umask(0o077);root.mkdir(mode=0o700);source=root/'source';source.mkdir()
    for n,b in decoded.items():
        p=source/n
        with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    sys.path.insert(0,str(source))
    from common import save,own
    from dependency_gate import validate_snapshot
    validate_snapshot(SNAPSHOT);save(root/'upstream-snapshot.json',SNAPSHOT);save(root/'install-intent.json',dict(CONTEXT,installer=own()))
    from verify_package import verify
    verify(source)
    from host_launch import main as host
    return host(CONTEXT,SNAPSHOT)

if __name__=='__main__':raise SystemExit(main())
