"""Bounded source stdin bootstrap; first receiver alone creates atomic shared context root."""
import base64,hashlib,json,math,os,signal,sys,time
from pathlib import Path,PurePosixPath

def need(v,w):
    if not v:raise ValueError(w)

def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def syncdir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def main():
    epoch=CONTEXT['epoch'];need(math.isfinite(epoch) and 0<epoch<=time.time()<epoch+180 and CONTEXT['deadline']==epoch+86400,'ORIGINAL_EARLY86400_EPOCH')
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,epoch+86370-time.time()))
    decoded={n:base64.b64decode(b,validate=True) for n,b in FILES.items()};need(sum(map(len,decoded.values()))<=196608,'SMALL_DECODED_SOURCE_BOUND')
    mf=decoded['SOURCE_FILES.json'];need(hashlib.sha256(mf).hexdigest()==CONTEXT['source_manifest_SHA256'],'PUBLISHED_MF');pins=json.loads(mf)['files'];need(set(decoded)==set(pins)|{'SOURCE_FILES.json'},'EXACT_SOURCE_MEMBERS')
    for n,p in pins.items():
        q=PurePosixPath(n);need(not q.is_absolute() and '..' not in q.parts and n==q.as_posix(),'RELATIVE_SOURCE_MEMBER');need(len(decoded[n])==p['bytes'] and hashlib.sha256(decoded[n]).hexdigest()==p['SHA256'],'EXACT_SOURCE_PIN')
    plan=json.loads(decoded['PLAN.json']);claim=json.loads(decoded['CLAIM.json'])
    need(all(plan[k] is claim[k] is True for k in ['root_GO','provision_GO','READ_GO']) and all(plan[k] is claim[k] is False for k in ['assembly_GO','GPU_GO','Model_GO','training_GO']) and claim['max_invocations']==1,'PUBLISHED_ONCE_CPU_THREE_SHARDS_ONLY')
    rows=[r for r in plan['files'] if r['file_id']==CONTEXT['file_id']];need(len(rows)==1 and ROLE in ('source','receiver'),'EXACT_FILE_ROLE');row=rows[0]
    need(os.getuid()==(1001 if ROLE=='source' else 154459),'EXPECTED_INSTALLER_UID');os.umask(0o077)
    root=Path(row['source_namespace' if ROLE=='source' else 'target_namespace'])
    for q in [*reversed(root.parents),root]:need(not q.is_symlink(),'NAMESPACE_SYMLINK')
    if ROLE=='receiver':
        common=Path(plan['common_root']);marker=dict(schema=plan['schema'],label=plan['label'],context={k:CONTEXT[k] for k in ['publication','request_id','epoch','deadline','source_manifest_SHA256','namespace_nonce']},first_receiver_file_id=2,common_root=plan['common_root'],modelroot=plan['modelroot']);raw=canonical(marker)
        if row['file_id']==2:
            need(not common.exists() and common.parent.is_dir(),'FIRST_ONLY_NEW_COMMON_ROOT');common.mkdir(mode=0o700);(common/'streams').mkdir();Path(plan['modelroot']).mkdir()
            tmp=common/'CONTEXT.pending';final=common/'CONTEXT.json'
            with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            need(not final.exists(),'NO_MARKER_REPLACE');tmp.rename(final);syncdir(common);syncdir(common.parent)
        else:
            # No mkdir/repair/recreate for the common root or marker in later receivers.
            final=common/'CONTEXT.json';need(common.is_dir() and final.is_file() and not final.is_symlink() and final.stat().st_size<=16384 and final.read_bytes()==raw,'LATER_RECEIVER_REQUIRES_ORIGINAL_MARKER')
        need(root.parent==common/'streams' and root.parent.is_dir(),'NEW_PER_ARM_CONTROL_PARENT')
    need(root.is_absolute() and root.parent.is_dir() and not root.exists(),'EXCLUSIVE_NEW_FILE_NAMESPACE');root.mkdir(mode=0o700);source=root/'source';source.mkdir()
    for n,b in decoded.items():
        p=source/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    syncdir(source);syncdir(root);sys.path.insert(0,str(source))
    from common import save,own,verify_marker
    save(root/'install-intent.json',dict(CONTEXT,role=ROLE,installer=own(),source_members=len(pins),common_marker=verify_marker(CONTEXT) if ROLE=='receiver' else None))
    from verify_package import verify
    verify(source)
    if ROLE=='source':
        from endpoint import main as endpoint
        return endpoint('source',CONTEXT)
    from target_host import main as target
    return target(CONTEXT)

if __name__=='__main__':raise SystemExit(main())
