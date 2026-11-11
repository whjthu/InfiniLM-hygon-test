"""Exercise real POSIX assembly and verify failures cannot alter retained bytes."""
import hashlib,json,tempfile
from pathlib import Path
from assemble_tail import assemble
def sha(raw):return hashlib.sha256(raw).hexdigest()
normal=0;rejected=0
with tempfile.TemporaryDirectory() as work:
    root=Path(work)
    for i,kind in enumerate(('normal','corrupt-prefix','corrupt-tail','short-tail','wrong-total','changed-stat','existing-final')):
        folder=root/str(i);folder.mkdir();p=folder/'partial';t=folder/'tail';f=folder/'final'
        prefix=b'fixed-prefix'*256;tail=b'actual-tail'*19;p.write_bytes(prefix);t.write_bytes(tail)
        stat=p.stat();expected={'inode':stat.st_ino,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        total=len(prefix+tail);full_sha=sha(prefix+tail);tail_sha=sha(tail);journal=[]
        if kind=='corrupt-prefix':full_sha=sha(b'x'+prefix[1:]+tail)
        if kind=='corrupt-tail':tail_sha='0'*64
        if kind=='short-tail':total+=1
        if kind=='wrong-total':total-=1
        if kind=='changed-stat':expected['mtime_ns']-=1
        if kind=='existing-final':f.write_bytes(b'unrelated')
        try:result=assemble(p,t,f,total,full_sha,tail_sha,expected,journal.append)
        except ValueError:
            assert kind!='normal' and p.read_bytes()==prefix and not journal
            rejected+=1
        else:
            assert kind=='normal' and not p.exists() and f.read_bytes()==prefix+tail
            assert result['SHA256']==full_sha and len(journal)==1 and journal[0]['state']=='COMPOSITE_FULL_DIGEST_VERIFIED_BEFORE_APPEND'
            normal+=1
print(json.dumps({'state':'POSIX_COMPOSITE_HASH_BEFORE_APPEND_CPU_PASS','normal':normal,'rejected_unchanged_partials':rejected,'checkpoint_loaded':False,'GPU_execution':False}))
