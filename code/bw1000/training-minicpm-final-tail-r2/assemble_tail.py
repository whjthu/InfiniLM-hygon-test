"""Verify retained bytes plus a small frozen tail before changing the retained file."""
import hashlib,os

def assemble(partial,tail,final,total,full_sha,tail_sha,expected,journal):
    if partial.is_symlink() or tail.is_symlink() or final.exists():raise ValueError('unsafe or existing assembly path')
    before=partial.stat();tail_data=tail.read_bytes()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(expected['inode'],expected['bytes'],expected['mtime_ns']):raise ValueError('retained partial changed')
    if not tail_data or before.st_size+len(tail_data)!=total or hashlib.sha256(tail_data).hexdigest()!=tail_sha:raise ValueError('tail size or identity differs')
    with partial.open('r+b') as handle:
        import fcntl
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=os.fstat(handle.fileno())
        def same(a,b):return (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)
        if not same(before,current):raise ValueError('partial changed before lock')
        digest=hashlib.sha256()
        for data in iter(lambda:handle.read(1024*1024),b''):digest.update(data)
        digest.update(tail_data)
        if digest.hexdigest()!=full_sha:raise ValueError('complete composite digest mismatch; partial unchanged')
        if not same(before,os.fstat(handle.fileno())) or not same(before,partial.stat()):raise ValueError('partial changed during digest')
        journal({'state':'COMPOSITE_FULL_DIGEST_VERIFIED_BEFORE_APPEND','partial_bytes':before.st_size,'tail_bytes':len(tail_data),'full_SHA256':full_sha,'tail_SHA256':tail_sha})
        handle.seek(0,os.SEEK_END);handle.write(tail_data);handle.flush();os.fsync(handle.fileno())
        if os.fstat(handle.fileno()).st_size!=total:raise ValueError('final size differs')
        os.link(partial,final);partial.unlink()
    directory=os.open(final.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return {'state':'RETAINED_BYTES_AND_TAIL_ASSEMBLED_FULL_SHA_VERIFIED','bytes':total,'retained_bytes':before.st_size,'tail_bytes':len(tail_data),'SHA256':full_sha,'tail_SHA256':tail_sha,'GPU_execution':False,'model_loaded':False}
