"""Verify the preserved prefix before appending exactly the missing original bytes."""
import hashlib,os


def append_verified_suffix(part,source,cfg,bound):
    h=hashlib.sha256();n=0;stat=part.stat()
    assert stat.st_size==cfg['prefix_bytes']
    with part.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            bound();h.update(b);n+=len(b)
    assert n==cfg['prefix_bytes'] and h.hexdigest()==cfg['prefix_SHA256']
    assert cfg['remaining_bytes']==cfg['archive_bytes']-n and cfg['remaining_bytes']>0
    assert (part.stat().st_dev,part.stat().st_ino,part.stat().st_size)==(stat.st_dev,stat.st_ino,n)
    remaining=cfg['remaining_bytes']
    with part.open('ab') as f:
        opened=os.fstat(f.fileno())
        assert (opened.st_dev,opened.st_ino,opened.st_size)==(stat.st_dev,stat.st_ino,n)
        while remaining:
            bound();b=source.read(min(1024**2,remaining));assert b,'incomplete missing suffix'
            f.write(b);h.update(b);remaining-=len(b)
        assert not source.read(1),'extra suffix bytes'
        bound();f.flush();os.fsync(f.fileno())
    assert part.stat().st_size==cfg['archive_bytes'] and h.hexdigest()==cfg['archive_SHA256']
