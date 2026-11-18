"""Same-filesystem no-replace installation; only our tracked partial may be removed."""
import os
from pathlib import Path
import stat
from reuse import require,relative,no_links,read_pinned,stable,sha,sync_directory,tick

def bare(name):
    path=relative(name)
    require(len(path.parts)==1,'ONE_BARE_MEMBER_REQUIRED')
    return path.name

def existing(root,name,pin,maximum):
    path=no_links(Path(root)/bare(name))
    try:info=path.lstat()
    except FileNotFoundError:return None
    require(stat.S_ISREG(info.st_mode) and info.st_size==pin['bytes'],'EXISTING_DIFFERENT_SIZE_OR_TYPE')
    raw,stamp=read_pinned(path,pin,maximum)
    return dict(bytes=len(raw),SHA256=sha(raw),file_identity=stamp,state='EXISTING_CORRECT_REUSED')

def install(root,name,raw,pin,request,maximum):
    tick();root=no_links(Path(root));name=bare(name)
    require(root.is_dir() and root.stat().st_uid==os.getuid(),'OWN_TARGET_DIRECTORY')
    require(0<len(raw)<=maximum and len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'COMPLETE_MEMBER_PIN')
    require(len(request)==32 and all(c in '0123456789abcdef' for c in request),'EXACT_NEW_REQUEST')
    final=root/name;partial=root/('.'+name+'.'+request+'.partial')
    fd=os.open(partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    owned=os.fstat(fd);linked=False;partial_removed=False
    def remove_own_partial():
        current=partial.lstat()
        require(stat.S_ISREG(current.st_mode) and current.st_dev==owned.st_dev and current.st_ino==owned.st_ino,'DO_NOT_REMOVE_FOREIGN_PARTIAL')
        partial.unlink();sync_directory(root)
    try:
        with os.fdopen(fd,'wb',buffering=0) as stream:
            require(owned.st_dev==root.stat().st_dev,'SAME_FILESYSTEM_PARTIAL')
            offset=0
            while offset<len(raw):
                tick();count=stream.write(raw[offset:offset+65536]);require(count and count>0,'PARTIAL_SHORT_WRITE');offset+=count
            os.fsync(stream.fileno());after=os.fstat(stream.fileno())
            require(after.st_ino==owned.st_ino and after.st_dev==owned.st_dev and after.st_size==len(raw),'OWN_PARTIAL_CHANGED')
        tick()
        try:
            os.link(partial,final,follow_symlinks=False);linked=True
        except FileExistsError:
            # A concurrent name never authorizes replacement, even if our data is correct.
            pass
        # Dropping our extra hardlink changes the shared inode's ctime. Finish
        # that operation before the one final read/hash records its stable stamp.
        remove_own_partial();partial_removed=True
        found=existing(root,name,pin,maximum)
        require(found is not None,'NEW_FINAL_MISSING')
        return dict(found,state='NEW_ATOMIC_NOREPLACE' if linked else 'CONCURRENT_CORRECT_REUSED',new_file_linked=linked)
    finally:
        if not partial_removed:
            try:remove_own_partial()
            except FileNotFoundError:pass
