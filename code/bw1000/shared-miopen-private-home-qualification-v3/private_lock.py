"""One private same-inode two-process flock test. No library or model loading."""
import errno
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import time


CHILD=r"""
import errno,fcntl,os,stat,sys
directory,name,dev,inode=sys.argv[1:]
d=int(directory)
f=os.open(name,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK,dir_fd=d)
try:
    s=os.fstat(f)
    if (s.st_dev,s.st_ino)!=(int(dev),int(inode)) or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
        raise RuntimeError('different file')
    if sys.stdin.buffer.readline(16)!=b'CHECK\n':
        raise RuntimeError('missing first gate')
    try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in (errno.EAGAIN,errno.EWOULDBLOCK,errno.EACCES):raise
        os.write(1,b'BUSY\n')
    else:
        fcntl.flock(f,fcntl.LOCK_UN)
        os.write(1,b'UNEXPECTED\n')
        raise RuntimeError('not locked')
    if sys.stdin.buffer.readline(16)!=b'RETRY\n':
        raise RuntimeError('missing released gate')
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try: os.write(1,b'ACQUIRED\n')
    finally: fcntl.flock(f,fcntl.LOCK_UN)
finally: os.close(f)
"""


def need(ok,code):
    if not ok:raise ValueError(code)


def identity(info):return [info.st_dev,info.st_ino]


def wait_for(child,request,expected,deadline):
    need(time.time()<deadline,'LOCK_TEST_ORIGINAL_BUDGET')
    child.stdin.write(request)
    child.stdin.flush()
    ready,_,_=select.select([child.stdout],[],[],max(0,min(6,deadline-time.time())))
    need(ready and os.read(child.stdout.fileno(),32)==expected,'LOCK_CHILD_RESPONSE_DIFFERS')


def safely_remove(directory,name,expected):
    item=os.stat(name,dir_fd=directory,follow_symlinks=False)
    need(identity(item)==expected and stat.S_ISREG(item.st_mode) and item.st_uid==os.getuid()
         and item.st_nlink==1,'CLEANUP_OWN_ORIGINAL_SENTINEL_ONLY')
    os.unlink(name,dir_fd=directory)
    need(name not in os.listdir(directory),'LOCK_SENTINEL_STILL_PRESENT')


def execute(path,expected_directory,request,deadline,result):
    import fcntl
    name='sentinel-'+request+'.lock'
    root=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    lock=None
    child=None
    owned=None
    try:
        here=os.fstat(root)
        need(identity(here)==expected_directory and here.st_uid==os.getuid() and
             stat.S_IMODE(here.st_mode)==0o700 and not os.listdir(root),'EXACT_EMPTY_OWN_PRIVATE_LOCK_FD')
        need(time.time()+15<deadline,'INSUFFICIENT_TIME_FOR_SCOPED_LOCK_TEST')
        lock=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=root)
        info=os.fstat(lock)
        owned=identity(info)
        result.update(created_exclusively=True,sentinel_dev_inode=owned)
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.getuid()
             and stat.S_IMODE(info.st_mode)==0o600,'EXACT_OWN_SENTINEL')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result['parent_acquired']=True
        child=subprocess.Popen([sys.executable,'-I','-S','-B','-c',CHILD,str(root),name,*map(str,owned)],
                               pass_fds=(root,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL,start_new_session=True)
        result['separate_PID']=child.pid!=os.getpid()
        need(result['separate_PID'],'SEPARATE_LOCK_PARTICIPANT')
        wait_for(child,b'CHECK\n',b'BUSY\n',deadline)
        result['child_blocked_nonblocking']=True
        fcntl.flock(lock,fcntl.LOCK_UN)
        result['parent_released']=True
        wait_for(child,b'RETRY\n',b'ACQUIRED\n',deadline)
        result['same_child_acquired_after_release']=True
        remaining=max(.01,min(5,deadline-time.time()))
        need(child.wait(timeout=remaining)==0,'OWN_LOCK_PARTICIPANT_FAILED')
        result['child_clean_exit']=True
    finally:
        try:
            try:
                if child is not None:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=3)
                    result['child_reaped_for_cleanup']=child.poll() is not None
            finally:
                if child is not None:
                    child.stdin.close()
                    child.stdout.close()
                if lock is not None:
                    os.close(lock)
            need(child is None or result.get('child_reaped_for_cleanup') is True,
                 'UNREAPED_PRIVATE_CHILD_RETAINS_SENTINEL')
            if owned is not None:
                safely_remove(root,name,owned)
                result['sentinel_unlinked_by_identity']=True
            result['lock_dir_empty_after']=not os.listdir(root)
        finally:
            os.close(root)
    need(result.get('lock_dir_empty_after') is True,'PRIVATE_LOCK_DIR_UNEXPECTED_ENTRY')
    return result
