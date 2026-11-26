"""Project the three independent Linux readonly signals for each opened object."""
from hashlib import sha256
import os
from pathlib import Path
import re
import stat


def require(ok,code):
    if not ok:raise ValueError(code)


def hash_text(text):
    return sha256(text.encode()).hexdigest()


def read_proc(path,cap):
    with Path(path).open('rb') as stream:
        raw=stream.read(cap+1)
    require(0<len(raw)<=cap,'MOUNT_PROC_METADATA_EXTENT')
    return raw.decode('utf8','strict')


def choice(value):
    tokens=value.split(',')
    ro=tokens.count('ro')
    rw=tokens.count('rw')
    return dict(ro_occurrences=ro,rw_occurrences=rw,options_SHA256=hash_text(value),
                value='RO' if (ro,rw)==(1,0) else 'RW' if (ro,rw)==(0,1) else 'MALFORMED')


def decode(field):
    found=re.sub(r'\\(040|011|012|134)',lambda m:chr(int(m[1],8)),field)
    require(found.startswith('/'),'MALFORMED_MOUNT_ROOT_OR_POINT')
    return found


def select(raw,mnt_id,path):
    records=[]
    for row in raw.splitlines():
        if row.split(maxsplit=1)[:1]==[str(mnt_id)]:
            records.append(row)
    selected=dict(exact_mount_record_count=len(records),at_exact_point=None,
                  per_mount=None,superblock=None)
    if len(records)!=1:
        selected['metadata_status']='MISSING_MOUNT_ROW' if not records else 'DUPLICATE_MOUNT_ROW'
        return selected
    fields,sep,more=records[0].partition(' - ')
    front,back=fields.split(),more.split()
    if not sep or len(front)<6 or len(back)!=3 or not front[1].isdigit() or not re.fullmatch(r'\d+:\d+',front[2]):
        selected['metadata_status']='MALFORMED_MOUNT_ROW'
        return selected
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.+-]{0,47}',back[0]):
        selected['metadata_status']='MALFORMED_MOUNT_FSTYPE'
        return selected
    try:
        root,point=(decode(front[offset]) for offset in (3,4))
    except ValueError:
        selected['metadata_status']='MALFORMED_MOUNT_PATH'
        return selected
    selected.update(per_mount=choice(front[5]),superblock=choice(back[2]),parent_mount_id=int(front[1]),
                    fstype=back[0],
                    mount_major_minor=front[2],mount_root_SHA256=hash_text(root),mount_point_SHA256=hash_text(point),
                    at_exact_point=point==str(path))
    selected['metadata_status']=('RECORDED' if selected['per_mount']['value']!='MALFORMED'
                                 and selected['superblock']['value']!='MALFORMED' else 'MALFORMED_MOUNT_FLAGS')
    return selected


def error_record(error):
    code=str(error)
    return dict(error_type=type(error).__name__,errno=getattr(error,'errno',None),
                error_code=code if type(error) is ValueError and re.fullmatch('[A-Z0-9_]{1,96}',code) else None)


def observations(fd,path,*,proc=read_proc,stat_fd=None,vfs_fd=None):
    stat_fd=os.fstat if stat_fd is None else stat_fd
    vfs_fd=os.fstatvfs if vfs_fd is None else vfs_fd
    out=dict(present=True,metadata_status='UNKNOWN_MOUNT',path_SHA256=hash_text(str(path)),mount_id=None,
             exact_mount_record_count=None,per_mount=None,superblock=None,fstatvfs=None)
    try:
        info=stat_fd(fd)
        out.update(dev_inode=[info.st_dev,info.st_ino],owner=info.st_uid,permissions=stat.S_IMODE(info.st_mode),
                   object_type='DIRECTORY' if stat.S_ISDIR(info.st_mode) else 'REGULAR')
    except (ValueError,OSError) as error:
        out['descriptor_identity_error']=error_record(error)
    try:
        flags=vfs_fd(fd).f_flag
        out['fstatvfs']=dict(flags=flags,value='RO' if bool(flags&os.ST_RDONLY) else 'RW')
    except (ValueError,OSError) as error:
        out['vfs_error']=error_record(error)
    try:
        text=proc(f'/proc/self/fdinfo/{fd}',16384)
        known=re.findall(r'(?m)^mnt_id:[ \t]*([0-9]+)[ \t]*$',text)
        require(len(known)==1 and int(known[0])>0,'ONE_POSITIVE_FD_MOUNT_ID')
        out['mount_id']=int(known[0])
        raw=proc('/proc/self/mountinfo',4<<20)
        out.update(select(raw,out['mount_id'],path))
    except (ValueError,OSError) as error:
        out['mount_error']=error_record(error)
    reads=tuple((out[key] or {}).get('value','UNKNOWN') for key in ('per_mount','superblock','fstatvfs'))
    out['three_reads']=list(reads)
    if any(item not in ('RO','RW') for item in reads):
        out['reading_relation']='INCOMPLETE_OR_MALFORMED'
    elif len(set(reads))==1:
        out['reading_relation']='UNANIMOUS_'+reads[0]
    elif reads[0]==reads[2]:
        out['reading_relation']='PER_MOUNT_AND_FD_AGREE_SUPERBLOCK_DIFFERS'
    else:
        out['reading_relation']='PER_MOUNT_AND_FD_DISAGREE'
    return out


def observe(path):
    start=dict(path_SHA256=hash_text(str(path)),metadata_status='NOT_OPENED',present=None)
    try:
        before=path.lstat()
        start['present']=True
        require((stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode))
                and path.resolve(strict=True)==path,'PROBE_EXACT_REGULAR_OR_DIRECTORY')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
        try:
            out=observations(fd,path)
            try:
                final=path.lstat()
                out['same_before_open_after_dev_inode']=(out.get('dev_inode')==[before.st_dev,before.st_ino]
                                                        ==[final.st_dev,final.st_ino])
            except OSError as error:
                out['same_before_open_after_dev_inode']=False
                out['last_path_error']=error_record(error)
            return out
        finally:os.close(fd)
    except FileNotFoundError as error:
        return dict(start,present=False,error=error_record(error))
    except (ValueError,OSError) as error:
        return dict(start,error=error_record(error))


def shared_readonly(obs):
    return (obs.get('metadata_status')=='RECORDED' and obs.get('same_before_open_after_dev_inode') is True
            and obs['three_reads'][0]==obs['three_reads'][2]=='RO')

def explicit_readonly(obs):
    return shared_readonly(obs) and obs.get('at_exact_point') is True

def strict_private_writable(obs,owner):
    """Pure three-read predicate only. This observer never offers write authorization."""
    return (obs.get('metadata_status')=='RECORDED' and obs.get('same_before_open_after_dev_inode') is True
            and obs.get('at_exact_point') is True and obs['three_reads']==['RW','RW','RW']
            and obs['owner']==owner and obs['permissions']==0o700)
