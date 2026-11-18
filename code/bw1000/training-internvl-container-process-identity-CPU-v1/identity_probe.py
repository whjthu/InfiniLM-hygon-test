"""Only held Popen identities and their exact task/children descendants. No RSS."""
import os,stat
from common import need,sha
KEYS=('Uid','Gid','State','TracerPid','NoNewPrivs')
def proc_read(fd,name,cap):
    child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    try:
        data=b''
        while len(data)<=cap:
            v=os.read(child,min(8192,cap+1-len(data)))
            if not v:break
            data+=v
        need(len(data)<=cap,'PROC_FIELD_BOUND');return data
    finally:os.close(child)
def parse_stat(raw):
    v=raw.decode('ascii').rsplit(') ',1)[1].split()
    return dict(PID=int(raw.split(b' ',1)[0]),PPID=int(v[1]),PGID=int(v[2]),state=v[0],startticks=int(v[19]))
def observe(pid,saved,parent,held,expected_uid,persist):
    row=dict(PID=pid,saved_identity=saved,phase='open_proc_directory',accepted=False,status=None,proc_directory=None,error=None)
    fd=None;found=None;children=[]
    try:
        fd=os.open('/proc/'+str(pid),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        st=os.fstat(fd);row['proc_directory']={k:getattr(st,k) for k in ('st_dev','st_ino','st_uid','st_gid','st_mode')}
        row['expected_UID']=expected_uid;row['phase']='relative_stat_before'
        before=parse_stat(proc_read(fd,'stat',8192));row['before']=before
        need(before['PID']==pid,'PROC_PID_DIFFERENT')
        if saved is not None:
            need(before['startticks']==saved['startticks'] and before['PPID']==saved['PPID'] and before['PGID']==saved['PGID'],'SAVED_IDENTITY_CHANGED')
        elif parent is None:
            need(held.returncode is None and pid==held.pid and before['PPID']==os.getpid() and before['PGID']==pid,'HELD_POPEN_IDENTITY_CHANGED')
        else:
            need(before['PPID']==parent['PID'] and before['PGID']==parent['PGID'] and before['startticks']>=parent['startticks'],'NEW_CHILD_IDENTITY_CHANGED')
        # A second relative stat anchors any permitted credential-only status read.
        middle=parse_stat(proc_read(fd,'stat',8192));row['middle']=middle
        need(all(middle[k]==before[k] for k in ('PID','PPID','PGID','startticks')),'IDENTITY_CHANGED_BEFORE_STATUS')
        found={k:before[k] for k in ('PID','PPID','PGID','startticks')}
        row['phase']='bounded_status_identity_only'
        raw=proc_read(fd,'status',65536);fields={}
        for line in raw.splitlines():
            key,sep,value=line.partition(b':')
            if sep and key.decode('ascii',errors='replace') in KEYS:
                name=key.decode();need(name not in fields,'DUPLICATE_STATUS_IDENTITY');fields[name]=value.decode('ascii').strip()
        row['status']=dict(fields=fields,bytes=len(raw),SHA256=sha(raw),RSS_not_parsed=True)
        row['phase']='relative_stat_after';after=parse_stat(proc_read(fd,'stat',8192));row['after']=after
        end=os.fstat(fd);row['proc_directory_after']={k:getattr(end,k) for k in ('st_dev','st_ino','st_uid','st_gid','st_mode')}
        need(all(after[k]==before[k] for k in ('PID','PPID','PGID','startticks')),'IDENTITY_CHANGED_AFTER_STATUS')
        need((st.st_dev,st.st_ino)==(end.st_dev,end.st_ino),'PROC_FD_INODE_CHANGED')
        need(set(fields)==set(KEYS),'STATUS_IDENTITY_FIELDS_MISSING')
        uids=fields['Uid'].split();gids=fields['Gid'].split()
        need(len(uids)==len(gids)==4 and all(x.isdigit() for x in uids+gids),'STATUS_CREDENTIAL_SCHEMA')
        row['UID_values']=list(map(int,uids));row['GID_values']=list(map(int,gids))
        row['phase']='strict_UID_gate'
        need(st.st_uid==end.st_uid==expected_uid and all(x==expected_uid for x in row['UID_values']),'OWNED_UID_TRANSITION_OR_MISMATCH_NO_WAIVER')
        row['phase']='exact_task_children'
        rawchildren=proc_read(fd,'task/'+str(pid)+'/children',8192)
        pieces=rawchildren.split();need(len(pieces)<=32 and all(x.isdigit() for x in pieces),'CHILDREN_BOUND')
        children=list(map(int,pieces));need(len(set(children))==len(children),'CHILDREN_DUPLICATE')
        row['children']=children;row['accepted']=True;row['phase']='COMPLETE'
    except Exception as error:
        row['error']=dict(type=type(error).__name__,errno=getattr(error,'errno',None),reason=str(error)[:256])
    finally:
        if fd is not None:os.close(fd)
        persist(row)
    return row,found,children
