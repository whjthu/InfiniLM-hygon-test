"""Held-child anchored tree, including the actual359 privileged UID pattern.

No global /proc scan, executable-name authority, privileged RSS, or new signals.
"""
import os
from common import need, sha
from step_memory import relative

IDENTITY=('PID','PPID','PGID','startticks')

def parse_stat(raw):
    tail=raw.decode('ascii').rsplit(') ',1)[1].split()
    return dict(PID=int(raw.split(b' ',1)[0]),PPID=int(tail[1]),PGID=int(tail[2]),
                startticks=int(tail[19]),state=tail[0])

def observe(pid,saved,parent,held,memory,persist):
    row=dict(PID=pid,phase='proc_open',accepted=False,RSS_bytes=None,RSS_coverage='UNKNOWN_NOT_READ')
    fd=None;found=None;children=[]
    try:
        memory.check_time()
        fd=os.open('/proc/'+str(pid),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        first=os.fstat(fd);before=parse_stat(relative(fd,'stat',8192))
        row.update(before=before,proc_directory=dict(dev=first.st_dev,ino=first.st_ino,UID=first.st_uid))
        need(before['PID']==pid,'TREE_PROC_PID')
        if saved is not None:
            need(all(before[k]==saved[k] for k in IDENTITY),'TREE_SAVED_IDENTITY_CHANGED')
        elif parent is None:
            need(held.returncode is None and pid==held.pid and before['PPID']==os.getpid()
                 and before['PGID']==pid,'HELD_ROOT_IDENTITY')
        else:
            need(before['PPID']==parent['PID'] and before['PGID']==parent['PGID']
                 and before['startticks']>=parent['startticks'],'EXACT_CHILD_PARENT_IDENTITY')
        found={k:before[k] for k in IDENTITY}
        middle=parse_stat(relative(fd,'stat',8192));row['middle']=middle
        need(all(middle[k]==before[k] for k in IDENTITY),'TREE_IDENTITY_BEFORE_STATUS')
        row['phase']='identity_status_and_memory_membership'
        cg_before=relative(fd,'cgroup',16384);path=memory.check_member(cg_before)
        raw=relative(fd,'status',65536);fields={}
        for line in raw.splitlines():
            key,sep,value=line.partition(b':')
            if key in (b'Uid',b'Gid',b'State',b'TracerPid',b'NoNewPrivs'):
                need(sep and key.decode() not in fields,'IDENTITY_STATUS_DUPLICATE')
                fields[key.decode()]=value.decode('ascii').strip()
        need(set(fields)=={'Uid','Gid','State','TracerPid','NoNewPrivs'},'IDENTITY_STATUS_FIELDS')
        uids=fields['Uid'].split();gids=fields['Gid'].split()
        need(len(uids)==len(gids)==4 and all(v.isdigit() for v in uids+gids),'IDENTITY_UID_GID_SCHEMA')
        uids=list(map(int,uids));uid=os.getuid()
        ordinary=first.st_uid==uid and uids==[uid]*4
        privileged=first.st_uid==0 and uids==[uid,0,0,uid]
        row.update(status_fields=fields,status_SHA256=sha(raw),UID_values=uids,
                   memory_membership=path,memory_membership_SHA256=sha(cg_before),
                   UID_classification='ORDINARY_OWNED' if ordinary else 'ANCHORED_359_PRIVILEGED_INTERMEDIARY' if privileged else 'REJECTED')
        need(ordinary or privileged,'UNREVIEWED_OWNED_UID_PATTERN')
        row['phase']='exact_task_children'
        rawchildren=relative(fd,'task/'+str(pid)+'/children',8192);parts=rawchildren.split()
        need(len(parts)<=64 and all(p.isdigit() and int(p)>0 for p in parts),'EXACT_CHILDREN_BOUND')
        children=list(map(int,parts));need(len(set(children))==len(children),'DUPLICATE_CHILD_PID')
        after=parse_stat(relative(fd,'stat',8192));last=os.fstat(fd)
        cg_after=relative(fd,'cgroup',16384)
        row.update(after=after,children=children,proc_directory_after=dict(dev=last.st_dev,ino=last.st_ino,UID=last.st_uid))
        need(all(after[k]==before[k] for k in IDENTITY),'TREE_IDENTITY_AFTER')
        need((first.st_dev,first.st_ino,first.st_uid)==(last.st_dev,last.st_ino,last.st_uid),'TREE_PROC_FD_CHANGED')
        need(cg_before==cg_after,'PROCESS_MEMORY_MEMBERSHIP_CHANGED')
        row.update(accepted=True,phase='COMPLETE')
    except (FileNotFoundError,ProcessLookupError) as e:
        row.update(vanished_during_observation=True,original_PID_termination_proven=False)
        row['error']=dict(type=type(e).__name__,errno=getattr(e,'errno',None),reason='PROCESS_DISAPPEARED_DURING_EXACT_READ')
    except BaseException as e:
        row['error']=dict(type=type(e).__name__,errno=getattr(e,'errno',None),reason=str(e)[:256])
    finally:
        if fd is not None:os.close(fd)
        persist(row)
    return row,found,children

class OwnedTree:
    def __init__(self,held,memory,persist):
        self.held=held;self.memory=memory;self.persist=persist;self.registry={}

    def discover(self):
        queue=[(self.held.pid,None)];seen=set()
        while queue:
            pid,parent=queue.pop(0)
            if pid in seen:continue
            seen.add(pid);need(len(seen)<=64,'OWNED_TREE_PROCESS_BOUND')
            row,found,children=observe(pid,self.registry.get(pid),parent,self.held,self.memory,self.persist)
            if found is not None:self.registry[pid]=found
            need(len(self.registry)<=64,'LIFETIME_OWNED_IDENTITY_BOUND')
            if row.get('vanished_during_observation'):
                # Not an identity/UID waiver: it must be absent from the next
                # actual cgroup membership sample; known identities get post checks.
                seen.remove(pid)
                continue
            need(row['accepted'],'EXACT_OWNED_TREE_NOT_PROVEN')
            queue.extend((p,found) for p in children)
        return seen

    def post(self):
        remaining=[];unknown=[];records=[]
        for pid,saved in self.registry.items():
            fd=None
            try:
                fd=os.open('/proc/'+str(pid),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
                a=parse_stat(relative(fd,'stat',8192));b=parse_stat(relative(fd,'stat',8192))
                need(a['startticks']==b['startticks'] and a['PID']==b['PID']==pid,'POST_IDENTITY_RACE')
                if a['startticks']!=saved['startticks']:state='ORIGINAL_ABSENT_PID_REUSED'
                elif b['state'] in ('Z','X'):state='ORIGINAL_TERMINAL'
                else:state='ORIGINAL_LIVE';remaining.append(pid)
                records.append(dict(PID=pid,startticks=saved['startticks'],state=state))
            except (FileNotFoundError,ProcessLookupError):records.append(dict(PID=pid,startticks=saved['startticks'],state='ORIGINAL_ABSENT'))
            except BaseException as e:unknown.append(pid);records.append(dict(PID=pid,state='UNKNOWN',error_type=type(e).__name__))
            finally:
                if fd is not None:os.close(fd)
        return dict(remaining_known_owned=remaining,unknown_owned_liveness=unknown,records=records,
                    privileged_members_included=True,no_RSS_read=True)
