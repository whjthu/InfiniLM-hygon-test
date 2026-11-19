"""Explicit new5400 budget/cooperative lease/host headroom; no compute or other-PID signals."""
import fcntl,json,math,os,re,stat,time
from pathlib import Path
from gpu_admission import query
def need(ok,code):
    if not ok:raise ValueError(code)
def schedule(epoch,now):
    need(all(math.isfinite(x) for x in (epoch,now)) and 0<epoch<=now<epoch+600,'ONE_ORIGINAL5400_ADMISSION')
    return dict(epoch=epoch,deadline=epoch+5400,native_deadline=epoch+5250,supervisor_stop=epoch+5300)
def host_memory():
    raw=Path('/proc/meminfo').read_bytes();need(len(raw)<=65536,'BOUNDED_HOST_MEMINFO')
    def item(key):
        rows=re.findall(rb'^'+key+rb':\s+(\d+) kB$',raw,re.M);need(len(rows)==1,'HOST_MEMORY_FIELD');return int(rows[0])*1024
    total=item(b'MemTotal');available=item(b'MemAvailable');need(0<available<=total,'HOST_MEMORY_RANGE')
    return dict(MemTotal_bytes=total,MemAvailable_bytes=available,observed_monotonic=time.monotonic(),memory_reserved=False)
def acquire_lease(plan,owner,request):
    parent=Path('/home/whj/nvidia-baseline/alignment-assets/gpu-cooperative-locks')
    parent.mkdir(mode=0o700,exist_ok=True)
    need(parent.is_dir() and not parent.is_symlink() and parent.stat().st_uid==os.getuid(),'OWN_COOPERATIVE_LOCK_DIRECTORY')
    path=Path(plan['cooperative_lock']);need(path==parent/(plan['expected_GPU']['UUID']+'.lock'),'EXACT_UUID_LOCK')
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid(),'OWN_REGULAR_LOCK')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=path.stat();need((current.st_dev,current.st_ino)==(before.st_dev,before.st_ino),'LOCK_PATH_IDENTITY')
        record=dict(request_id=request,owner=owner,GPU_UUID=plan['expected_GPU']['UUID'],path=str(path),lock_device=before.st_dev,lock_inode=before.st_ino,LOCK_EX_NB=True,cooperative_only=True,excludes_uncooperative_processes=False,lock_file_deleted=False)
        raw=(json.dumps(record,sort_keys=True)+'\n').encode();os.ftruncate(fd,0);os.write(fd,raw);os.fsync(fd)
        return fd,record
    except BaseException:os.close(fd);raise
def runtime_foreign_processes(plan,registry,deadline):
    rows=query('gpu_uuid,pid','compute-apps',deadline);selected=[];foreign=[]
    for row in rows:
        need(len(row)==2,'RUNTIME_GPU_PROCESS_COLUMNS')
        uuid,pid=[x.strip() for x in row]
        need(re.fullmatch('[0-9]+',pid) is not None,'RUNTIME_GPU_PID')
        if uuid!=plan['expected_GPU']['UUID']:continue
        pid=int(pid);selected.append(pid);owned=registry.get(pid,registry.get(str(pid)))
        if owned is None:foreign.append(pid);continue
        try:
            raw=(Path('/proc')/str(pid)/'stat').read_text();fields=raw.rsplit(') ',1)[1].split()
            if int(fields[19])!=owned['startticks']:foreign.append(pid)
        except (FileNotFoundError,ProcessLookupError):foreign.append(pid)
    return dict(observed_monotonic=time.monotonic(),selected_GPU_PIDs=selected,foreign_selected_GPU_PIDs=foreign,action_on_foreign='STOP_ONLY_OWN_HELD_SCOPE',other_processes_signalled=False)
