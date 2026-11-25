"""Two persistent markers in one actual container process, with a live helper thread."""
import json
import os
from pathlib import Path
import sys
import threading
import time


ROOT=Path('/t09-detach-work')


def identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    return int(raw[raw.rfind(')')+2:].split()[19])


def value(fd):
    st=os.fstat(fd)
    return [st.st_dev,st.st_ino]


def marker(name,row):
    flags=os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_WRONLY
    pending=ROOT/(name+'.pending')
    fd=os.open(pending,flags,0o600)
    row=dict(row,fd=fd,dev_inode=value(fd))
    data=(json.dumps(row,sort_keys=True)+'\n').encode()
    if os.write(fd,data)!=len(data):
        raise RuntimeError('SHORT_MARKER')
    os.fsync(fd)
    must(not (ROOT/name).exists(),'MARKER_HAS_NEVER_BEEN_PUBLISHED')
    os.rename(pending,ROOT/name)
    directory=os.open(ROOT,os.O_DIRECTORY|os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return fd


def must(condition,reason):
    if not condition:
        raise ValueError(reason)


def read_tracer(tid):
    for line in Path(f'/proc/self/task/{tid}/status').read_text().splitlines():
        if line.startswith('TracerPid:'):
            return int(line.split()[1])
    raise ValueError('MISSING_GUEST_THREAD_TRACER_PID')


def main():
    from mount_observation import explicit_readonly, observe

    context=json.loads(sys.argv[1])
    request,step,deadline=(context[key] for key in ('request','step','deadline'))
    must(os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']==step
         and all(os.environ.get(key)=='' for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
         and sys.dont_write_bytecode and sys.pycache_prefix is None,
         'EXACT_GUEST_SCOPE_CGROUP_AND_NO_ACCELERATOR')
    home=observe(Path('/miopen-private-home-v3'))
    elf=observe(Path('/opt/dtk/lib/libMIOpen.so.1.0'))
    mount=observe(ROOT)
    must(home['dev_inode']==context['home_dev_inode'] and home['three_reads']==['RW','RW','RW']
         and elf['dev_inode']==context['elf_dev_inode'] and explicit_readonly(elf)
         and elf['owner']==os.getuid() and elf['permissions']==0o400
         and mount['dev_inode']==context['work_dev_inode'] and mount['three_reads']==['RW','RW','RW']
         and all(home['owner']==item['owner']==os.getuid() and item['permissions']==0o700
                 and item['at_exact_point'] is True and item['same_before_open_after_dev_inode'] is True
                 for item in (home,mount)),
         'EXACT_PRIVATE_BINDS_SAME_COMPUTE_HOST_IDENTITIES')
    stopped=threading.Event()
    worker=threading.Thread(target=stopped.wait,name='trace-detach-live-sentinel',daemon=True)
    worker.start()
    pid=os.getpid()
    row=dict(request=request,step=step,pid=pid,startticks=identity(pid),uid=os.getuid(),
             thread_tid=worker.native_id,thread_startticks=identity(worker.native_id))
    first=marker('before.marker',row)
    try:
        while not (ROOT/'allow-after-detach.json').is_file():
            must(time.time()<deadline,'NO_VERIFIED_DETACH_WITHIN_SINGLE_EPOCH')
            time.sleep(.025)
        with (ROOT/'allow-after-detach.json').open('rb') as stream:
            permit=json.loads(stream.read(8192))
        live_tids={int(tid.name) for tid in Path('/proc/self/task').iterdir()}
        must(permit['request']==request and permit['step']==step and permit['pid']==pid
             and permit['startticks']==row['startticks'] and permit['dev_inode']==value(first)
             and set(permit['guest_live_tids'])==live_tids
             and all(read_tracer(tid)==0 for tid in live_tids) and time.time()<deadline,
             'GUEST_CANNOT_CONFIRM_ALL_OWN_THREADS_UNTRACED')
        second=marker('after.marker',dict(row,first_dev_inode=value(first),genuine_marker_after_detach=True))
        os.close(second)
    finally:
        os.close(first)
        stopped.set()
        worker.join(timeout=1)
    must(not worker.is_alive(),'HELPER_THREAD_ALREADY_RETURNED')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
