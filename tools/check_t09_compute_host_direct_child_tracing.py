"""Check ordinary compute-HOST direct-child tracing under a new private directory."""
import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


TRACER=Path('/usr/bin/strace')
CAP=256*1024


def tracer_argv(path):
    return [str(TRACER),'-qq','-f','-yy','-s','256','-e','trace=openat,flock,close',
            '-o',str(path/'native-tracer-private.log'),sys.executable,'-B',
            str(Path(__file__).resolve()),'--child',str(path)]


def child(path):
    import fcntl

    check_owned(path)
    if any(item.name!='native-tracer-private.log' for item in path.iterdir()):
        raise ValueError('EXACT_EMPTY_PROBE_CHILD_DIRECTORY')
    marker=path/'owned-lock-fixture'
    fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(fd,fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return 0


def classify(raw):
    hit={'exclusive_open':False,'lock_acquired':False,'lock_released':False}
    for line in raw.decode('utf-8','replace').splitlines():
        if 'owned-lock-fixture' not in line:
            continue
        hit['exclusive_open']|=('openat(' in line and 'O_EXCL' in line and 'O_WRONLY' in line
                                and re.search(r'= [0-9]+<',line) is not None)
        hit['lock_acquired']|=('flock(' in line and 'LOCK_EX' in line and line.rstrip().endswith('= 0'))
        hit['lock_released']|=('flock(' in line and 'LOCK_UN' in line and line.rstrip().endswith('= 0'))
    return hit


def check_owned(path):
    if (not path.is_absolute() or path.resolve(strict=True)!=path or path.stat().st_uid!=os.getuid()
        or stat.S_IMODE(path.stat().st_mode)!=0o700 or len(str(path))>160):
        raise ValueError('EXACT_OWNER_COMPUTE_HOST_PROBE_DIRECTORY')


def main():
    import resource
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_private_probe_path',type=Path)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args()
    path=args.new_private_probe_path
    if args.child:
        return child(path)
    check_owned(path.parent)
    if len(str(path))>160:
        raise ValueError('BOUND_PROBE_PATH')
    path.mkdir(mode=0o700)
    result=dict(actual_model_calls=0,actual_optimizer_updates=0,native_observer_admitted=False,
                direct_child_capability=False)
    try:
        before=TRACER.stat()
        if not stat.S_ISREG(before.st_mode) or not os.access(TRACER,os.X_OK):
            raise ValueError('NO_ORDINARY_COMPUTE_HOST_TRACER')
        resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))
        done=subprocess.run(tracer_argv(path),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,timeout=8)
        result.update(tracer_rc=done.returncode,stderr_bytes=len(done.stderr),tracer_bytes=before.st_size)
        result['permission_denied_reported']=bool(re.search(rb'operation not permitted|permission denied',done.stderr,re.I))
        with (path/'native-tracer-private.log').open('rb') as stream:
            raw=stream.read(CAP+1)
        if len(raw)>=CAP:
            raise ValueError('TRACE_OUTPUT_CAP')
        result.update(classify(raw))
        result['direct_child_capability']=done.returncode==0 and all(result[k] for k in
            ('exclusive_open','lock_acquired','lock_released'))
    except (OSError,ValueError,subprocess.TimeoutExpired) as error:
        result.update(error_type=type(error).__name__,errno=getattr(error,'errno',None))
    print(json.dumps(result,sort_keys=True))
    return 0 if result['direct_child_capability'] else 2


if __name__=='__main__':
    raise SystemExit(main())
