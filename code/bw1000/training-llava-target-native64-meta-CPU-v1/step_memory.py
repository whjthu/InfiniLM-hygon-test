"""Read-only, exact numeric Slurm step memory-controller scope (v1 and v2).

Charged memory is NOT sum(VmRSS), allocation size, or a verified hard limit.
No cgroup writes, migration, global /proc scan, Native imports, or device calls.
"""
import os
from pathlib import PurePosixPath
import re
import stat
import time
from common import need, sha

LIMIT = 8 * 1024**3

def bounded_fd(fd, cap):
    out = b''
    while len(out) <= cap:
        part = os.read(fd, min(8192, cap + 1 - len(out)))
        if not part: break
        out += part
    need(len(out) <= cap, 'CGROUP_RECORD_BOUND')
    return out

def raw_file(path, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try: return bounded_fd(fd, cap)
    finally: os.close(fd)

def relative(fd, name, cap=65536):
    f = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try: return bounded_fd(f, cap)
    finally: os.close(f)

def absolute(value):
    p = PurePosixPath(value)
    need(p.is_absolute() and '..' not in p.parts and str(p) == value, 'CGROUP_CANONICAL_ABSOLUTE_PATH')
    return p

def unescape(value):
    # mountinfo uses precisely these octal escapes; do not shell-evaluate paths.
    return re.sub(r'\\(040|011|012|134)', lambda m: chr(int(m[1], 8)), value)

def memberships(raw):
    result = []
    for line in raw.decode('ascii').splitlines():
        h, controllers, path = line.split(':', 2)
        need(h.isdigit(), 'CGROUP_HIERARCHY_ID')
        absolute(path)
        result.append(dict(hierarchy=int(h), controllers=controllers.split(',') if controllers else [], path=path))
    need(result and len(result) <= 64, 'CGROUP_MEMBERSHIP_COUNT')
    return result

def mounts(raw):
    out = []
    for line in raw.decode('ascii').splitlines():
        left, sep, right = line.partition(' - ')
        need(bool(sep), 'MOUNTINFO_SEPARATOR')
        a, b = left.split(), right.split()
        need(len(a) >= 6 and len(b) >= 3, 'MOUNTINFO_SCHEMA')
        if b[0] not in ('cgroup', 'cgroup2'): continue
        out.append(dict(id=int(a[0]), device=a[2], root=unescape(a[3]), mount=unescape(a[4]),
                        fs=b[0], options=b[2].split(',')))
    return out

def stable_directory(path):
    # Traverse from / without following physical-directory symlinks.
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute(path).parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd); raise

def stamp(fd):
    s = os.fstat(fd)
    need(stat.S_ISDIR(s.st_mode), 'CGROUP_DIRECTORY_TYPE')
    return dict(dev=s.st_dev, ino=s.st_ino, uid=s.st_uid, mode=s.st_mode)

def scalar(raw, allow_max=False):
    value = raw.decode('ascii').strip()
    if allow_max and value == 'max': return 'max'
    need(value.isdigit(), 'CGROUP_INTEGER_SCHEMA')
    return int(value)

def counters(raw):
    out = {}
    for line in raw.decode('ascii').splitlines():
        parts = line.split()
        need(len(parts) == 2 and parts[0] not in out and parts[1].isdigit(), 'CGROUP_COUNTER_SCHEMA')
        out[parts[0]] = int(parts[1])
    need(bool(out), 'CGROUP_COUNTERS_EMPTY')
    return out

class StepMemory:
    def __init__(self, parent, step, deadline, monotonic_end):
        self.fd = None; self.deadline = deadline; self.end = monotonic_end
        need(parent.isdigit() and step.isdigit(), 'ACTUAL_NUMERIC_SLURM_STEP_REQUIRED')
        need(os.environ.get('SLURM_JOB_ID') == parent and os.environ.get('SLURM_STEP_ID') == step,
             'ACTUAL_SLURM_STEP_CONTEXT')
        ownraw = raw_file('/proc/self/cgroup', 16384)
        mountraw = raw_file('/proc/self/mountinfo', 1024*1024)
        rows = memberships(ownraw); candidates = []
        for row in rows:
            if any('memory' in r['controllers'] for r in rows) and 'memory' not in row['controllers']: continue
            version = 1 if 'memory' in row['controllers'] else 2 if row['hierarchy'] == 0 and not row['controllers'] else None
            if version is None: continue
            for mount in mounts(mountraw):
                if not (mount['fs'] == ('cgroup2' if version == 2 else 'cgroup') and
                        (version == 2 or 'memory' in mount['options'])): continue
                parts = absolute(row['path']).parts
                jobs = [i for i, p in enumerate(parts) if p == 'job_' + parent]
                steps = [i for i, p in enumerate(parts) if p == 'step_' + step]
                if len(jobs) != 1 or len(steps) != 1 or steps[0] != jobs[0] + 1: continue
                # Never select the job parent or step_batch/step_extern/system root.
                if any(p.startswith(('job_', 'step_')) for p in parts[steps[0]+1:]): continue
                scope = PurePosixPath(*parts[:steps[0]+1]); root = absolute(mount['root'])
                if scope != root and root not in scope.parents: continue
                physical = absolute(mount['mount']) / scope.relative_to(root)
                candidates.append((version, row, mount, str(scope), str(physical)))
        need(len(candidates) == 1, 'EXACT_MEMORY_STEP_MOUNT_NOT_PROVEN')
        self.version, self.row, self.mount, self.scope, self.path = candidates[0]
        self.fd = stable_directory(self.path)
        try:
            self.identity = stamp(self.fd)
            need(str(os.major(self.identity['dev']))+':'+str(os.minor(self.identity['dev']))==self.mount['device'],
                 'CGROUP_DIRECTORY_NOT_ON_SELECTED_MOUNT')
            self.proof = dict(parent=parent, step=parent+'.'+step, memory_controller_version=self.version,
                              hierarchy=self.row['hierarchy'], hierarchy_scope=self.scope, physical_scope=self.path,
                              mount=self.mount, directory_identity=self.identity,
                              self_membership_SHA256=sha(ownraw), mountinfo_SHA256=sha(mountraw),
                              self_PID=os.getpid(), scope_is_numeric_step_not_job=True,
                              charged_memory_is_sum_RSS=False, allocation_is_hard_limit=False,
                              memory_control_files_written=False)
            self.check_time()
        except BaseException:
            self.close(); raise

    def close(self):
        if self.fd is not None: os.close(self.fd); self.fd = None

    def check_time(self):
        need(time.time() < self.deadline and time.monotonic() < self.end, 'CGROUP_ORIGINAL_DEADLINE')

    def check_identity(self):
        self.check_time(); need(stamp(self.fd) == self.identity, 'CGROUP_SCOPE_FD_CHANGED')
        fd = stable_directory(self.path)
        try: need(stamp(fd) == self.identity, 'CGROUP_SCOPE_PATH_CHANGED')
        finally: os.close(fd)
        # No silently remapped controller or cgroup namespace between samples.
        need(sha(raw_file('/proc/self/mountinfo',1024*1024)) == self.proof['mountinfo_SHA256'], 'MEMORY_MOUNTINFO_CHANGED')

    def check_member(self, raw):
        matches = [r for r in memberships(raw) if r['hierarchy'] == self.row['hierarchy'] and
                   r['controllers'] == self.row['controllers']]
        need(len(matches) == 1, 'PROCESS_MEMORY_HIERARCHY_CHANGED')
        p = absolute(matches[0]['path']); root = absolute(self.scope)
        need(p == root or root in p.parents, 'PROCESS_OUTSIDE_ORIGINAL_NUMERIC_STEP')
        need(not any(q.startswith(('job_','step_')) for q in p.relative_to(root).parts), 'NESTED_OTHER_STEP')
        return matches[0]['path']

    def sample(self):
        self.check_identity()
        self.check_member(raw_file('/proc/self/cgroup',16384))
        all_pids = set(); dirs = []; queue = [(os.dup(self.fd), '')]
        try:
            while queue:
                fd, name = queue.pop(0)
                try:
                    self.check_time(); need(len(dirs) < 64, 'STEP_CGROUP_DIRECTORY_BOUND')
                    before = stamp(fd); body = relative(fd, 'cgroup.procs', 65536)
                    pids = body.split(); need(all(p.isdigit() and int(p)>0 for p in pids), 'CGROUP_PROCS_SCHEMA')
                    all_pids.update(map(int,pids)); need(len(all_pids) <= 512, 'STEP_MEMBER_BOUND')
                    if self.version == 1:
                        need(scalar(relative(fd,'memory.use_hierarchy')) == 1, 'V1_HIERARCHICAL_CHARGING_REQUIRED')
                    dirs.append(dict(path=name or '.', identity=before, PIDs=sorted(set(map(int,pids)))))
                    for entry in sorted(os.listdir(fd)):
                        self.check_time()
                        st = os.stat(entry,dir_fd=fd,follow_symlinks=False)
                        need(not stat.S_ISLNK(st.st_mode), 'CGROUP_SYMLINK_UNSUPPORTED')
                        if stat.S_ISDIR(st.st_mode):
                            need(not entry.startswith(('job_','step_')), 'OTHER_STEP_INSIDE_MEMORY_SCOPE')
                            need(len(queue)+len(dirs)<64,'STEP_CGROUP_DIRECTORY_BOUND')
                            queue.append((os.open(entry,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd),name+'/'+entry))
                    need(stamp(fd)==before,'CGROUP_DESCENDANT_CHANGED')
                finally: os.close(fd)
        finally:
            for fd,_ in queue: os.close(fd)
        usage_name, max_name = ('memory.current','memory.max') if self.version==2 else ('memory.usage_in_bytes','memory.limit_in_bytes')
        current = scalar(relative(self.fd,usage_name)); maximum=scalar(relative(self.fd,max_name),allow_max=self.version==2)
        stats=relative(self.fd,'memory.stat'); parsed=counters(stats)
        if self.version==2:
            events=counters(relative(self.fd,'memory.events'))
        else:
            events=dict(failcnt=scalar(relative(self.fd,'memory.failcnt')))
            need('hierarchical_memory_limit' in parsed,'V1_HIERARCHICAL_LIMIT_METADATA')
        self.check_identity()
        return dict(observed_monotonic=time.monotonic(), charged_memory_bytes=current, observed_limit=maximum,
                    limit_is_not_allocation_assertion=True, events=events, memory_stat_SHA256=sha(stats),
                    memory_stat=parsed, directories=dirs, member_PIDs=sorted(all_pids),
                    soft_limit_bytes=LIMIT, soft_limit_exceeded=current>LIMIT, hard_limit=False)
