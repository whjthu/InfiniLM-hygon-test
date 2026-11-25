"""Read state identities or verify private state mounts in the model-free image process."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

from scope import HOME, IDS, LIBRARY, LIBRARY_BYTES, LIBRARY_SHA, PARENT, DEPS, ROLES, RUNTIME_VARS, require

from mount_observation import error_record, explicit_readonly, observe, shared_readonly
CAP_BYTES=64<<20
CAP_FILES=128
CAP_DIRS=128


def sha(raw):return sha256(raw).hexdigest()


def identity(info):
    return [info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]


def read_file(path,limit,deadline):
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size<=limit,'BOUNDED_EXACT_REGULAR_STATE_FILE')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        require(identity(os.fstat(stream.fileno()))==identity(before),'STATE_FILE_OPEN_IDENTITY')
        total=0
        digest=sha256()
        for raw in iter(lambda:stream.read(1<<20),b''):
            require(time.time()<deadline,'STATE_READ_DEADLINE')
            total+=len(raw)
            require(total<=limit,'STATE_FILE_GREW')
            digest.update(raw)
        require(identity(os.fstat(stream.fileno()))==identity(before),'STATE_FILE_FD_CHANGED')
    require(total==before.st_size and identity(path.lstat())==identity(before),'STATE_FILE_PATH_CHANGED')
    return dict(bytes=total,SHA256=digest.hexdigest())


def node_shape(path):
    try:
        return 'DIRECTORY' if stat.S_ISDIR(path.lstat().st_mode) and path.resolve(strict=True)==path else 'UNSAFE'
    except FileNotFoundError:return 'ABSENT'
    except OSError:return 'UNKNOWN'


def inventory(path,deadline,base=None):
    state=node_shape(path)
    if state!='DIRECTORY':
        return dict(state=state,files=None,bytes=None)
    fingerprints=[]
    names=[]
    stack=[path]
    total=0
    while stack:
        require(time.time()<deadline and len(names)<CAP_DIRS,'STATE_INVENTORY_BOUND')
        current=stack.pop()
        require(node_shape(current)=='DIRECTORY','NONCANONICAL_CACHE_ANCESTOR')
        before=identity(current.lstat())
        entries=[]
        with os.scandir(current) as scan:
            for child in scan:
                require(len(entries)<CAP_FILES+CAP_DIRS,'BOUNDED_STATE_DIRECTORY')
                entries.append(child)
        entries.sort(key=lambda item:item.name)
        names.append(current.relative_to(path).as_posix())
        for child in entries:
            childpath=Path(child.path)
            if child.is_dir(follow_symlinks=False):
                stack.append(childpath)
                require(len(stack)<=CAP_DIRS,'BOUNDED_STATE_SUBDIRECTORIES')
            else:
                require(len(fingerprints)<CAP_FILES and child.is_file(follow_symlinks=False),'STATE_SPECIAL_FILE_OR_FILE_LIMIT')
                checked=read_file(childpath,CAP_BYTES-total,deadline)
                total+=checked['bytes']
                fingerprints.append((childpath.relative_to(path).as_posix(),checked))
        require(identity(current.lstat())==before,'STATE_DIRECTORY_CHANGED')
    body=json.dumps(sorted(fingerprints),sort_keys=True).encode()
    actual=identity(path.lstat())
    if base is not None:require(actual[:2]==base,'STATE_ROOT_IDENTITY_CHANGED')
    return dict(state='DIRECTORY',files=len(fingerprints),bytes=total,tree_SHA256=sha(body),
                directories=len(names),directory_names_SHA256=sha(json.dumps(sorted(names)).encode()),
                root_dev_inode=actual[:2],readonly=bool(os.statvfs(path).f_flag & os.ST_RDONLY))


def capture(deadline,paths,report):
    report['phase']='SYSTEM_DB_OBSERVATION'
    report['system_db_candidates']=[
        dict(observation_only=True,role=role,target=observe(Path(path)))
        for role,path in (('SHARE_LAYOUT','/opt/dtk/share/miopen/db'),('LIB_LAYOUT','/opt/dtk/lib/miopen/db'))]
    report['phase']='HOME_OBSERVATION'
    exact=os.environ.get('HOME')==str(HOME)
    report['home']=dict(runtime_HOME_matches_expected=exact,mount=observe(HOME))
    report['candidate_parents']={key:node_shape(HOME/('.'+key)) for key in ('config','cache')}
    report['explicit_asset_mounts']=[observe(Path(path)) for path in paths]
    settings={}
    for key in DEPS+RUNTIME_VARS:
        value=os.environ.get(key)
        settings[key]=dict(configured=bool(value),value_SHA256=sha(value.encode()) if value else None)
    report['variables']=settings
    report['explicit_readonly_verified']=sum(explicit_readonly(item) for item in report['explicit_asset_mounts'])
    report['explicit_readonly_planned']=len(paths)
    require(exact,'ORIGINAL_HOME_UNVERIFIED')
    require(not any(settings[key]['configured'] for key in DEPS),'MIOPEN_OR_XDG_OVERRIDE_UNCLASSIFIED')
    require(all(item in ('DIRECTORY','ABSENT') for item in report['candidate_parents'].values()),'UNSAFE_ORIGINAL_PARENT')
    report['phase']='CACHE_INVENTORY'
    report['original_trees']={}
    for role,path in (('config','.config/miopen'),('cache','.cache/miopen'),('lock','.config/miopen/miopen-lockfiles')):
        report['original_trees'][role]=inventory(HOME/path,deadline)
    report['phase']='READONLY_SCOPE_CONSISTENCY'
    home=report['home']['mount']
    require(shared_readonly(home),'HOME_READONLY_BOUNDARY_UNVERIFIED')
    require(report['explicit_readonly_verified']==report['explicit_readonly_planned']>0,
            'EXPLICIT_SCOPE_READONLY_MOUNT_CHANGED')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=ROLES);parser.add_argument('--context-json',required=True)
    a=parser.parse_args();context=json.loads(a.context_json)
    deadline=context['deadline']
    report=dict(request=IDS[a.role],role=a.role,mode='current-v2',phase='START',passed=False,
        scope='READONLY_MOUNT_OBSERVATION_ONLY',qualify_authorized=False,private_writes_permitted=False,
        actual_GPU=0,actual_models_loaded=0,actual_gold_reads=0,actual_optimizer_updates=0,
        historical_cache_empty_proven=False,historical_warm_state_equivalent=False)
    try:
        step=os.environ.get('SLURM_STEP_ID','')
        require(sys.dont_write_bytecode and os.environ.get('SLURM_JOB_ID')==PARENT and step.isdecimal()
              and 0<deadline-time.time()<=120 and all(os.environ.get(key)==''
                for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')),
                'EXACT_ZERO_GRES_START')
        report.update(step=PARENT+'.'+step,phase='LIBRARY',asset_runtime_pins=context['asset_runtime_pins'])
        report['library']=read_file(LIBRARY,LIBRARY_BYTES,deadline)
        require(report['library']==dict(bytes=LIBRARY_BYTES,SHA256=LIBRARY_SHA),'SAME_EXACT_INSTALLED_MIOPEN_LIBRARY')
        capture(deadline,context['asset_mounts'],report)
        report.update(phase='COMPLETE',passed=True)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError) as error:
        report.update(error_record(error))
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
