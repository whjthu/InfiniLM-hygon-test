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

from scope import HOME, IDS, LIBRARY, LIBRARY_BYTES, LIBRARY_SHA, PARENT, DEPS, RUNTIME_VARS, require

ALIAS=Path('/miopen-private-state')
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


def read_bounded(path,cap):
    with path.open('rb') as stream:raw=stream.read(cap+1)
    require(len(raw)<=cap,'MOUNT_METADATA_EXTENT')
    return raw.decode('utf8','strict')


def effective(path):
    real=path.resolve(strict=True)
    require(real==path,'EXACT_EFFECTIVE_PATH_NO_SYMLINK')
    mode=os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK
    fd=os.open(path,mode)
    try:
        info=os.fstat(fd)
        raw=read_bounded(Path('/proc/self/fdinfo')/str(fd),16384)
        ids=re.findall(r'(?m)^mnt_id:\s*(\d+)\s*$',raw)
        require(len(ids)==1,'EXACT_EFFECTIVE_MOUNT_ID')
        match=[]
        for record in read_bounded(Path('/proc/self/mountinfo'),4<<20).splitlines():
            fields=record.split(' - ',1)[0].split()
            if fields[0]!=ids[0]:continue
            require(len(fields)>=6,'EXPECTED_MOUNT_RECORD')
            point=re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),fields[4])
            match.append((point,set(fields[5].split(','))))
        require(len(match)==1,'ONE_EFFECTIVE_MOUNT_ROW')
        point,options=match[0]
        vfs=bool(os.fstatvfs(fd).f_flag & os.ST_RDONLY)
        require(('ro' in options)==vfs and ('rw' in options)!=vfs,'MOUNT_AND_FS_FLAGS_DISAGREE')
        return dict(path_SHA256=sha(str(path).encode()),mount_id=int(ids[0]),point_SHA256=sha(point.encode()),
                    at_exact_point=point==str(path),readonly=vfs,dev_inode=[info.st_dev,info.st_ino],
                    owner=info.st_uid,permissions=stat.S_IMODE(info.st_mode))
    finally:os.close(fd)


def baseline(deadline):
    actual=Path(os.environ.get('HOME',''))
    require(actual==HOME and node_shape(HOME)=='DIRECTORY','ORIGINAL_HOME_UNVERIFIED')
    env={}
    for key in DEPS+RUNTIME_VARS:
        value=os.environ.get(key)
        env[key]=dict(configured=bool(value),value_SHA256=sha(value.encode()) if value else None)
    require(not any(env[key]['configured'] for key in DEPS),'MIOPEN_OR_XDG_OVERRIDE_UNCLASSIFIED')
    parents={key:node_shape(actual/('.'+key)) for key in ('config','cache')}
    trees={key:inventory(actual/path,deadline) for key,path in {
        'config':'.config/miopen','cache':'.cache/miopen','lock':'.config/miopen/miopen-lockfiles'}.items()}
    return dict(candidate_parents=parents,original_trees=trees,variables=env,reference_HOME=effective(HOME),
                historical_state_proven=False,historical_empty_proven=False)


def check_private(layout,inherited,explicit_assets,deadline):
    home=ALIAS/'home' if layout=='PRIVATE_HOME' else HOME
    require(Path(os.environ['HOME'])==home,'EXACT_SELECTED_HOME')
    for key in RUNTIME_VARS:
        value=os.environ.get(key)
        seen=inherited[key]
        require((sha(value.encode()) if value else None)==seen['value_SHA256'],'FIND_AND_CACHE_BEHAVIOR_ENV_DRIFT')
    for key in ('MIOPEN_SYSTEM_DB_PATH','MIOPEN_LOCKFILE_DIR','MIOPEN_LOCKFILE_PATH',
                'XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_DATA_HOME'):
        require(not os.environ.get(key),'UNEXPECTED_SYSTEM_DB_OR_HOME_OVERRIDE')
    locations={'config':home/'.config/miopen','cache':home/'.cache/miopen',
               'lock':home/'.config/miopen/miopen-lockfiles','user_db':home/'.config/miopen/user-db'}
    require(os.environ.get('MIOPEN_USER_DB_PATH')==str(locations['user_db']) and
            os.environ.get('MIOPEN_CUSTOM_CACHE_DIR')==str(locations['cache']),'EXACT_PRIVATE_DB_AND_CACHE_SELECTORS')
    shared_home=effective(HOME)
    private_home=effective(ALIAS)
    require(shared_home['readonly'] and not private_home['readonly'] and private_home['at_exact_point']
            and private_home['owner']==os.getuid() and private_home['permissions']==0o700,'SAME_RUNTIME_READONLY_AND_PRIVATE_ROOT')
    trees={}
    for name,path in locations.items():
        most=effective(path)
        baseline=effective(ALIAS/'home'/path.relative_to(home))
        require(most['dev_inode']==baseline['dev_inode'] and not most['readonly'] and
                most['owner']==os.getuid() and most['permissions']==0o700,'EXACT_PRIVATE_STATE_ALIAS_IDENTITY')
        if layout=='NARROW_MIOPEN_SUBTREES' and name in ('config','cache'):
            require(most['at_exact_point'],'NARROW_EXACT_MOUNTPOINT_REQUIRED')
        trees[name]=dict(mount=most,contents=inventory(path,deadline,most['dev_inode']))
    require(all(x['contents']['files']==0 for x in trees.values()),'NEW_QUALIFICATION_PRIVATE_STATE_NOT_EMPTY')
    readonly=[]
    for path in explicit_assets:
        record=effective(Path(path))
        require(record['readonly'] and record['at_exact_point'],'ORIGINAL_EXPLICIT_READONLY_ASSET_MOUNT_DIFFERS')
        readonly.append(record)
    return dict(layout=layout,owned_private_mounts=trees,protected_home=shared_home,
                pinned_readonly_asset_mounts=readonly,
                system_db_selector_proven=False,qualified_performance_cache_equivalence=False)


def system_db_candidates(deadline):
    locations=(Path('/opt/dtk/share/miopen/db'),Path('/opt/dtk/lib/miopen/db'))
    return [dict(path_SHA256=sha(str(path).encode()),status=node_shape(path),
                 mount=effective(path) if node_shape(path)=='DIRECTORY' else None)
            for path in locations]


def inspect_assets(paths):
    results=[]
    for path in paths:
        try:
            results.append(effective(Path(path)))
        except (OSError,ValueError) as error:
            results.append(dict(path_SHA256=sha(path.encode()),observed=False,error_type=type(error).__name__,
                                errno=getattr(error,'errno',None)))
    return results


def conv3d(dline):
    require(time.time()<dline,'SMOKE_FEASIBLE_BEFORE_DEADLINE')
    import torch
    require(torch.version.hip and torch.cuda.device_count()==1,'EXACT_ONE_ASSIGNED_HIP_DEVICE')
    x=torch.ones((1,4,8,8,8),device='cuda',dtype=torch.bfloat16)
    w=torch.ones((4,4,3,3,3),device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        result=torch.nn.functional.conv3d(x,w)
        torch.cuda.synchronize()
        require(tuple(result.shape)==(1,4,6,6,6) and bool(torch.isfinite(result).all())
                and bool(torch.all(result==108)), 'SYNTHETIC_CONV3D_NUMERIC_RESULT')
    return dict(planned_operations=1,actual_operations=1,bfloat16=True,numeric_ok=True,
                training_state_imported=False,warm_state_reuse_permitted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role');parser.add_argument('mode');parser.add_argument('--context-json',required=True)
    a=parser.parse_args();context=json.loads(a.context_json)
    deadline=context['deadline']
    report=dict(request=IDS[a.role,a.mode],role=a.role,mode=a.mode,phase='START',passed=False,
                actual_original_model_loads=0,actual_gold_reads=0,actual_optimizer_updates=0,
                cache_warm_state_comparable=False,formal_performance_eligible=False)
    try:
        require(sys.dont_write_bytecode and 0<deadline-time.time()<=180 and
                os.environ.get('SLURM_JOB_ID')==PARENT and os.environ.get('SLURM_STEP_ID','').isdecimal(),
                'EXACT_IMAGE_STEP_AND_DEADLINE')
        require(a.mode=='conv' or all(os.environ.get(key)=='' for key in
                ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')),'ZERO_GRES_NO_VISIBLE_SELECTORS')
        report.update(step=PARENT+'.'+os.environ['SLURM_STEP_ID'],phase='LIBRARY')
        library=read_file(LIBRARY,LIBRARY_BYTES,deadline)
        report['library']=library
        require(library==dict(bytes=LIBRARY_BYTES,SHA256=LIBRARY_SHA),'ORIGINAL_3570_EXACT_MIOPEN_LIBRARY')
        report['system_db_candidates']=system_db_candidates(deadline)
        require(all(x['mount'] is None or x['mount']['readonly'] for x in report['system_db_candidates']),
                'SHARED_SYSTEM_DB_CANDIDATE_MUST_BE_READONLY')
        report['phase']='LAYOUT'
        if a.mode in ('current','legacy'):
            report.update(baseline(deadline))
            report.update(history_projection=context['history_projection'], asset_runtime_pins=context['asset_runtime_pins'],
                          explicit_asset_mount_observation=inspect_assets(context['asset_mounts']))
        else:
            report['original_observation']=context['original_observation']
            report.update(check_private(context['layout'],context['variables'],context['asset_mounts'],deadline))
            if a.mode=='conv':
                report['phase']='SYNTHETIC_CONV3D'
                report['synthetic_conv']=conv3d(deadline)
                report['private_post']={p:inventory(ALIAS/'home'/p,deadline)
                    for p in ('.config/miopen','.cache/miopen')}
            else:
                report['model_free_mount_only']=True
        report.update(phase='COMPLETE',passed=True)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError) as exc:
        reason=str(exc)
        report.update(error_type=type(exc).__name__,errno=getattr(exc,'errno',None),
                      error_code=reason if re.fullmatch('[A-Z0-9_]{1,120}',reason) else None)
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
