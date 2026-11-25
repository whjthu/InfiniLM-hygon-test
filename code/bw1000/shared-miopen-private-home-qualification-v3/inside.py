"""Read state identities or verify private state mounts in the model-free image process."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
import subprocess
import time

from scope import EVIDENCE, HOME, IDS, LIBRARY, LIBRARY_BYTES, LIBRARY_SHA, PARENT, DEPS, ROLES, RUNTIME_VARS, require

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


from private_lock import execute as lock_execute

ALIAS=Path('/miopen-private-home-v3')
ROOT_SHA='8a5edab282632443219e051e4ade2d1d5bbc671c781051bf1437897cbdfea0f1'
OWNED={
    '':{'.config','.cache'},
    '.config':{'miopen'},
    '.config/miopen':{'user-db','miopen-lockfiles'},
    '.config/miopen/user-db':set(),
    '.config/miopen/miopen-lockfiles':set(),
    '.cache':{'miopen'},
    '.cache/miopen':set(),
}
LOCK_RELATIVE='.config/miopen/miopen-lockfiles'


def kernel_or_matches(reading):
    modes=reading.get('three_reads',[])
    if reading.get('metadata_status')!='RECORDED' or len(modes)!=3 or any(mode not in ('RW','RO') for mode in modes):
        return False
    return (modes[0]=='RO' or modes[1]=='RO')==(modes[2]=='RO')


def shared_before():
    observed=observe(HOME)
    parents={key:node_shape(HOME/('.'+key)) for key in ('config','cache')}
    good=(kernel_or_matches(observed) and observed.get('three_reads')==['RW','RO','RO']
          and observed.get('fstype')=='overlay' and observed.get('mount_point_SHA256')==ROOT_SHA
          and observed.get('at_exact_point') is False and observed.get('same_before_open_after_dev_inode') is True
          and parents=={'config':'ABSENT','cache':'ABSENT'})
    return dict(home=observed,parents=parents,linux_effective_readonly_OR=kernel_or_matches(observed),
                same_class_as_archived=good)


def private_checks(context,results):
    uid=os.getuid()
    expected=context['host_private_identity']
    require(uid==context['host_uid'] and set(expected)==set(OWNED),'EXACT_HOST_PRIVATE_UID_AND_MEMBERS')
    anchor=None
    for relative,children in OWNED.items():
        path=ALIAS/relative
        mount=results[relative or 'HOME']=observe(path)
        wanted=expected[relative]
        flags=(mount.get('metadata_status')=='RECORDED' and mount.get('three_reads')==['RW','RW','RW']
               and mount.get('same_before_open_after_dev_inode') is True and mount.get('object_type')=='DIRECTORY'
               and mount.get('owner')==uid and mount.get('permissions')==0o700)
        mount['host_dev_inode_matches']=mount.get('dev_inode')==wanted
        if not relative:
            anchor=mount.get('mount_id') if mount.get('at_exact_point') is True else None
        mount['same_private_alias_mount_id']=anchor is not None and mount.get('mount_id')==anchor
        require(flags and mount['host_dev_inode_matches'] and mount['same_private_alias_mount_id'],
                'EXACT_HOST_GUEST_PRIVATE_ALIAS_IDENTITY_AND_RW_REQUIRED')
        # Names are used for a comparison only and never returned.
        has={item.name for item in path.iterdir()}
        mount['directory_child_shape_matches']=has==children
        require(has==children,'PRIVATE_STATE_UNKNOWN_CONTENT')
    return results


def checks(role,context,report):
    report['phase']='SHARED_AND_ASSET_RO'
    prior=report['original_home_before']=shared_before()
    paths=context['asset_mounts']
    report['explicit_assets']=[observe(Path(path)) for path in paths]
    report['explicit_readonly_planned']=EVIDENCE[role]['explicit_readonly_verified']
    report['explicit_readonly_verified']=sum(explicit_readonly(item) for item in report['explicit_assets'])
    require(prior['same_class_as_archived'],'UNEXPECTED_SHARED_HOME_EFFECTIVE_MOUNT_STATE')
    require(len(paths)==len(set(paths))==report['explicit_readonly_verified']==report['explicit_readonly_planned'],
            'EXPLICIT_SHARED_ASSET_READONLY_MOUNT_DIFFERS')
    report['system_db_observation_only']=[observe(Path('/opt/dtk/share/miopen/db')),observe(Path('/opt/dtk/lib/miopen/db'))]
    report['phase']='PRIVATE_MOUNT_AND_SELECTORS'
    require(os.environ.get('HOME')==str(ALIAS)
            and os.environ.get('MIOPEN_USER_DB_PATH')==str(ALIAS/'.config/miopen/user-db')
            and os.environ.get('MIOPEN_CUSTOM_CACHE_DIR')==str(ALIAS/'.cache/miopen')
            and not any(os.environ.get(key) for key in DEPS
                if key not in ('MIOPEN_USER_DB_PATH','MIOPEN_CUSTOM_CACHE_DIR')),
            'EXACT_PRIVATE_SELECTORS_AND_SHARED_DB_UNCHANGED')
    report['private_selector_binding']=True
    report['private_before']={}
    private_checks(context,report['private_before'])
    report['phase']='PRIVATE_FLOCK'
    report['lock_test']={}
    lock_execute(ALIAS/LOCK_RELATIVE,context['host_private_identity'][LOCK_RELATIVE],IDS[role],
                 context['deadline'],report['lock_test'])
    report['phase']='READBACK_AND_CLEANUP'
    report['private_after']={}
    private_checks(context,report['private_after'])
    final=report['original_home_after']=shared_before()
    require(final==prior,'SHARED_HOME_CLASS_OR_IDENTITY_CHANGED')
    require(report['lock_test'].get('sentinel_unlinked_by_identity') is True
            and report['lock_test'].get('lock_dir_empty_after') is True,'ORIGINAL_SENTINEL_CLEANUP_NOT_PROVEN')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=ROLES);parser.add_argument('--context-json',required=True)
    a=parser.parse_args();context=json.loads(a.context_json)
    deadline=context['deadline']
    report=dict(request=IDS[a.role],role=a.role,mode='private-home-v3',phase='START',passed=False,
                actual_GPU=0,actual_models_loaded=0,actual_gold_reads=0,actual_optimizer_updates=0,
                runtime_provider_loaded=False,benchmark_or_blind_ready=False,miopen_lock_selection_proven=False,
                code_scope='PRIVATE_ORDINARY_FILE_FLOCK_ONLY',shared_paths_written=0)
    try:
        step=os.environ.get('SLURM_STEP_ID','')
        require(sys.dont_write_bytecode and os.environ.get('SLURM_JOB_ID')==PARENT and step.isdecimal()
              and 0<deadline-time.time()<=120 and all(os.environ.get(key)==''
                for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')),
                'EXACT_ZERO_GRES_START')
        report.update(step=PARENT+'.'+step,phase='LIBRARY_IDENTITY',
                      asset_runtime_manifest_SHA256=context['asset_runtime_manifest_SHA256'],
                      authority_SHA256=context['authority_SHA256'])
        report['library']=read_file(LIBRARY,LIBRARY_BYTES,deadline)
        require(report['library']==dict(bytes=LIBRARY_BYTES,SHA256=LIBRARY_SHA),'SAME_EXACT_INSTALLED_MIOPEN_LIBRARY')
        checks(a.role,context,report)
        report.update(phase='COMPLETE',passed=True)
    except (OSError,ValueError,RuntimeError,TypeError,KeyError,subprocess.SubprocessError) as error:
        report.update(error_record(error))
    print(json.dumps(report,sort_keys=True,allow_nan=False),flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
