"""Read state identities or verify private state mounts in the model-free image process."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
import socket
import subprocess
import time

from scope import HOME, REQUEST, NODE, LIBRARY, LIBRARY_BYTES, LIBRARY_SHA, PARENT, DEPS, require

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


import smoke
from runtime_pci import runtime_pci

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


def post_private(deadline):
    outputs={}
    for key,part in (('config','.config/miopen'),('user_db','.config/miopen/user-db'),
                     ('lock','.config/miopen/miopen-lockfiles'),('kernel_cache','.cache/miopen')):
        try:
            outputs[key]=inventory(ALIAS/part,deadline)
        except (OSError,ValueError) as error:
            outputs[key]=dict(state='UNKNOWN_INVENTORY_LIMIT_OR_VETO',**error_record(error))
    return outputs


def execute(context,report):
    deadline=context['deadline']
    report['phase']='SHARED_ASSET_AND_LIBRARY_IDENTITY'
    before=report['shared_home_before']=shared_before()
    paths=context['asset_mounts']
    mapped=[observe(Path(path)) for path in paths]
    report['explicit_assets']=mapped
    report['explicit_readonly_verified']=sum(explicit_readonly(item) for item in mapped)
    require(before['same_class_as_archived'] and len(paths)==len(set(paths))==report['explicit_readonly_verified']==10,
            'ORIGINAL_T09_TEN_READONLY_ASSETS_AND_OLD_HOME')
    library=report['library_before']=read_file(LIBRARY,LIBRARY_BYTES,deadline)
    require(library==dict(bytes=LIBRARY_BYTES,SHA256=LIBRARY_SHA),'ORIGINAL_EXACT_MIOPEN_PROVIDER_BYTES')
    report['phase']='PRIVATE_COLD_HOME'
    require(Path(os.environ['HOME'])==ALIAS and
            os.environ.get('MIOPEN_USER_DB_PATH')==str(ALIAS/'.config/miopen/user-db')
            and os.environ.get('MIOPEN_CUSTOM_CACHE_DIR')==str(ALIAS/'.cache/miopen')
            and not any(os.environ.get(key) for key in DEPS
                        if key not in ('MIOPEN_USER_DB_PATH','MIOPEN_CUSTOM_CACHE_DIR')),
            'EXACT_PRIVATE_SELECTORS_AND_SHARED_SYSTEM_DB_UNCHANGED')
    report['cold_private_before']={}
    private_checks(context,report['cold_private_before'])
    report['cold_leaves_empty']=True
    report['phase']='EXACT_NATIVE_STRUCTURAL_METADATA'
    report['original_conv_source']=smoke.origins(read_file,deadline)
    require(report['original_conv_source']['selected_channels']==65,'EXACT_FROZEN_65_CHANNEL_BRANCH')
    report['phase']='EXACT_ONE_DCU_MAPPING_AND_CONV3D'
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==context['host_boot_id'],
            'SAME_COMPUTE_BOOT_IN_CONTAINER')
    report['assigned_runtime_PCI']=runtime_pci()
    require(report['assigned_runtime_PCI'] in context['host_PCI'] and len(set(context['host_PCI']))==8,
            'ACTUAL_SINGLE_ASSIGNED_PCI_FROM_N13_INVENTORY')
    try:
        smoke.conv(deadline,report,smoke.SYNTHETIC)
    finally:
        report['private_state_after']=post_private(deadline)
        report['shared_home_after']=shared_before()
        report['library_after']=read_file(LIBRARY,LIBRARY_BYTES,deadline)
    require(report['library_after']==library and report['shared_home_after']==before,
            'PROVIDER_OR_PROTECTED_SHARED_HOME_CHANGED')
    report['phase']='COMPLETE'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-json',required=True)
    args=parser.parse_args()
    context=json.loads(args.context_json)
    result=dict(request=REQUEST,phase='START',passed=False,actual_weights_bytes_read=0,actual_model_loads=0,
           actual_training_samples_read=0,actual_optimizer_updates=0,actual_gold_reads=0,
           issued_conv3d_calls=0,completed_conv3d_calls=0,native_file_open_write_trace='UNKNOWN_NOT_CAPTURED',
           usable_as_formal_benchmark_or_cache_seed=False,actual_historical_latent_dimensions_proven=False)
    try:
        require(socket.gethostname().split('.')[0]==NODE and sys.dont_write_bytecode and
            os.environ.get('SLURM_JOB_ID')==PARENT and os.environ.get('SLURM_STEP_ID','').isdecimal()
            and context['step']==PARENT+'.'+os.environ['SLURM_STEP_ID']
            and re.fullmatch(r'\d+',os.environ.get('CUDA_VISIBLE_DEVICES','')) is not None
            and 0<context['deadline']-time.time()<=240,'EXACT_BOUNDED_SINGLE_STEP_CONTAINER_CONTEXT')
        result.update(step=context['step'],authority_SHA256=context['authority_SHA256'])
        execute(context,result)
        result['passed']=True
    except (OSError,ValueError,TypeError,RuntimeError,KeyError,ImportError,subprocess.SubprocessError) as error:
        result.update(error_record(error))
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
