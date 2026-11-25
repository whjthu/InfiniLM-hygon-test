"""Copy one original ELF or observe its sealed single-file bind after normal Torch import."""
import argparse
from hashlib import sha256
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

from maps_probe import EXPECTED, EXPECTED_PIN, MAX_MAPS, ROW, group_identity, inspect_maps, metadata, stamp
from mount_observation import explicit_readonly, observe
from scope import REQUEST, TORCH_ROOT, TORCH_VERSION_PIN, need

DEST=Path('/t09-bound-lib-staging')
PRIVATE_HOME='/miopen-private-home-v3'
FILENAME='libMIOpen.so.1.0'


def checksum(fd,deadline,mirror=None):
    before=os.fstat(fd)
    need(stat.S_ISREG(before.st_mode) and before.st_size==EXPECTED_PIN['bytes'],
         'EXACT_BOUNDED_REGULAR_MIOPEN_ELF')
    body=sha256()
    count=0
    while count<before.st_size:
        need(time.time()<deadline,'ORIGINAL_ONE_EPOCH_ELF_BUDGET')
        part=os.read(fd,min(1<<20,before.st_size-count))
        need(bool(part),'ELF_SHORT_READ')
        body.update(part)
        if mirror is not None:
            need(mirror.write(part)==len(part),'PRIVATE_ELF_SHORT_WRITE')
        count+=len(part)
    need(not os.read(fd,1) and stamp(os.fstat(fd))==stamp(before),'EXACT_STABLE_ELF_DESCRIPTOR')
    return dict(bytes=count,SHA256=body.hexdigest()),before


def provision(report,deadline):
    mount=report['private_destination_mount']=observe(DEST)
    need(mount['metadata_status']=='RECORDED' and mount['at_exact_point'] is True
         and mount['same_before_open_after_dev_inode'] is True and mount['owner']==os.getuid()
         and mount['permissions']==0o700 and mount['three_reads']==['RW','RW','RW'],'EXACT_OWNED_PRIVATE_COPY_MOUNT')
    need(EXPECTED.resolve(strict=True)==EXPECTED,'ORIGINAL_NATURAL_IMAGE_ELF_CANONICAL_PATH')
    source=os.open(EXPECTED,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK)
    try:
        path_stat=EXPECTED.lstat()
        need(stamp(path_stat)==stamp(os.fstat(source)),'ORIGINAL_NATURAL_PATH_FD_IDENTITY')
        directory=os.open(DEST,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            root=os.fstat(directory)
            need([root.st_dev,root.st_ino]==mount['dev_inode'] and not os.listdir(directory),
                 'EMPTY_PRIVATE_COPY_DIRECTORY_BEFORE_EXCLUSIVE_CREATE')
            output=os.open(FILENAME,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,
                           0o600,dir_fd=directory)
            with os.fdopen(output,'wb') as mirror:
                report['copy_exclusively_created']=True
                original,first=checksum(source,deadline,mirror)
                need(original==EXPECTED_PIN and stamp(EXPECTED.lstat())==stamp(first),'ORIGINAL_IMAGE_ELF_SHA256')
                mirror.flush()
                os.fchmod(mirror.fileno(),0o400)
                os.fsync(mirror.fileno())
                sealed=os.fstat(mirror.fileno())
                need(sealed.st_uid==os.getuid() and sealed.st_nlink==1 and stat.S_IMODE(sealed.st_mode)==0o400
                     and sealed.st_size==EXPECTED_PIN['bytes'],'EXACT_PRIVATE_OWNER_READONLY_COPY')
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(source)
    report.update(copy_pin=original,copy_dev_inode=[sealed.st_dev,sealed.st_ino],
                  file_and_directory_fsync=True,natural_source_dev_inode=[first.st_dev,first.st_ino],
                  provider_imported=False)


def project_maps(raw):
    groups,malformed,related=inspect_maps(raw)
    permission_counts={tuple(group['identity']):{} for group in groups}
    paths={tuple(group['identity']):set(group['paths']) for group in groups}
    for line in raw.decode('utf8','surrogateescape').splitlines():
        row=ROW.fullmatch(line)
        if not row:
            continue
        path=row[8].removesuffix(' (deleted)')
        key=(int(row[5],16),int(row[6],16),int(row[7]))
        if key in paths and path in paths[key]:
            permissions=permission_counts[key]
            permissions[row[3]]=permissions.get(row[3],0)+1
    public=[dict(identity=group['identity'],segments=group['segments'],executable_segments=group['executable_segments'],
                 permissions=permission_counts[tuple(group['identity'])],
                 paths=[dict(metadata(path),deleted_segments=item['deleted_segments'])
                        for path,item in group['paths'].items()]) for group in groups]
    return dict(groups=public,malformed=malformed,related_nonprovider_segments=related,
                maps_bytes=len(raw),maps_SHA256=sha256(raw).hexdigest())


def maps_groups():
    with Path('/proc/self/maps').open('rb') as stream:
        return project_maps(stream.read(MAX_MAPS+1))


def select_bound(maps,identity):
    actual=[group for group in maps['groups'] if group['executable_segments']>0]
    if maps['malformed']:return 'MALFORMED_PROVIDER_MAP_ROW'
    if any(sum(group['permissions'].values())!=group['segments']
           or sum(count for permission,count in group['permissions'].items() if permission[2]=='x')
              !=group['executable_segments'] for group in maps['groups']):
        return 'PROVIDER_MAP_PERMISSION_COUNTS_DISAGREE'
    if not actual:return 'BOUND_COPY_NOT_IN_EXECUTABLE_MAPS'
    if len(actual)!=1:return 'COMPETING_EXECUTABLE_PROVIDER_GROUPS'
    if actual[0]['identity']!=identity:return 'EXECUTABLE_MAP_DOES_NOT_IDENTIFY_EXACT_BOUND_FD'
    if any(path['deleted_segments'] for path in actual[0]['paths']):return 'DELETED_BOUND_MAP_PATH'
    return 'ONE_EXECUTABLE_GROUP_MATCHES_BOUND_FD'


def import_original():
    spec=PathFinder.find_spec('torch',sys.path)
    need(spec is not None and spec.origin is not None and 'torch' not in sys.modules,
         'EXACT_CURRENT_TORCH_SPEC_AND_NO_PRIOR_IMPORT')
    origin=Path(spec.origin).resolve(strict=True)
    need(origin==Path(TORCH_ROOT)/'__init__.py','UNCHANGED_PRODUCTION_TORCH_SEARCH_ROUTE')
    with (origin.parent/'version.py').open('rb') as stream:
        version=stream.read(65537)
    need(len(version)<=65536 and sha256(version).hexdigest()==TORCH_VERSION_PIN,'EXACT_TORCH_VERSION_SOURCE')
    import torch
    need(Path(torch.__file__).resolve(strict=True)==origin and torch.__version__=='2.10.0'
         and torch.version.hip=='6.3.26113' and torch.version.cuda is None,'ACTUAL_PRODUCTION_IMPORT_IDENTITY')
    return dict(torch_version=torch.__version__,hip_version=torch.version.hip,torch_source_pin=TORCH_VERSION_PIN)


def final_import(report,deadline,expected):
    from scope import PRIVATE_LEAVES, R
    need(os.environ.get('HOME')==PRIVATE_HOME
         and os.environ.get('MIOPEN_USER_DB_PATH')==PRIVATE_HOME+'/.config/miopen/user-db'
         and os.environ.get('MIOPEN_CUSTOM_CACHE_DIR')==PRIVATE_HOME+'/.cache/miopen','ORIGINAL_APPROVED_COLD_PRIVATE_HOME_ENV')
    for key in ('MIOPEN_SYSTEM_DB_PATH','MIOPEN_LOCKFILE_DIR','MIOPEN_LOCKFILE_PATH','XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_DATA_HOME'):
        need(not os.environ.get(key),'UNKNOWN_OVERRIDE_OF_SHARED_OR_PRIVATE_PATH')
    home=report['private_home_mount']=observe(Path(PRIVATE_HOME))
    need(home['metadata_status']=='RECORDED' and home['at_exact_point'] is True
         and home['same_before_open_after_dev_inode'] is True and home['owner']==os.getuid()
         and home['permissions']==0o700 and home['three_reads']==['RW','RW','RW']
         and home['dev_inode']==expected['host_private_home_dev_inode'],'EXACT_ORIGINAL_PRIVATE_HOME_MOUNT')
    original=report['original_home_mount']=observe(R.parent)
    need(original['fstype']=='overlay' and original['three_reads']==['RW','RO','RO']
         and original['same_before_open_after_dev_inode'] is True
         and all(not (R.parent/part).exists() for part in ('.config','.cache')),'OLD_HOME_READONLY_ROOT_OVERLAY')
    for part in ('.config','.config/miopen','.cache',*PRIVATE_LEAVES):
        state=observe(Path(PRIVATE_HOME)/part)
        need(state['owner']==os.getuid() and state['permissions']==0o700 and state['three_reads']==['RW','RW','RW']
             and state['mount_id']==home['mount_id'] and state['same_before_open_after_dev_inode'] is True,
             'EXACT_NEW_COLD_PRIVATE_STATE_DIRECTORY')
    need(all(not list((Path(PRIVATE_HOME)/leaf).iterdir()) for leaf in PRIVATE_LEAVES),'NEW_PRIVATE_HOME_COLD_LEAVES')
    asset_paths=expected['asset_mounts']
    report['explicit_RO_assets']=[observe(Path(path)) for path in asset_paths]
    need(len(asset_paths)==10 and all(explicit_readonly(obs) for obs in report['explicit_RO_assets']),
         'UNCHANGED_T09_TEN_READONLY_ASSET_BINDS')
    mount=report['canonical_single_file_mount']=observe(EXPECTED)
    need(explicit_readonly(mount) and mount['object_type']=='REGULAR'
         and mount['owner']==os.getuid() and mount['permissions']==0o400
         and mount['dev_inode']==expected['host_private_copy_dev_inode'],'EXACT_NEW_FILE_BIND_READONLY_AND_SAME_INODE')
    fd=os.open(EXPECTED,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK)
    try:
        first,physical=checksum(fd,deadline)
        identity=group_identity(physical)
        report.update(bound_FD_before=first,bound_FD_identity=identity)
        need(first==EXPECTED_PIN and [physical.st_dev,physical.st_ino]==mount['dev_inode'],
             'EXACT_BOUND_PRIVATE_ELF_SHA256_AND_FD')
        report['before_import']=maps_groups()
        need(not report['before_import']['groups'] and not report['before_import']['malformed'],
             'NO_PRIOR_MIOPEN_MAPPING_BEFORE_TORCH_IMPORT')
        report['imported_runtime']=import_original()
        report['after_import']=maps_groups()
        os.lseek(fd,0,os.SEEK_SET)
        after,later=checksum(fd,deadline)
        report['bound_FD_after']=after
        status=select_bound(report['after_import'],identity)
        same=stamp(physical)==stamp(later)==stamp(EXPECTED.lstat()) and after==first
        report.update(mapping_outcome=status,bound_FD_lifetime_stable=same,
                      mapped_copy_identity_verified=same and status=='ONE_EXECUTABLE_GROUP_MATCHES_BOUND_FD')
    finally:
        os.close(fd)
    report['cold_private_leaves_still_empty']=all(not list((Path(PRIVATE_HOME)/leaf).iterdir()) for leaf in PRIVATE_LEAVES)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('copy','observe'));parser.add_argument('--context-json',required=True)
    a=parser.parse_args();context=json.loads(a.context_json)
    report=dict(request=REQUEST,phase=a.mode,passed=False,actual_GPU=0,explicit_device_API_calls=0,
                original_models_loaded=0,actual_conv_calls=0,actual_gold_reads=0,
                MIOpen_actively_loaded_by_probe=False,map_files_open_attempted=False,
                no_retroactive_scope_qualification=True,formal_benchmark_ready=False)
    try:
        require=need
        deadline=context['deadline']
        step=os.environ.get('SLURM_STEP_ID','')
        require(0<deadline-time.time()<=120 and os.environ.get('SLURM_JOB_ID')=='1103606' and step.isdecimal()
                and context['step']=='1103606.'+step and sys.dont_write_bytecode and sys.pycache_prefix is None
                and all(os.environ.get(key)=='' for key in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES')),
                'EXACT_BOUNDED_ZERO_GRES_IMAGE_STEP')
        report['step']=context['step']
        if a.mode=='copy':
            provision(report,deadline)
        else:
            final_import(report,deadline,context)
        report['passed']=True
    except (OSError,ValueError,TypeError,ImportError,RuntimeError,KeyError) as error:
        code=str(error)
        report.update(error_type=type(error).__name__,errno=getattr(error,'errno',None),
                      error_code=code if re.fullmatch('[A-Z0-9_]{1,95}',code) else None)
    raw=json.dumps(report,sort_keys=True,allow_nan=False)
    need(len(raw.encode())<=128*1024,'BOUNDED_PRIVATE_IMPORT_BIND_REPORT')
    print(raw,flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
