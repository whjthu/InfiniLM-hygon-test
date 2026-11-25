"""Attempt the kernel-owned self map_files link only; never hash its pathname fallback."""
from hashlib import sha256
import errno
import os
import stat
import time

from maps_probe import EXPECTED, EXPECTED_PIN, ROW, group_identity, inspect_maps, metadata, stamp


def choose(raw):
    groups,malformed,related=inspect_maps(raw)
    summary=[dict(identity=group['identity'],total_segments=group['segments'],
                  executable_segments=group['executable_segments'],
                  paths=[dict(metadata(path),deleted_segments=item['deleted_segments'])
                         for path,item in group['paths'].items()]) for group in groups]
    facts=dict(groups=summary,malformed=malformed,related_non_provider_segments=related)
    executable=[group for group in groups if group['executable_segments']>0]
    def outcome(code):
        return None,dict(facts,status=code)
    if malformed:return outcome('MALFORMED_PROVIDER_MAPPING_ROWS')
    if not groups:return outcome('NO_MIOPEN_MAPPING')
    if not executable:return outcome('NO_EXECUTABLE_PROVIDER_MAPPING')
    if len(executable)!=1:return outcome('AMBIGUOUS_MULTIPLE_EXECUTABLE_GROUPS')
    chosen=executable[0]
    if chosen['executable_segments']!=1:return outcome('EXPECTED_ONE_EXECUTABLE_SEGMENT')
    if set(chosen['paths'])!={EXPECTED.as_posix()} or any(item['deleted_segments'] for item in chosen['paths'].values()):
        return outcome('ORIGINAL_MAP_PROVIDER_PATH_DIFFERS_OR_DELETED')
    identity=chosen['identity']
    for line in raw.decode('utf8','surrogateescape').splitlines():
        row=ROW.fullmatch(line)
        if row and 'x' in row[3] and [int(row[5],16),int(row[6],16),int(row[7])]==identity and row[8]==EXPECTED.as_posix():
            # Kernel key stays internal and is not returned by the public projector.
            leaf=f'{row[1]}-{row[2]}'
            return (leaf,identity),dict(facts,status='ONE_EXPECTED_EXECUTABLE_SEGMENT')
    return outcome('EXECUTABLE_SEGMENT_NOT_FOUND_ON_RECHECK')


def error_summary(error,phase):
    permission=error.errno in (errno.EPERM,errno.EACCES)
    return dict(status='MAP_FILES_PERMISSION_DENIED' if permission else 'MAP_FILES_OPEN_OR_READ_ERROR',
                phase=phase,errno=error.errno,error_type=type(error).__name__)


def read_exact(candidate,read_maps,deadline):
    leaf,expected=candidate
    phase='DIRECTORY_OPEN'
    out=dict(status='NOT_OBSERVED',method='PROC_SELF_MAP_FILES_ONLY',opened_descriptor=False,
             requested_file_identity=expected,matched_pinned_SHA256=False,
             canonical_path_fallback_used=False,executable_virtual_addresses_exposed=False)
    try:
        root=os.open('/proc/self/map_files',os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        try:
            phase='EXECUTABLE_LINK_OPEN'
            # The procfs entry is a kernel-generated magic link. Following just
            # this validated current-process leaf is the operation being tested.
            fd=os.open(leaf,os.O_RDONLY|os.O_CLOEXEC|os.O_NONBLOCK,dir_fd=root)
            out['opened_descriptor']=True
            try:
                phase='BACKING_FSTAT'
                before=os.fstat(fd)
                actual=group_identity(before)
                out.update(descriptor_identity=actual,descriptor_bytes=before.st_size)
                if (actual!=expected or not stat.S_ISREG(before.st_mode)
                    or before.st_size!=EXPECTED_PIN['bytes']):
                    return dict(out,status='MAP_FILES_DESCRIPTOR_IDENTITY_OR_EXTENT_DIFFERS')
                digest=sha256()
                count=0
                phase='BOUNDED_SHA256'
                while count<before.st_size:
                    if time.monotonic()>=deadline:
                        return dict(out,status='MAP_FILES_HASH_DEADLINE_EXHAUSTED',bytes_hashed=count)
                    block=os.read(fd,min(1<<20,before.st_size-count))
                    if not block:
                        return dict(out,status='MAP_FILES_SHORT_READ',bytes_hashed=count)
                    count+=len(block)
                    digest.update(block)
                trailing=os.read(fd,1)
                phase='STABLE_DESCRIPTOR_AND_SEGMENT_READBACK'
                stable=not trailing and stamp(os.fstat(fd))==stamp(before)
                pin=dict(bytes=count,SHA256=digest.hexdigest())
                try:
                    rechecked,_=choose(read_maps())
                except ValueError:
                    return dict(out,status='MAP_FILES_MAPPING_READBACK_UNQUALIFIED',pin=pin,fd_lifetime_stable=stable)
                same_segment=rechecked==candidate
                out.update(pin=pin,fd_lifetime_stable=stable,same_live_exec_segment=same_segment)
                if not stable or not same_segment:
                    return dict(out,status='MAP_FILES_DESCRIPTOR_OR_MAPPING_CHANGED')
                if pin!=EXPECTED_PIN:
                    return dict(out,status='MAP_FILES_ELF_BYTES_DIFFER')
                return dict(out,status='EXACT_MAPPED_BACKING_ELF_SHA256',matched_pinned_SHA256=True)
            finally:
                os.close(fd)
        finally:
            os.close(root)
    except OSError as error:
        return dict(out,**error_summary(error,phase))


def observe(read_maps,deadline):
    selected,projected=choose(read_maps())
    projected['actual_map_files_open_attempted']=selected is not None
    projected['map_files']=read_exact(selected,read_maps,deadline) if selected is not None else None
    return projected
