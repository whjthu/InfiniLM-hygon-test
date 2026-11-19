"""Minimal proposed Windows copy guard; source-only helper, no network or installer."""
import ctypes
from ctypes import wintypes
import hashlib
import msvcrt
import os
from pathlib import Path
import stat


class INFO(ctypes.Structure):
    _fields_=[('dwFileAttributes',wintypes.DWORD),('ftCreationTime',wintypes.FILETIME),
        ('ftLastAccessTime',wintypes.FILETIME),('ftLastWriteTime',wintypes.FILETIME),
        ('dwVolumeSerialNumber',wintypes.DWORD),('nFileSizeHigh',wintypes.DWORD),('nFileSizeLow',wintypes.DWORD),
        ('nNumberOfLinks',wintypes.DWORD),('nFileIndexHigh',wintypes.DWORD),('nFileIndexLow',wintypes.DWORD)]


def need(ok,reason):
    if not ok:raise ValueError(reason)


def fd_stamp(value):
    # Compare this tuple only within fstat, never fstat.ctime against path.stat.ctime.
    return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns]


def native(fd):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetFileInformationByHandle.argtypes=[wintypes.HANDLE,ctypes.POINTER(INFO)]
    kernel.GetFileInformationByHandle.restype=wintypes.BOOL
    info=INFO()
    if not kernel.GetFileInformationByHandle(wintypes.HANDLE(msvcrt.get_osfhandle(fd)),ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(volume_serial=info.dwVolumeSerialNumber,file_index=(info.nFileIndexHigh<<32)|info.nFileIndexLow,
                bytes=(info.nFileSizeHigh<<32)|info.nFileSizeLow,reparse=bool(info.dwFileAttributes&0x400),
                last_write_100ns=(info.ftLastWriteTime.dwHighDateTime<<32)|info.ftLastWriteTime.dwLowDateTime)


def path_stamp(path):
    path=Path(path).absolute()
    for item in (path,*path.parents):
        value=item.lstat()
        need(not value.st_file_attributes&0x400,'PATH_REPARSE_FORBIDDEN')
    value=path.lstat();need(stat.S_ISREG(value.st_mode),'SOURCE_REGULAR_FILE_REQUIRED')
    return fd_stamp(value)


def validate_handles(before,after,second,expected_bytes):
    need(not before['reparse'] and not after['reparse'] and not second['reparse'],'HANDLE_REPARSE_FORBIDDEN')
    need(before==after,'ORIGINAL_OWNED_HANDLE_CHANGED')
    need(before==second,'SECOND_PATH_HANDLE_FILE_ID_CHANGED')
    need(before['bytes']==expected_bytes,'NATIVE_FILE_SIZE_DIFFERS')


def verified_copy(source,target,expected,check=lambda:None):
    source=Path(source);target=Path(target);path_before=path_stamp(source)
    for parent in (target.parent,*target.parent.parents):
        need(not parent.lstat().st_file_attributes&0x400,'TARGET_PARENT_REPARSE_FORBIDDEN')
    digest=hashlib.sha256();used=0
    with source.open('rb',buffering=0) as incoming,target.open('xb') as output:
        before_stat=os.fstat(incoming.fileno());before_native=native(incoming.fileno())
        need(stat.S_ISREG(before_stat.st_mode) and before_stat.st_size==expected['bytes'],'SOURCE_FD_TYPE_SIZE')
        while True:
            check();block=incoming.read(1024**2)
            if not block:break
            used+=len(block);need(used<=expected['bytes'],'SOURCE_COPY_BYTES_EXCEEDED')
            digest.update(block);need(output.write(block)==len(block),'SHORT_COPY_WRITE')
        output.flush();os.fsync(output.fileno())
        after_stat=os.fstat(incoming.fileno());after_native=native(incoming.fileno())
        need(fd_stamp(before_stat)==fd_stamp(after_stat),'SAME_FD_CHANGED_DURING_COPY')
        need(path_stamp(source)==path_before,'SAME_PATH_CHANGED_DURING_COPY')
        with source.open('rb',buffering=0) as second:
            second_before=os.fstat(second.fileno());second_native=native(second.fileno())
            validate_handles(before_native,after_native,second_native,expected['bytes'])
            need(fd_stamp(second_before)==fd_stamp(os.fstat(second.fileno())),'SECOND_FD_CHANGED')
        need(path_stamp(source)==path_before,'SAME_PATH_CHANGED_AFTER_SECOND_OPEN')
        need(used==expected['bytes'] and digest.hexdigest()==expected['SHA256'],'FULL_SOURCE_SHA_DIFFERS')
        need(os.fstat(output.fileno()).st_size==expected['bytes'],'FULL_TARGET_WRITE_SIZE_DIFFERS')
    return dict(bytes=used,SHA256=digest.hexdigest(),FD_before=fd_stamp(before_stat),FD_after=fd_stamp(after_stat),
                path_before=path_before,path_after=path_stamp(source),native_before=before_native,
                native_after=after_native,second_owned_path_handle=second_native,
                full_SHA_verified=True,same_FD_and_same_path_stable=True,second_handle_fileID_verified=True,
                cross_API_ctime_compared=False,original_modified=False)
