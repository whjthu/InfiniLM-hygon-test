"""Read-only adaptation of qualified Windows handle identity; never calls copy."""
import os
from pathlib import Path
import stat
from windows_copy_identity import fd_stamp, native, path_stamp, validate_handles


def need(value, code):
    if not value:
        raise ValueError(code)


def read_stable(path, size, check, account):
    need(os.name == 'nt', 'WINDOWS_NATIVE_IDENTITY_REQUIRED')
    path = Path(path)
    path_before = path_stamp(path)
    need(path_before[2] == size, 'PATH_SIZE_CHANGED_BEFORE_OPEN')
    parts = []
    used = 0
    # No body bytes are read from either metadata path handle.
    with path.open('rb', buffering=0) as before_path:
        path_fd_before = os.fstat(before_path.fileno())
        path_native_before = native(before_path.fileno())
        need(stat.S_ISREG(path_fd_before.st_mode), 'PATH_HANDLE_REGULAR')
        check()
        with path.open('rb', buffering=0) as held:
            held_stat_before = os.fstat(held.fileno())
            held_native_before = native(held.fileno())
            need(stat.S_ISREG(held_stat_before.st_mode) and held_stat_before.st_size == size,
                 'HELD_REGULAR_SIZE')
            validate_handles(path_native_before, held_native_before, held_native_before, size)
            while used < size:
                check()
                block = held.read(min(65536, size-used))
                need(bool(block), 'HELD_BODY_SHORT')
                account(len(block)); used += len(block); parts.append(block)
            check()
            extra = held.read(1); account(len(extra))
            need(not extra and used == size, 'HELD_BODY_SIZE_CHANGED')
            held_stat_after = os.fstat(held.fileno())
            held_native_after = native(held.fileno())
            need(fd_stamp(held_stat_before) == fd_stamp(held_stat_after), 'SAME_FD_METADATA_CHANGED')
            need(path_stamp(path) == path_before, 'SAME_PATH_METADATA_CHANGED')
            with path.open('rb', buffering=0) as after_path:
                after_stat = os.fstat(after_path.fileno())
                path_native_after = native(after_path.fileno())
                validate_handles(held_native_before, held_native_after, path_native_after, size)
                need(fd_stamp(after_stat) == fd_stamp(os.fstat(after_path.fileno()))
                     and path_native_after == native(after_path.fileno()), 'AFTER_PATH_HANDLE_CHANGED')
            need(path_native_before == held_native_before == held_native_after == path_native_after,
                 'NATIVE_PATH_HELD_IDENTITY_CHANGED')
        need(fd_stamp(path_fd_before) == fd_stamp(os.fstat(before_path.fileno()))
             and path_native_before == native(before_path.fileno()), 'BEFORE_PATH_HANDLE_CHANGED')
    path_after = path_stamp(path)
    need(path_after == path_before, 'FINAL_PATH_METADATA_CHANGED')
    evidence = dict(native_identity=path_native_before, native_path_held_after_equal=True,
        same_path_before_after_equal=True, same_held_FD_before_after_equal=True,
        path_ctime_ns=path_before[4], held_FD_ctime_ns=held_stat_before.st_ctime_ns,
        cross_API_ctime_compared=False, metadata_handle_payload_reads=0, payload_read_once=True)
    return b''.join(parts), evidence
