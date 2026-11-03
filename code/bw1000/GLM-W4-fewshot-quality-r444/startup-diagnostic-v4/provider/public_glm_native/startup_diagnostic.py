"""Startup-only observations; no Native call, helper thread or READY authority.

Only the stage1/rank0 emitter arms the process-global C watchdog after writing
its real owner-ready record. A bounded repeating faulthandler dump has no Python locals;
it cannot show Native C/CUDA stacks or establish GIL causality by itself.
"""
import faulthandler
import json
import os
from pathlib import Path
import threading
import time

_lock = threading.Lock()
_resolved = False
_attempted = False
_armed = False
_stack_file = None  # Keep FD valid until process exit, including dump/cancel races.


def marker(config, event, rank=None, **facts):
    path = Path(config['runtime_directory']).resolve() / config['run']
    path.mkdir(parents=True, exist_ok=True)
    native_id = threading.get_native_id()
    record = dict(event=event, stage=config['stage'], rank=rank,
        PID=os.getpid(), thread_id=threading.get_ident(), native_thread_id=native_id,
        thread_name=threading.current_thread().name, realtime=time.time(),
        monotonic=time.monotonic(), original_epoch=config['gpu_start_epoch'],
        run=config['run'], actual_step=config['actual_step'],
        source_manifest_SHA256=config['source_manifest_SHA256'], **facts)
    raw = (json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n').encode()
    if len(raw) > 4096:
        raise ValueError('Startup marker exceeded fixed small-record bound')
    name = 'startup-%d-%s-%d-%d.jsonl' % (
        config['stage'], 'leader' if rank is None else rank, os.getpid(), native_id)
    fd = os.open(path / name, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, raw) != len(raw):
            raise OSError('Incomplete startup marker write')
    finally:
        os.close(fd)


def arm_after_owner_ready(config, rank):
    global _attempted, _armed, _stack_file
    if (config['stage'], rank) != (1, 0):
        return
    with _lock:
        if _resolved or _attempted:
            marker(config, 'watchdog.not_armed', rank, already_resolved=_resolved,
                   already_attempted=_attempted)
            return
        _attempted = True
        path = Path(config['runtime_directory']).resolve() / config['run']
        _stack_file = (path / ('startup-stack-stage1-PID%d.txt' % os.getpid())).open('xb', buffering=0)
        marker(config, 'watchdog.arm.before', rank, timeout_seconds=20, repeat=True,
               exit=False, Python_locals=False, Native_stack=False)
        faulthandler.dump_traceback_later(20, repeat=True, file=_stack_file, exit=False)
        _armed = True
        marker(config, 'watchdog.arm.return', rank)


def resolve(config, reason):
    """Serialize publish-before-arm race; never rearm after startup resolves."""
    global _resolved, _armed
    if config['stage'] != 1:
        return
    with _lock:
        _resolved = True
        if _armed:
            faulthandler.cancel_dump_traceback_later()
            _armed = False
        marker(config, 'watchdog.resolved', reason=reason, attempted=_attempted)
