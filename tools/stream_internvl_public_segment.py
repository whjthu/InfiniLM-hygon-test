"""Relay the exact public InternVL weight ranges to the existing durable staging receiver."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import stage_internvl_weight_segment as stage


REVISION = '741a7d03020411e666c6109218ab71e08151ef86'
BASE = f'https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/resolve/{REVISION}/'
STAGE_SHA256 = 'dd578f3d92f287225347e0cbcfa181e6d6d3788d48d734a6d9a5be1725700d98'
BOOTSTRAP = (
    'import sys\n'
    'n=int.from_bytes(sys.stdin.buffer.read(8),"little")\n'
    'if not 0<n<=32768: raise ValueError("invalid receiver program size")\n'
    'source=sys.stdin.buffer.read(n)\n'
    'if len(source)!=n: raise ValueError("short receiver program")\n'
    'exec(compile(source,"internvl-public-receiver","exec"))\n'
)
REMOTE_COMMAND = ('/usr/local/python3.12/bin/python3', '-I', '-S', '-B', '-c', BOOTSTRAP)
OPERATIONS = {'inspect', 'initialize', 'append', 'seal'}


def public_range(spec, offset, deadline, *, open_url=urlopen):
    count = stage.segment_length(spec, offset)
    end = offset + count - 1
    raw = bytearray()
    for attempt in range(4):
        start = offset + len(raw)
        request = Request(BASE + spec['filename'], headers={
            'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
        })
        remaining = deadline - time.monotonic()
        stage.need(remaining > 0, 'public range budget expired')
        try:
            with open_url(request, timeout=min(20, remaining)) as response:
                verify_response(response, spec, start, end)
                while len(raw) < count:
                    stage.need(time.monotonic() < deadline, 'public range budget expired')
                    block = response.read1(min(1024 * 1024, count - len(raw)))
                    stage.need(bool(block), 'short public range')
                    raw.extend(block)
                stage.need(response.read(1) == b'' and time.monotonic() < deadline,
                           'overlong public range or expired budget')
                return raw
        except (TimeoutError, ConnectionError, URLError) as exc:
            cause = exc.reason if isinstance(exc, URLError) else exc
            if (not isinstance(cause, (TimeoutError, ConnectionError)) or attempt == 3
                    or len(raw) == count or time.monotonic() >= deadline):
                raise


def verify_response(response, spec, start, end):
    endpoint = urlsplit(response.geturl())
    stage.need(endpoint.scheme == 'https' and not endpoint.username and
               (endpoint.hostname == 'huggingface.co' or
                (endpoint.hostname or '').endswith(('.cdn.hf.co', '.hf.co'))),
               'unexpected official range endpoint')
    stage.need(response.status == 206 and
               response.headers.get('Content-Range') == f'bytes {start}-{end}/{spec["bytes"]}' and
               response.headers.get('Content-Length') == str(end - start + 1) and
               response.headers.get('Content-Encoding', 'identity').lower() == 'identity',
               'publisher did not return the exact immutable range')


def make_program(spec, operation, offset, digest, allowance):
    stage.need(operation in OPERATIONS and 0 < allowance <= 180, 'unsupported receiver operation or budget')
    path = Path(stage.__file__)
    source = path.read_bytes()
    stage.need(hashlib.sha256(source).hexdigest() == STAGE_SHA256, 'receiver source differs')
    context = repr((spec, operation, offset, digest, allowance))
    body = f'''
import signal, sys
spec, operation, offset, digest, allowance = {context}
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('receiver deadline')))
signal.setitimer(signal.ITIMER_REAL, allowance)
root = TARGET / f'shard{{spec["file_id"]:02}}'
if operation == "inspect" and not (root.exists() or root.is_symlink()):
    result = dict(state='ABSENT', file_id=spec['file_id'])
elif operation == "initialize":
    TARGET.mkdir(mode=0o700, exist_ok=True)
    initialize(root, spec)
    result = inspect(root, spec)
elif operation == "append":
    result = append(root, spec, offset, digest, sys.stdin.buffer)
elif operation == "seal":
    result = seal(root, spec)
else:
    result = inspect(root, spec)
signal.setitimer(signal.ITIMER_REAL, 0)
print(json.dumps(result, allow_nan=False))
'''
    program = b"__name__='internvl_staging_receiver'\n__file__='/tmp/internvl_receiver.py'\n" + source + body.encode()
    stage.need(len(program) <= 32768, 'receiver program exceeds bound')
    return struct.pack('<Q', len(program)) + program


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def run_segment(file_id, publication, log_root, start_target, *, budget=150, expected_offset=None):
    """Obtain and confirm one new public segment. Resume from authoritative target state."""
    stage.need(type(budget) is int and 30 <= budget <= 420, 'segment budget outside bounds')
    spec = stage.shard_spec(file_id)
    deadline = time.monotonic() + budget
    request = uuid4().hex
    directory = log_root / request
    directory.mkdir(mode=0o700)

    def remote(operation, *, offset=None, digest=None, data=b''):
        remaining = min(180, deadline - time.monotonic())
        stage.need(remaining > 0, 'relay budget expired before target operation')
        prefix = make_program(spec, operation, offset, digest, remaining)
        if operation != 'inspect':
            save(directory / f'{operation}-intent.json', dict(
                request=request, publication=publication, file_id=file_id, operation=operation,
                offset=offset, digest=digest, official_revision=REVISION, epoch=time.time()))
        child = start_target(REMOTE_COMMAND)
        try:
            out, err = child.communicate(prefix + data, timeout=max(0.01, remaining))
        except (subprocess.TimeoutExpired, OSError):
            child.kill()
            child.communicate(timeout=5)
            raise
        if child.returncode:
            (directory / f'{operation}-private.stderr').write_bytes(err[-8192:])
            raise RuntimeError(f'target {operation} returned {child.returncode}. Inspect before next mutation.')
        stage.need(0 < len(out) <= 16384, 'target receipt outside size bound')
        return json.loads(out)

    before = remote('inspect')
    save(directory / 'before.json', before)
    offset = 0 if before.get('state') == 'ABSENT' else before['confirmed_bytes']
    stage.need(expected_offset is None or offset == expected_offset,
               'target durable offset differs from the published continuation')
    if before.get('whole_file_qualified_at_seal') is True:
        return dict(request=request, action='already_sealed', target=before)
    if offset == spec['bytes']:
        result = remote('seal')
        stage.need(result.get('whole_file_qualified') is True and result['SHA256'] == spec['SHA256'],
                   'whole-shard seal does not match the frozen original')
        save(directory / 'outcome.json', result)
        return dict(request=request, action='sealed', target=result)
    source_begin = time.monotonic()
    raw = public_range(spec, offset, deadline)
    digest = hashlib.sha256(raw).hexdigest()
    source_wall = time.monotonic() - source_begin
    save(directory / 'publisher-range.json', dict(
        file_id=file_id, offset=offset, raw_bytes=len(raw), segment_SHA256=digest,
        official_revision=REVISION, wall_seconds=source_wall))
    if before.get('state') == 'ABSENT':
        initialized = remote('initialize')
        stage.need(initialized['confirmed_bytes'] == 0 and initialized['whole_file_qualified_at_seal'] is False,
                   'new target checkpoint has an unexpected state')
    target_begin = time.monotonic()
    result = remote('append', offset=offset, digest=digest, data=raw)
    stage.need(result['file_id'] == file_id and result['offset'] == offset and
               result['segment_SHA256'] == digest and result['raw_bytes'] == len(raw) and
               result['confirmed_bytes'] == offset + len(raw) and result['whole_file_qualified'] is False,
               'target durable-range receipt disagrees with the source')
    save(directory / 'outcome.json', result)
    return dict(request=request, action='segment_committed', target=result,
                source_wall_seconds=source_wall, target_wall_seconds=time.monotonic() - target_begin)


def continue_shard(file_id, publication, directory, start_target, *, start_offset,
                   max_range_count, window_seconds, range_budget=420, report=print):
    """Run one exclusive local continuation window, retaining each committed range."""
    spec = stage.shard_spec(file_id)
    stage.need(type(start_offset) is int and 0 <= start_offset <= spec['bytes'] and
               (start_offset == spec['bytes'] or start_offset % spec['segment_bytes'] == 0),
               'invalid continuation offset')
    stage.need(type(max_range_count) is int and 1 <= max_range_count <= 200 and
               type(window_seconds) is int and 420 <= window_seconds <= 21600 and
               type(range_budget) is int and 30 <= range_budget <= 420, 'invalid continuation limits')
    directory.mkdir(mode=0o700)
    save(directory / 'window-intent.json', dict(
        file_id=file_id, publication=publication, started_epoch=time.time(), start_offset=start_offset,
        window_seconds=window_seconds, range_budget=range_budget, max_range_count=max_range_count,
        transport='existing_authenticated_project_route'))
    deadline = time.monotonic() + window_seconds
    offset, committed = start_offset, 0
    sealed = False
    state = 'WINDOW_BUDGET_OR_RANGE_LIMIT'
    try:
        while offset == spec['bytes'] or committed < max_range_count:
            budget = min(range_budget, int(deadline - time.monotonic()))
            if budget < 30:
                break
            entry = run_segment(file_id, publication, directory, start_target,
                                budget=budget, expected_offset=offset)
            if entry['action'] in ('sealed', 'already_sealed'):
                sealed = True
                state = 'WHOLE_SHARD_VERIFIED'
                report(entry)
                break
            stage.need(entry['action'] == 'segment_committed' and entry['target']['offset'] == offset,
                       'unexpected continuation receipt')
            offset = entry['target']['confirmed_bytes']
            committed += 1
            save(directory / f'committed-{committed:04}.json', entry)
            report(entry)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        state = 'STOPPED_REQUIRES_FRESH_TARGET_RECONCILIATION'
        raise
    finally:
        save(directory / 'window-terminal.json', dict(
            state=state, file_id=file_id, start_offset=start_offset,
            last_acknowledged_offset=offset, committed_ranges=committed, whole_file_qualified=sealed))
