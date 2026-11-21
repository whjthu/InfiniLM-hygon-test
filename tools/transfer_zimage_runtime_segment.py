"""Relay one pinned Z-Image runtime range through an existing authenticated transport."""

from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import stage_internvl_weight_segment as stage
from stream_internvl_public_segment import REMOTE_COMMAND


REPO = Path(__file__).resolve().parents[1]
PACKAGE = Path('code/bw1000/training-zimage-publisher-runtime-transfer-manifest-v1')
MANIFEST_SHA = 'fd2c299b1d6dccc36b021a22f7f3d285f926d19cb0e631c3d865c5e5212ca172'
PLANNER_SHA = '3ca29814cea41de9bcddc912eb94663e950b6e7838e279efa81e2abecf132ea6'
STAGE_SHA = 'dd578f3d92f287225347e0cbcfa181e6d6d3788d48d734a6d9a5be1725700d98'
TARGET = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-zimage-runtime-assets-v1'
SEGMENT_BYTES = 32 * 1024**2


def load_pins(repo=REPO):
    """Revalidate the committed 20-file manifest against its independent evidence."""
    raw = (repo / PACKAGE / 'RUNTIME_FETCH_MANIFEST.json').read_bytes()
    planner = repo / PACKAGE / 'manifest.py'
    code = planner.read_bytes()
    stage.need(sha256(raw).hexdigest() == MANIFEST_SHA and sha256(code).hexdigest() == PLANNER_SHA,
               'frozen Z-Image publisher components differ')
    namespace = {'__name__':'pinned_zimage_publisher_planner','__file__':str(planner)}
    exec(compile(code,str(planner),'exec'),namespace)
    return namespace['validate'](json.loads(raw),repo)


def asset_spec(row):
    return {
        'file_id':row['asset_id'], 'filename':row['filename'], 'bytes':row['bytes'],
        'SHA256':row['SHA256'], 'segment_bytes':SEGMENT_BYTES, 'kind':row['kind'],
        'target_relative_path':row['target_relative_path'], 'publisher_URL':row['publisher_URL'],
    }


def source_range(spec, offset, deadline, *, open_url=urlopen):
    count = stage.segment_length(spec,offset)
    end = offset + count - 1
    data = bytearray()
    for attempt in range(4):
        start = offset + len(data)
        remaining = deadline - time.monotonic()
        stage.need(remaining > 0,'publisher time budget expired')
        request = Request(spec['publisher_URL'],headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        try:
            with open_url(request,timeout=min(20,remaining)) as response:
                uri = urlsplit(response.geturl())
                stage.need(uri.scheme == 'https' and not uri.username and not uri.password and
                           (uri.hostname == 'huggingface.co' or (uri.hostname or '').endswith(('.hf.co','.huggingface.co'))),
                           'unexpected official publisher redirect')
                complete = start == 0 and count == spec['bytes'] and response.status == 200
                partial = (response.status == 206 and
                           response.headers.get('Content-Range') == f'bytes {start}-{end}/{spec["bytes"]}')
                stage.need((complete or partial) and response.headers.get('Content-Length') == str(end-start+1)
                           and response.headers.get('Content-Encoding','identity').lower() == 'identity',
                           'publisher did not return the exact immutable range')
                while len(data) < count:
                    stage.need(time.monotonic() < deadline,'publisher time budget expired')
                    chunk = response.read1(min(1024**2,count-len(data)))
                    stage.need(0 < len(chunk) <= count-len(data),'publisher range was short or oversized')
                    data.extend(chunk)
                stage.need(response.read(1) == b'' and time.monotonic() < deadline,'publisher range overran its bounds')
            if offset == 0 and count == spec['bytes']:
                stage.need(sha256(data).hexdigest() == spec['SHA256'],'complete publisher file SHA differs')
            return data
        except (TimeoutError,ConnectionError,URLError) as error:
            cause = error.reason if isinstance(error,URLError) else error
            if not isinstance(cause,(TimeoutError,ConnectionError)) or attempt == 3 or len(data) == count:
                raise
    raise RuntimeError('publisher attempt budget exhausted')


def receive_asset(backend, target, request, spec, operation, offset, checksum, stream):
    """Shared receiver body. The production program embeds this with the frozen stage source."""
    asset = target / ('asset-%02d' % spec['file_id'])
    if operation == 'inspect' and not (asset.exists() or asset.is_symlink()):
        result = dict(state='ABSENT',file_id=spec['file_id'],confirmed_bytes=0)
    elif operation == 'initialize':
        target.mkdir(mode=0o700,exist_ok=True)
        backend['initialize'](asset,spec)
        result = backend['inspect'](asset,spec)
    elif operation == 'append':
        result = backend['append'](asset,spec,offset,checksum,stream)
    elif operation == 'seal':
        result = backend['seal'](asset,spec)
    elif operation == 'inspect':
        lock = asset / 'stage.lock'
        if not lock.is_file() or lock.is_symlink():
            raise ValueError('original stage lock is missing or aliased')
        result = backend['inspect'](asset,spec)
    else:
        raise ValueError('unrecognized transfer operation')
    return dict(request=request,operation=operation,**result)


def target_input(request, spec, operation, offset, checksum, timeout):
    stage_source = Path(stage.__file__).read_bytes()
    operations = {'inspect','initialize','append','seal'}
    stage.need(sha256(stage_source).hexdigest() == STAGE_SHA and operation in operations and 0 < timeout <= 180,
               'frozen receiver or operation budget differs')
    header = ('import sys\n'
              '__name__="zimage_stage_receiver"\n'
              '__file__="/tmp/zimage_stage_receiver.py"\n').encode()
    parameters = repr((TARGET, request, spec, operation, offset, checksum, timeout))
    epilogue = f'''
target, request, spec, operation, offset, checksum, timeout = {parameters}
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('receiver deadline')))
signal.setitimer(signal.ITIMER_REAL, timeout)
result = receive_asset(globals(), Path(target), request, spec, operation, offset, checksum, sys.stdin.buffer)
signal.setitimer(signal.ITIMER_REAL, 0)
print(json.dumps(result,allow_nan=False))
'''
    program = header + stage_source + b'\n' + inspect.getsource(receive_asset).encode() + epilogue.encode()
    stage.need(len(program) <= 32768,'sealed receiver program exceeds the original finite channel envelope')
    return struct.pack('<Q',len(program)) + program


def save(path, value):
    # Only bounded JSON receipts are stored on the caller.
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def bound_spec(asset_id, repo):
    stage.need(type(asset_id) is int,'an exact integer asset ID is required')
    rows = [row for row in load_pins(repo)['assets'] if row['asset_id'] == asset_id]
    stage.need(len(rows) == 1,'unrecognized immutable asset ID')
    return asset_spec(rows[0])


def exchange_target(exchange, request, spec, operation, remaining, offset=None, checksum=None, data=b''):
    program = target_input(request,spec,operation,offset,checksum,remaining)
    try:
        response = exchange(REMOTE_COMMAND,program+data,remaining)
    except (OSError,subprocess.SubprocessError) as error:
        raise RuntimeError(f'target {operation} unresolved; inspect original target prefix before further action') from error
    if response.returncode != 0:
        raise RuntimeError(f'target {operation} returned {response.returncode}; no replay without target inspection')
    stage.need(isinstance(response.stdout,bytes) and 0 < len(response.stdout) <= 16384,'unbounded target receipt')
    value = json.loads(response.stdout)
    stage.need((value['request'],value['file_id'],value['operation']) == (request,spec['file_id'],operation),
               'target receipt belongs to a different request')
    return value


def inspect_target_asset(asset_id, exchange, *, timeout=30, repo=REPO):
    """Read the authoritative durable prefix without opening the publisher or appending."""
    stage.need(callable(exchange) and type(timeout) is int and 1 <= timeout <= 180,
               'invalid read-only target inspection')
    spec = bound_spec(asset_id,repo)
    result = exchange_target(exchange,uuid4().hex,spec,'inspect',timeout)
    offset = result['confirmed_bytes']
    stage.need(type(offset) is int and 0 <= offset <= spec['bytes']
               and (offset == spec['bytes'] or offset % SEGMENT_BYTES == 0)
               and (result.get('whole_file_qualified_at_seal') is not True or offset == spec['bytes']),
               'unexpected durable target prefix')
    return result


def relay(asset_id, expected_offset, publication, log_root, exchange, *, budget=150, repo=REPO):
    """One explicit append or seal, after inspecting the exact authoritative target prefix.

    exchange(argv, body, timeout) must use the existing authorized project route
    and return a subprocess.CompletedProcess. This module opens no SSH itself.
    """
    stage.need(type(expected_offset) is int
               and isinstance(publication,str) and re.fullmatch(r'[0-9a-f]{40}',publication)
               and type(budget) is int and 30 <= budget <= 420 and callable(exchange), 'invalid one-range request')
    spec = bound_spec(asset_id,repo)
    stage.need(0 <= expected_offset <= spec['bytes'] and
               (expected_offset == spec['bytes'] or expected_offset % SEGMENT_BYTES == 0),'invalid confirmed offset')
    deadline = time.monotonic()+budget
    directory = log_root / uuid4().hex
    directory.mkdir(mode=0o700)
    request = directory.name

    def remote(operation, offset=None, checksum=None, data=b''):
        remaining = min(180,deadline-time.monotonic())
        if operation != 'inspect':
            save(directory/f'{operation}-intent.json',dict(request=request,asset_id=asset_id,operation=operation,
                 offset=offset,segment_SHA256=checksum,publication=publication))
        return exchange_target(exchange,request,spec,operation,remaining,offset,checksum,data)

    before = remote('inspect')
    save(directory/'before.json',before)
    stage.need(type(before.get('confirmed_bytes')) is int and before['confirmed_bytes'] == expected_offset,
               'target durable prefix differs from the requested offset')
    if before.get('whole_file_qualified_at_seal') is True:
        stage.need(expected_offset == spec['bytes'],'sealed target receipt is incomplete')
        return dict(action='already_sealed',request=request,target=before)

    if expected_offset == spec['bytes']:
        result = remote('seal')
        stage.need(result.get('whole_file_qualified') is True and
                   (result['bytes'],result['SHA256']) == (spec['bytes'],spec['SHA256']),'target whole-file seal differs')
        action = 'sealed'
    else:
        data = source_range(spec,expected_offset,deadline)
        checksum = sha256(data).hexdigest()
        save(directory/'publisher-range.json',dict(asset_id=asset_id,offset=expected_offset,
             bytes=len(data),SHA256=checksum,publication=publication))
        if before.get('state') == 'ABSENT':
            initialized = remote('initialize')
            stage.need(initialized['confirmed_bytes'] == 0 and initialized['whole_file_qualified_at_seal'] is False,
                       'initialized target scope is not empty')
        result = remote('append',expected_offset,checksum,data)
        stage.need((result['offset'],result['raw_bytes'],result['segment_SHA256'],result['confirmed_bytes']) ==
                   (expected_offset,len(data),checksum,expected_offset+len(data)) and result['whole_file_qualified'] is False,
                   'target durable append receipt differs')
        action = 'segment_committed'
    save(directory/'outcome.json',result)
    return dict(action=action,request=request,target=result)
