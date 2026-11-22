"""Send and stage bounded InternVL shard segments with durable offset receipts."""

import argparse
import ast
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import struct
import sys
from tempfile import NamedTemporaryFile
import time
import zlib


REPO = Path(__file__).resolve().parents[1]
ORIGINAL_PLAN = REPO/'code/bw1000/training-internvl-three-shard-zlib-24h-v2/PLAN.json'
ORIGINAL_PLAN_SHA256 = 'ecf53ec728af0c106a92e6c5ff98b89919c157f7db67dfdaeaff0fb4144482dd'
ORIGINAL_PROTOCOL_SHA256 = 'aad004a277cebaf252d31d33460d24266bcb04810f8d01f2ed369d939ea34bdb'
SOURCE_RECEIPT = REPO/'results/bw1000/2026-11-12/training-internvl-original-four-shard-full-SHA-and-index841-qualified.json'
SOURCE_RECEIPT_SHA256 = '0c10ff71c842d683082d619dda9e992902c0025c77c05d5c785bf8c86258bc70'
BLOCK_BYTES = 1024**2
TARGET = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-internvl-exact-shard-segments-v1')
SEGMENT_BYTES = 32*1024**2
SEGMENT_HEADER = struct.Struct('<8sIQQ32s')
SEGMENT_MAGIC = b'IVLSEG1\n'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def shard_row(file_id):
    raw = ORIGINAL_PLAN.read_bytes()
    need(sha256(raw).hexdigest() == ORIGINAL_PLAN_SHA256, 'pinned original shard identities differ')
    rows = json.loads(raw)['files']
    need({row['file_id'] for row in rows} == {2, 3, 4}, 'exact missing three shards required')
    [row] = [row for row in rows if row['file_id'] == file_id]
    return row


def shard_spec(file_id):
    row = shard_row(file_id)
    leaf = PurePosixPath(row['name']).name
    need(leaf == f'model-0000{file_id}-of-00004.safetensors', 'original shard filename differs')
    return dict(file_id=file_id, filename=leaf, bytes=row['bytes'], SHA256=row['SHA256'], segment_bytes=SEGMENT_BYTES)


def qualified_source(spec):
    raw = SOURCE_RECEIPT.read_bytes()
    need(sha256(raw).hexdigest() == SOURCE_RECEIPT_SHA256, 'original source qualification differs')
    record = json.loads(raw)['independent_root_reader']
    evidence = record['shards'][spec['filename']]
    need(evidence['SHA256'] == spec['SHA256'] and evidence['bytes'] == spec['bytes'],
         'original source receipt identifies another file')
    path = Path(shard_row(spec['file_id'])['source'])
    need(path.name == spec['filename'] and path.resolve(strict=True) == path, 'source path is not the original canonical file')
    return path,evidence


def sync_dir(root):
    if os.name == 'posix':
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def locked(root):
    need(root.is_dir() and root.resolve(strict=True) == root, 'shard stage root is missing or aliased')
    path = root/'stage.lock'
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    with os.fdopen(os.open(path, flags, 0o600), 'r+b') as file:
        need(stat.S_ISREG(os.fstat(file.fileno()).st_mode) and not path.is_symlink(), 'unexpected stage lock')
        if os.name == 'posix':
            import fcntl
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            if file.read(1) == b'':
                file.write(b'\0')
                file.flush()
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        yield


def checkpoint(root, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()
    need(len(raw) <= 131072, 'bounded stage checkpoint exceeded')
    with NamedTemporaryFile(dir=root, prefix='checkpoint-', delete=False) as stream:
        path = Path(stream.name)
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            path.replace(root/'checkpoint.json')
            sync_dir(root)
        finally:
            path.unlink(missing_ok=True)


@contextmanager
def payload_open(root, flags):
    path = root/'payload.partial'
    need(not path.is_symlink(), 'aliased staging payload')
    fd = os.open(path, flags | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'r+b' if flags & os.O_RDWR else 'rb') as file:
        need(stat.S_ISREG(os.fstat(file.fileno()).st_mode), 'expected a regular staging payload')
        yield file


def load(root, spec):
    p = root/'checkpoint.json'
    need(not p.is_symlink(), 'aliased checkpoint')
    with p.open('rb') as stream:
        raw = stream.read(131073)
    need(0 < len(raw) <= 131072, 'bounded checkpoint required')
    state = json.loads(raw)
    need(state['spec'] == spec, 'checkpoint belongs to another original shard')
    confirmed = 0
    for entry in state['segments']:
        expected = min(spec['segment_bytes'],spec['bytes']-confirmed)
        need(expected > 0 and entry['offset'] == confirmed and entry['bytes'] == expected
             and re.fullmatch('[0-9a-f]{64}',entry['SHA256']), 'committed segment ledger is discontinuous')
        confirmed += expected
    need(state['confirmed_bytes'] == confirmed and (not state['sealed'] or confirmed == spec['bytes']),
         'confirmed prefix or sealed stage differs')
    return state


def check_file(file, state):
    metadata = os.fstat(file.fileno())
    need([metadata.st_dev,metadata.st_ino] == state['file'] and metadata.st_size >= state['confirmed_bytes'],
         'staging payload identity changed or committed bytes were lost')
    if state['segments']:
        last = state['segments'][-1]
        file.seek(last['offset'])
        need(sha256(file.read(last['bytes'])).hexdigest() == last['SHA256'], 'last committed segment changed')
    return metadata.st_size


def initialize(root, spec):
    need(root.parent.resolve(strict=True) == root.parent, 'aliased parent stage')
    root.mkdir(mode=0o700)
    with locked(root), payload_open(root, os.O_RDWR | os.O_CREAT | os.O_EXCL) as file:
        s = os.fstat(file.fileno())
        os.fsync(file.fileno())
        checkpoint(root, dict(spec=spec, file=[s.st_dev,s.st_ino], segments=[], confirmed_bytes=0, sealed=False))


def append(root, spec, offset, digest, stream, *, framed=False, envelope=False):
    need((envelope and digest is None) or (isinstance(digest,str) and re.fullmatch('[a-f0-9]{64}',digest)),
         'original segment SHA required')
    with locked(root):
        state = load(root,spec)
        need(not state['sealed'] and offset == state['confirmed_bytes'] and offset < spec['bytes'],
             'append must start at the confirmed prefix; do not replay a committed segment')
        count = min(spec['segment_bytes'],spec['bytes']-offset)
        if envelope:
            header = stream.read(SEGMENT_HEADER.size)
            need(len(header) == SEGMENT_HEADER.size, 'missing original segment envelope')
            magic,file_id,declared,nbytes,raw_digest = SEGMENT_HEADER.unpack(header)
            need((magic,file_id,declared,nbytes) == (SEGMENT_MAGIC,spec['file_id'],offset,count),
                 'original segment envelope scope differs')
            digest = raw_digest.hex()
        if framed or envelope:
            stream = framed_segment(spec,offset,stream)
        raw = stream.read(count+1)
        need(len(raw) == count and sha256(raw).hexdigest() == digest, 'short, overlong or different segment')
        with payload_open(root, os.O_RDWR) as file:
            previous = check_file(file,state)
            if previous > offset:
                # Discard only bytes beyond the last durable checkpoint in this owned staging file.
                file.truncate(offset)
            file.seek(offset)
            need(file.write(raw) == count, 'short staged payload write')
            file.flush()
            os.fsync(file.fileno())
        state['segments'].append(dict(offset=offset, bytes=count, SHA256=digest))
        state['confirmed_bytes'] = offset+count
        checkpoint(root,state)
        return dict(file_id=spec['file_id'],offset=offset,raw_bytes=count,segment_SHA256=digest,
                    confirmed_bytes=state['confirmed_bytes'],segment_count=len(state['segments']),
                    recovered_uncommitted_bytes=previous-offset,whole_file_qualified=False)


def original_protocol():
    path = ORIGINAL_PLAN.with_name('protocol.py')
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == ORIGINAL_PROTOCOL_SHA256, 'original zlib framing parser differs')
    names = {'unpack','pack','raw_length','decompress_bounded'}
    parsed = ast.parse(raw, filename=str(path))
    keep = [node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in names]
    need({node.name for node in keep} == names, 'original independent compression blocks differ')
    scope = dict(P={'block_bytes':BLOCK_BYTES}, HEADER=struct.Struct('<8sIIQQII32s32s24s'), MAGIC=b'IVLZB01\n', need=need)
    exec(compile(ast.Module(body=keep,type_ignores=[]),str(path),'exec'),scope)
    return scope


def framed_segment(spec, offset, stream):
    count = segment_length(spec,offset)
    protocol = original_protocol()
    index = offset//BLOCK_BYTES
    row = dict(spec,blocks=(spec['bytes']+BLOCK_BYTES-1)//BLOCK_BYTES)
    data = BytesIO()
    while data.tell() < count:
        meta = protocol['unpack'](stream.read(128),row,index)
        comp = stream.read(meta['compressed_bytes'])
        need(len(comp) == meta['compressed_bytes'] and sha256(comp).digest() == meta['compressed_digest'],
             'original compressed block digest differs')
        raw = protocol['decompress_bounded'](comp,meta['raw_bytes'])
        need(sha256(raw).digest() == meta['raw_digest'], 'original raw block digest differs')
        data.write(raw)
        index += 1
    need(data.tell() == count and stream.read(1) == b'', 'segment must contain only its exact fixed frame range')
    data.seek(0)
    return data


def source_stamp(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)


def send_segment(path, evidence, spec, offset, stream):
    count = segment_length(spec,offset)
    wanted = tuple(evidence[key] for key in ('device','inode','bytes','mtime_ns'))
    need(evidence['SHA256'] == spec['SHA256'] and evidence['bytes'] == spec['bytes'], 'source full SHA binding differs')
    fd = os.open(path, os.O_RDONLY | getattr(os,'O_NOFOLLOW',0))
    with os.fdopen(fd,'rb') as file:
        before = os.fstat(file.fileno())
        need(stat.S_ISREG(before.st_mode) and source_stamp(before) == source_stamp(path.lstat()) == wanted,
             'original source file stamp changed')
        file.seek(offset)
        raw = file.read(count)
        after = os.fstat(file.fileno())
        at_path = path.lstat()
        need(len(raw) == count and source_stamp(after) == source_stamp(at_path) == wanted
             and after.st_ctime_ns == before.st_ctime_ns
             and (os.name != 'posix' or after.st_ctime_ns == at_path.st_ctime_ns),
             'original source changed during the bounded range read')
    protocol = original_protocol()
    row = dict(spec,blocks=(spec['bytes']+BLOCK_BYTES-1)//BLOCK_BYTES)
    digest = sha256(raw).digest()
    stream.write(SEGMENT_HEADER.pack(SEGMENT_MAGIC,spec['file_id'],offset,count,digest))
    for index, begin in enumerate(range(0,count,BLOCK_BYTES), offset//BLOCK_BYTES):
        block = raw[begin:begin+BLOCK_BYTES]
        comp = zlib.compress(block,1)
        stream.write(protocol['pack'](row,index,len(comp),sha256(block).digest(),sha256(comp).digest()))
        stream.write(comp)
    stream.flush()
    return dict(file_id=spec['file_id'],offset=offset,raw_bytes=count,segment_SHA256=digest.hex(),source_stamp_verified=True)


def segment_length(spec, offset):
    need(type(offset) is int and 0 <= offset < spec['bytes'] and offset % spec['segment_bytes'] == 0
         and spec['segment_bytes'] % BLOCK_BYTES == 0, 'offset must identify one fixed segment')
    return min(spec['segment_bytes'],spec['bytes']-offset)


def seal(root, spec):
    with locked(root):
        state = load(root,spec)
        need(not state['sealed'] and state['confirmed_bytes'] == spec['bytes'], 'all exact segments must be committed')
        final = root/spec['filename']
        with payload_open(root, os.O_RDONLY) as file:
            need(check_file(file,state) == spec['bytes'], 'staged payload size differs')
            before = os.fstat(file.fileno())
            file.seek(0)
            digest = sha256()
            for raw in iter(lambda:file.read(1024**2),b''):
                digest.update(raw)
            after = os.fstat(file.fileno())
            def frozen(info):
                identity = (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
                return identity+(info.st_ctime_ns,) if os.name == 'posix' else identity
            need(frozen(before) == frozen(after) == frozen((root/'payload.partial').lstat())
                 and digest.hexdigest() == spec['SHA256'], 'original whole shard SHA or file identity differs')
        if final.exists() or final.is_symlink():
            s = final.lstat()
            need(stat.S_ISREG(s.st_mode) and [s.st_dev,s.st_ino] == state['file'], 'final path already contains another file')
        else:
            os.link(root/'payload.partial',final)
            sync_dir(root)
        identity = final.lstat()
        state['sealed_stat'] = [identity.st_mtime_ns,identity.st_ctime_ns]
        state['sealed'] = True
        checkpoint(root,state)
        return dict(file_id=spec['file_id'],bytes=spec['bytes'],SHA256=spec['SHA256'],whole_file_qualified=True)


def inspect(root, spec):
    with locked(root):
        state = load(root,spec)
        with payload_open(root, os.O_RDONLY) as file:
            observed = check_file(file,state)
        if state['sealed']:
            published = (root/spec['filename']).lstat()
            need(stat.S_ISREG(published.st_mode) and [published.st_dev,published.st_ino] == state['file']
                 and published.st_size == spec['bytes'] and [published.st_mtime_ns,published.st_ctime_ns] == state['sealed_stat'],
                 'sealed inode no longer matches its qualification')
        return dict(file_id=spec['file_id'],confirmed_bytes=state['confirmed_bytes'],actual_bytes=observed,
                    segment_count=len(state['segments']),whole_file_qualified_at_seal=state['sealed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init','inspect','append','receive','send','seal'))
    parser.add_argument('--shard',type=int,choices=(2,3,4),required=True)
    parser.add_argument('--offset',type=int)
    parser.add_argument('--segment-sha256')
    parser.add_argument('--epoch',type=float,required=True)
    args = parser.parse_args()
    need(os.name == 'posix', 'the admitted source and target HOSTs are POSIX')
    now = time.time()
    need(math.isfinite(args.epoch) and 0 < args.epoch <= now < args.epoch+600, 'original segment work window expired')
    def expired(*_):
        raise TimeoutError('original segment work window expired')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,args.epoch+600-now)
    spec = shard_spec(args.shard)
    root = TARGET/f'shard{args.shard:02}'
    if args.action == 'send':
        path,evidence = qualified_source(spec)
        result = send_segment(path,evidence,spec,args.offset,sys.stdout.buffer)
        signal.setitimer(signal.ITIMER_REAL,0)
        print(json.dumps(result, allow_nan=False), file=sys.stderr)
        return
    if args.action == 'init':
        TARGET.mkdir(mode=0o700, exist_ok=True)
        initialize(root,spec)
        result = inspect(root,spec)
    elif args.action == 'inspect':
        result = inspect(root,spec)
    elif args.action == 'append':
        need(args.offset is not None and args.segment_sha256 is not None, 'exact original range offset and digest required')
        result = append(root,spec,args.offset,args.segment_sha256,sys.stdin.buffer,framed=True)
    elif args.action == 'receive':
        need(args.offset is not None, 'exact original range offset required')
        result = append(root,spec,args.offset,None,sys.stdin.buffer,envelope=True)
    else:
        result = seal(root,spec)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
