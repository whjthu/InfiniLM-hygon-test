"""Exact purestdlib qualified relay primitives; original network/metadata worker not executed."""
import gzip,hashlib,json,math,os,re,stat,struct,time,zlib
from pathlib import Path,PurePosixPath

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8') + b'\n'

def require(test, message):
    if not test:
        raise ValueError(message)

def relative(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name == p.as_posix() and not p.is_absolute()
            and '..' not in p.parts and '\\' not in name and name not in ('', '.'),
            'unsafe member path')
    return p

def stable(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]

def no_links(path):
    path = Path(path)
    require(path.is_absolute(), 'absolute path required')
    for part in [*reversed(path.parents), path]:
        require(not part.is_symlink(), 'symlink path rejected')
    return path

def tick():
    require(time.time() < DEADLINE and time.monotonic() < MONO_END, 'global epoch exhausted')

def save(path, value):
    raw = canonical(value)
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())

def sync_directory(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def read_pinned(path, pin, maximum):
    tick()
    path = no_links(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == pin['bytes']
                and 0 < before.st_size <= maximum, 'pinned file size/type differs')
        raw = stream.read(before.st_size + 1)
        require(len(raw) == before.st_size and sha(raw) == pin['SHA256'], 'pinned file SHA differs')
        require(stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()),
                'file identity changed')
    tick()
    return raw, stable(before)

def pack(files, plan, context):
    rows = []
    for name in sorted(files):
        relative(name)
        raw = files[name]
        require(0 < len(raw) <= plan['max_file_bytes'], 'member budget exceeded')
        rows.append(dict(name=name, bytes=len(raw), SHA256=sha(raw)))
    index = canonical(dict(context=context, files=rows))
    require(len(index) <= plan['max_index_bytes'], 'index budget exceeded')
    raw = struct.pack('<Q', len(index)) + index + b''.join(files[x['name']] for x in rows)
    require(len(raw) <= plan['max_decompressed_bytes'], 'raw bundle budget exceeded')
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    require(len(compressed) <= plan['max_compressed_bytes'], 'compressed budget exceeded')
    return compressed

def unpack(compressed, plan, context):
    require(0 < len(compressed) <= plan['max_compressed_bytes'], 'compressed budget exceeded')
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = dec.decompress(compressed, plan['max_decompressed_bytes'] + 1)
    require(len(raw) <= plan['max_decompressed_bytes'] and dec.eof and not dec.unconsumed_tail
            and not dec.unused_data, 'truncated, trailing or oversized compressed bundle')
    require(len(raw) >= 8, 'truncated index prefix')
    n = struct.unpack('<Q', raw[:8])[0]
    require(0 < n <= plan['max_index_bytes'] and len(raw) >= 8 + n, 'index bounds differ')
    index = json.loads(raw[8:8+n])
    require(canonical(index) == raw[8:8+n] and index['context'] == context, 'index/context differs')
    rows = index['files']
    require([x['name'] for x in rows] == sorted({x['name'] for x in rows}), 'duplicate/unsorted members')
    files, offset = {}, 8 + n
    for row in rows:
        relative(row['name'])
        size = row['bytes']
        require(type(size) is int and 0 < size <= plan['max_file_bytes'], 'member size differs')
        data = raw[offset:offset+size]
        require(len(data) == size and sha(data) == row['SHA256'], 'member SHA/truncation differs')
        files[row['name']] = data
        offset += size
    require(offset == len(raw), 'trailing bundle body')
    return files
