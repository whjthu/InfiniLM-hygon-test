"""Bounded PD state transport. No tensor allocation, pickle, or model imports.

The destination supplies its own exact ABI block descriptors and transactional
sink. A source manifest cannot authorize allocation or name executable code.
Only complete verified transfers may publish; any attempted failure poisons the
one-use ticket. This module alone does not qualify Native export/import or PD.
"""
import hashlib
import json
import math
import struct
import time

MAGIC = b'ILMPD01\0'
CHUNK = 1 << 20
MAX_HEADER = 1 << 20
MAX_BLOCKS = 8192
MAX_BYTES = 64 << 30


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_identity(identity):
    require(type(identity) is dict and set(identity) == {
        'model_SHA256', 'ABI_SHA256', 'prompt_IDs_SHA256', 'request_nonce',
        'epoch', 'past', 'TP', 'next_token', 'output_budget'}, 'Identity fields differ')
    for key in ['model_SHA256', 'ABI_SHA256', 'prompt_IDs_SHA256']:
        require(valid_sha(identity[key]), 'Identity SHA differs')
    nonce = identity['request_nonce']
    require(type(nonce) is str and 16 <= len(nonce) <= 128 and nonce.isascii()
            and nonce.isalnum(), 'Invalid request nonce')
    for key in ['epoch', 'past', 'TP', 'next_token', 'output_budget']:
        require(type(identity[key]) is int, 'Identity integer differs')
    require(identity['epoch'] > 0 and identity['TP'] == 8, 'Expected current TP8 ABI')
    require(0 < identity['past'] < 32768 and 0 < identity['output_budget'] <= 8192
            and identity['past'] + identity['output_budget'] <= 32768, 'Context budget differs')
    require(0 <= identity['next_token'] < 129280, 'Invalid selected token')


def validate_blocks(blocks):
    require(type(blocks) is list and 0 < len(blocks) <= MAX_BLOCKS, 'Invalid block count')
    seen, total = set(), 0
    for block in blocks:
        require(type(block) is dict and set(block) == {'key', 'shard', 'nbytes', 'layout_SHA256'},
                'Block descriptor fields differ')
        key, shard, size = block['key'], block['shard'], block['nbytes']
        require(type(key) is str and 0 < len(key) <= 192 and key.isascii()
                and all(c.isalnum() or c in '_.:-' for c in key), 'Invalid semantic block key')
        require(type(shard) is int and 0 <= shard < 8, 'Invalid shard')
        require(type(size) is int and 0 < size <= MAX_BYTES, 'Invalid block size')
        require(valid_sha(block['layout_SHA256']), 'Invalid local Native layout identity')
        require((shard, key) not in seen, 'Duplicate semantic block')
        seen.add((shard, key)); total += size
    require(total <= MAX_BYTES, 'Transfer exceeds hard bound')
    # Empty ranks must be explicitly represented by their Native state metadata.
    require({shard for shard, key in seen} == set(range(8)), 'Missing TP shard')
    return total


def remaining(sock, deadline):
    seconds = deadline - time.monotonic()
    require(math.isfinite(seconds) and seconds > 0, 'Original transfer deadline exhausted')
    sock.settimeout(seconds)


def read_exact(sock, count, deadline):
    result = bytearray()
    while len(result) < count:
        remaining(sock, deadline)
        part = sock.recv(min(CHUNK, count - len(result)))
        if not part:
            raise EOFError('Incomplete PD state frame')
        result.extend(part)
    return bytes(result)


def send(sock, data, deadline):
    view = memoryview(data)
    for offset in range(0, len(view), CHUNK):
        remaining(sock, deadline)
        sock.sendall(view[offset:offset + CHUNK])


def send_json(sock, value, deadline):
    raw = canonical(value)
    require(0 < len(raw) <= MAX_HEADER, 'Header exceeds bound')
    send(sock, struct.pack('!I', len(raw)) + raw, deadline)


def recv_json(sock, deadline):
    length, = struct.unpack('!I', read_exact(sock, 4, deadline))
    require(0 < length <= MAX_HEADER, 'Header exceeds bound')
    return decode(read_exact(sock, length, deadline))


def export_state(sock, identity, blocks, read_block, deadline):
    """read_block(index, offset, size)->bytes of the frozen P state snapshot.

    No source release is implied until the destination's publish ACK. D2H and
    Native snapshot fences belong to the concrete producer, not this timer.
    """
    identity, blocks = decode(canonical(identity)), decode(canonical(blocks))
    validate_identity(identity); total = validate_blocks(blocks)
    header = dict(protocol='InfiniLM-PD-1', identity=identity, blocks=blocks)
    header_sha = sha(canonical(header))
    started = time.monotonic()
    send(sock, MAGIC, deadline); send_json(sock, header, deadline)
    hashes = []
    for index, block in enumerate(blocks):
        digest = hashlib.sha256()
        for offset in range(0, block['nbytes'], CHUNK):
            size = min(CHUNK, block['nbytes'] - offset)
            part = read_block(index, offset, size)
            require(type(part) is bytes and len(part) == size, 'Exported block length differs')
            digest.update(part); send(sock, part, deadline)
        hashes.append(digest.hexdigest())
    footer = dict(header_SHA256=header_sha, block_SHA256=hashes)
    send_json(sock, footer, deadline)
    ack = recv_json(sock, deadline)
    require(canonical(ack) == canonical(dict(published=True, request_nonce=identity['request_nonce'],
                        epoch=identity['epoch'], footer_SHA256=sha(canonical(footer)))),
            'Destination publish ACK differs; source must not assume ownership retired')
    return dict(payload_bytes=total, source_transport_and_ACK_wall_s=time.monotonic() - started,
                header_SHA256=header_sha, block_SHA256=hashes, destination_published=True)


class ImportTicket:
    """One attempted state handoff. sink.prepare/write/verify/publish/abort.

    prepare() must isolate unpublished storage; write() may H2D into it. verify()
    must fence devices and validate actual Native state before publish(). Failure
    always aborts; the same ticket cannot be retried, including after timeout.
    """
    def __init__(self, identity, blocks, sink):
        validate_identity(identity); validate_blocks(blocks)
        self.identity = decode(canonical(identity))
        self.blocks = decode(canonical(blocks))
        self.sink, self.attempted, self.published = sink, False, False

    def receive(self, sock, deadline):
        require(not self.attempted, 'State handoff ticket already attempted')
        self.attempted = True
        started = time.monotonic()
        try:
            require(read_exact(sock, len(MAGIC), deadline) == MAGIC, 'Protocol magic differs')
            header = recv_json(sock, deadline)
            expected = dict(protocol='InfiniLM-PD-1', identity=self.identity, blocks=self.blocks)
            require(canonical(header) == canonical(expected),
                    'Source identity, epoch, or exact local ABI blocks differ')
            self.sink.prepare(self.identity, self.blocks)
            hashes = []
            for index, block in enumerate(self.blocks):
                digest = hashlib.sha256()
                for offset in range(0, block['nbytes'], CHUNK):
                    part = read_exact(sock, min(CHUNK, block['nbytes'] - offset), deadline)
                    digest.update(part); self.sink.write(index, offset, part)
                hashes.append(digest.hexdigest())
            footer = recv_json(sock, deadline)
            require(footer == dict(header_SHA256=sha(canonical(expected)), block_SHA256=hashes),
                    'Incomplete or corrupted state snapshot')
            remaining(sock, deadline)
            self.sink.verify()
            remaining(sock, deadline)
            self.sink.publish()
            self.published = True
            send_json(sock, dict(published=True, request_nonce=self.identity['request_nonce'],
                                 epoch=self.identity['epoch'], footer_SHA256=sha(canonical(footer))), deadline)
            return dict(payload_bytes=sum(b['nbytes'] for b in self.blocks),
                        destination_receive_verify_publish_wall_s=time.monotonic() - started,
                        block_SHA256=hashes, published=True)
        except BaseException:
            # Includes ACK loss after publish: destroy/poison this D request and
            # never expose its tokens, rather than attempting unknown-state retry.
            self.sink.abort()
            self.published = False
            raise
