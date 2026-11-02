"""Exercise real binary sockets with a recording transactional state consumer.

This is transport engineering evidence, not Native or device PD qualification.
"""
import copy
import hashlib
import json
import socket
import struct
import threading
import time

import state_wire as wire


IDENTITY = dict(model_SHA256='1' * 64, ABI_SHA256='2' * 64,
                prompt_IDs_SHA256='3' * 64, request_nonce='PDcpu0123456789abcdef',
                epoch=7, past=8489, TP=8, next_token=42, output_budget=128)
DATA = [bytes([i + 1]) * (wire.CHUNK + 13 if i == 0 else 19) for i in range(8)]
BLOCKS = [dict(key='recording_state', shard=i, nbytes=len(value), layout_SHA256='4' * 64)
          for i, value in enumerate(DATA)]
DATA.append(b'')
BLOCKS.append(dict(key='C128_unwritten_live_prefix', shard=0, nbytes=0, layout_SHA256='5' * 64))


class Sink:
    def __init__(self, fail_verify=False):
        self.prepared = self.verified = self.published = self.aborted = False
        self.data = []
        self.fail_verify = fail_verify

    def prepare(self, identity, blocks):
        assert not self.prepared and identity == IDENTITY and blocks == BLOCKS
        self.prepared = True
        self.data = [bytearray() for _ in blocks]

    def write(self, index, offset, part):
        assert self.prepared and not self.published and offset == len(self.data[index])
        self.data[index].extend(part)

    def verify(self):
        assert self.prepared and not self.published and self.data == DATA
        if self.fail_verify:
            raise RuntimeError('Recorded destination device/import failure')
        self.verified = True

    def publish(self):
        assert self.verified and not self.published
        self.published = True

    def abort(self):
        self.aborted = True
        self.published = False
        self.data = []


def packed(value):
    raw = wire.canonical(value)
    return struct.pack('!I', len(raw)) + raw


def frame(identity=None, corrupt=False):
    header = dict(protocol='InfiniLM-PD-1', identity=identity or IDENTITY, blocks=BLOCKS)
    footer = dict(header_SHA256=wire.sha(wire.canonical(header)),
                  block_SHA256=[wire.sha(value) for value in DATA])
    body = bytearray(b''.join(DATA))
    if corrupt:
        body[-1] ^= 1
    return wire.MAGIC + packed(header) + body + packed(footer)


def failed_transfer(payload, *, fail_verify=False, ack_failure=False):
    source, destination = socket.socketpair()
    sink = Sink(fail_verify)
    ticket = wire.ImportTicket(IDENTITY, BLOCKS, sink)
    errors = []

    class AckFailure:
        def __getattr__(self, name):
            return getattr(destination, name)

        def sendall(self, value):
            raise ConnectionError('Recorded ACK transport failure after publish')

    def reader():
        try:
            ticket.receive(AckFailure() if ack_failure else destination, time.monotonic() + 3)
        except BaseException as error:
            errors.append(type(error).__name__)
        finally:
            destination.close()

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        try:
            source.sendall(payload)
            source.shutdown(socket.SHUT_WR)
            source.recv(4096)
        except (BrokenPipeError, ConnectionResetError):
            pass
    finally:
        source.close()
    thread.join(4)
    assert not thread.is_alive() and errors and sink.aborted and not sink.published
    assert not ticket.published and ticket.attempted and not sink.data
    try:
        ticket.receive(None, time.monotonic() + 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Failed handoff ticket was reused')


def main():
    checks = []
    source, destination = socket.socketpair()
    sink = Sink()
    ticket = wire.ImportTicket(IDENTITY, BLOCKS, sink)
    result, errors = {}, []

    def receive():
        try:
            result.update(ticket.receive(destination, time.monotonic() + 5))
        except BaseException as error:
            errors.append(repr(error))
        finally:
            destination.close()

    thread = threading.Thread(target=receive)
    thread.start()
    exported = wire.export_state(source, IDENTITY, BLOCKS,
                                 lambda i, offset, size: DATA[i][offset:offset + size],
                                 time.monotonic() + 5)
    source.close(); thread.join(6)
    assert not errors and not thread.is_alive() and sink.published and ticket.published
    assert sink.data == DATA and exported['block_SHA256'] == result['block_SHA256']
    assert exported['payload_bytes'] == result['payload_bytes'] == sum(map(len, DATA))
    checks.append('Actual socket8shard chunkboundary roundtrip and publishACK')
    assert result['block_SHA256'][-1] == hashlib.sha256(b'').hexdigest()
    checks.append('Exact zero-length uninitialized C128 ABIblock retained without fakepayload')
    for key, value in [('epoch', 6), ('epoch', 7.0), ('model_SHA256', '5' * 64),
                       ('prompt_IDs_SHA256', '6' * 64), ('next_token', 43)]:
        identity = dict(IDENTITY); identity[key] = value
        failed_transfer(frame(identity))
        checks.append('Reject mismatched ' + key + ' before publish; poison oneuse ticket')
    failed_transfer(frame(corrupt=True))
    checks.append('Reject corrupted payload hash before Native publish')
    failed_transfer(frame()[:-31])
    checks.append('Reject truncated footer and discard unpublished state')
    failed_transfer(wire.MAGIC + struct.pack('!I', wire.MAX_HEADER + 1))
    checks.append('Reject oversized header without allocation')
    bad = b'{"identity":{},"identity":{}}'
    failed_transfer(wire.MAGIC + struct.pack('!I', len(bad)) + bad)
    checks.append('Reject duplicateJSON keys')
    failed_transfer(frame(), fail_verify=True)
    checks.append('Device/import verify failure aborts staged state')
    failed_transfer(frame(), ack_failure=True)
    checks.append('ACK failure after publish poisons and aborts request')
    bad_blocks = copy.deepcopy(BLOCKS); bad_blocks[7]['shard'] = 6
    try:
        wire.ImportTicket(IDENTITY, bad_blocks, Sink())
    except ValueError:
        checks.append('Reject duplicate or missing TP shard')
    else:
        raise AssertionError('Missing shard accepted')
    a, b = socket.socketpair()
    s = Sink(); t = wire.ImportTicket(IDENTITY, BLOCKS, s)
    try:
        t.receive(b, time.monotonic() - 1)
    except ValueError:
        assert s.aborted and t.attempted and not t.published
    else:
        raise AssertionError('Expired original deadline accepted')
    finally:
        a.close(); b.close()
    checks.append('Original deadline never refreshed on attempted handoff')
    print(json.dumps(dict(passed=True, checks=checks, CPU_transport_only=True,
                          Native_export_import_executed=False, GPU=0,
                          Model_loaded=False, no_PD_performance_claim=True)))


if __name__ == '__main__':
    main()
