"""Open one authenticated, bounded execution channel before reserving Graph64 nodes."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
import json
import struct
import subprocess
from uuid import uuid4


MAX_PROGRAM = 131072
HELLO = 'graph64-authenticated-exec-v1'
REMOTE = r"""
import json, signal, sys

deadline = 45
if hasattr(signal, 'SIGALRM'):
    signal.signal(signal.SIGALRM, lambda *_: sys.exit(124))
    signal.alarm(deadline)
handshake = dict(protocol='graph64-authenticated-exec-v1', nonce=sys.argv[1])
print(json.dumps(handshake, separators=(',', ':')), flush=True)
stream = sys.stdin.buffer
header = stream.read(8)
if len(header) != 8:
    raise SystemExit('missing one original program')
count = int.from_bytes(header, 'little')
if not 0 < count <= 131072:
    raise SystemExit('program exceeds its bound')
program = stream.read(count)
if len(program) != count or stream.read(1) != b'':
    raise SystemExit('short or overlong original program')
if hasattr(signal, 'SIGALRM'):
    signal.alarm(0)
exec(compile(program, 'graph64-verified-project-source', 'exec'))
"""


@contextmanager
def authenticated(start, *, hello_timeout=15):
    """Yield a single-use transport callable on one already authenticated channel.

    start(argv) must use the caller's existing approved, pinned route. The first
    remote process only emits the nonce and waits. It executes no project
    program until the explicitly reserved caller sends the bounded frame.
    """
    if not callable(start) or not 0 < hello_timeout <= 40:
        raise ValueError('invalid authenticated channel scope')
    nonce = uuid4().hex
    args = ('/usr/local/python3.12/bin/python3','-I','-S','-B','-u','-c',REMOTE,nonce)
    child = start(args)
    used = False
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(child.stdout.readline, 512)
            try:
                raw = pending.result(timeout=hello_timeout)
            except TimeoutError as error:
                child.kill()
                raise RuntimeError('authenticated target greeting timed out') from error
            try:
                valid = json.loads(raw) == dict(protocol=HELLO,nonce=nonce)
            except (ValueError,TypeError):
                valid = False
            if not valid or child.poll() is not None:
                raise RuntimeError('original target channel is unavailable before reservation')

        def send(program, stdout, stderr, timeout):
            nonlocal used
            if used or not isinstance(program,bytes) or not 0 < len(program) <= MAX_PROGRAM:
                raise ValueError('invalid or repeated original program')
            if child.poll() is not None:
                raise RuntimeError('verified target channel closed before program send')
            used = True
            try:
                result, diagnostics = child.communicate(struct.pack('<Q',len(program))+program,timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=5)
                raise
            stdout.write(result)
            stderr.write(diagnostics)
            return subprocess.CompletedProcess(args,child.returncode)

        yield send
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        for stream in (child.stdin,child.stdout,child.stderr):
            if stream is not None:
                stream.close()
