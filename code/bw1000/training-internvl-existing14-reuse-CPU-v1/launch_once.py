"""One held ordinary SSH, with preparation inside the original 300s epoch."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid

D = Path(__file__).resolve().parent
G = D.parents[2]
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        __import__('os').fsync(stream.fileno())


def main(publication, mf_sha):
    epoch = time.time()
    anchor = time.monotonic()
    request = uuid.uuid4().hex
    local = G / 'local-only/training-internvl-existing14-reuse-CPU-v1-original-dispatch'
    local.mkdir(mode=0o700)
    sys.path.insert(0, str(D))
    assert hashlib.sha256((D / 'SOURCE_FILES.json').read_bytes()).hexdigest() == mf_sha
    body, context = runpy.run_path(str(D / 'build_body.py'))['make_body'](publication, request, epoch)
    save(local / 'intent.json', dict(context, payload_bytes=len(body), payload_SHA256=hashlib.sha256(body).hexdigest(), GPU=0))
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    argv = runpy.run_path(str(HELPER))['ssh']
    argv = [argv[0], '-o', 'StrictHostKeyChecking=yes', *argv[1:], '/usr/local/python3.12/bin/python3 -I -S -B -']
    save(local / 'attempt.json', dict(context, SSH_attempt_about_to_start=True, remote_outcome='UNKNOWN', replay_allowed=False))
    child = None
    timed_out = False
    failure = None
    cleanup_errors = {}
    stdout = stderr = b''
    try:
        left = min(epoch + 295 - time.time(), anchor + 295 - time.monotonic())
        assert left > 5
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = child.communicate(input=body, timeout=left)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        failure = type(error).__name__
        stdout, stderr = error.output or b'', error.stderr or b''
    except BaseException as error:
        failure = type(error).__name__
    finally:
        if child is not None and child.poll() is None:
            try:
                child.kill()
            except ProcessLookupError:
                pass
            except BaseException as error:
                cleanup_errors['kill'] = type(error).__name__
            remaining = min(3, anchor + 298 - time.monotonic())
            if remaining > 0:
                try:
                    stdout, stderr = child.communicate(timeout=remaining)
                except subprocess.TimeoutExpired as error:
                    stdout, stderr = error.output or stdout, error.stderr or stderr
                    cleanup_errors['reap'] = type(error).__name__
                except BaseException as error:
                    cleanup_errors['reap'] = type(error).__name__
            else:
                cleanup_errors['reap'] = 'ORIGINAL_CLEANUP_DEADLINE_EXHAUSTED'
    rc = child.poll() if child is not None else None
    reaped = child is not None and rc is not None
    for name, raw in [('stdout.private', stdout), ('stderr.private', stderr)]:
        with (local / name).open('xb') as stream:
            stream.write(raw)
    normal = reaped and not timed_out and failure is None and not cleanup_errors and rc not in (None, 255)
    record = dict(context, transport_rc=rc, timed_out=timed_out, held_SSH_started=child is not None,
                  held_SSH_reaped=reaped, local_error_type=failure, cleanup_errors=cleanup_errors,
                  client_wall_seconds=time.monotonic() - anchor,
                  remote_outcome='RETURNED_REQUIRES_ORIGINAL_RECEIPT_AUDIT' if normal else 'UNKNOWN',
                  stdout_bytes=len(stdout), stdout_SHA256=hashlib.sha256(stdout).hexdigest(),
                  stderr_bytes=len(stderr), stderr_SHA256=hashlib.sha256(stderr).hexdigest(),
                  replay_allowed=False, GPU=0)
    save(local / 'transport.json', record)
    print(json.dumps(record, sort_keys=True))
    return 0 if normal and rc == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', required=True)
    parser.add_argument('--source-manifest-sha', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.publication, args.source_manifest_sha))
