"""Root's single published READY-handshake dispatch; no model work during preparation."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import runpy
import shlex
import subprocess
import threading
import uuid
import time

D = Path(__file__).resolve().parent
G = D.parents[3]
PACKAGE = D.relative_to(G).as_posix()
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
H = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    origin=time.time();mono_end=time.monotonic()+60000
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', required=True)
    parser.add_argument('--package-sha256', required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.publication)
    git = ['git', '-c', 'safe.directory='+G.as_posix(), '-C', str(G), '--no-optional-locks']

    def published(name):
        return subprocess.check_output(git+['show', args.publication+':'+PACKAGE+'/'+name], timeout=12)

    remote = subprocess.check_output(git+['ls-remote','origin','refs/heads/bw1000-experiments'], timeout=15).split()
    assert remote and remote[0].decode('ascii') == args.publication
    raw = published('PACKAGE_FILES.json')
    assert H(raw) == args.package_sha256 and raw == (D/'PACKAGE_FILES.json').read_bytes()
    files = {}
    for name, digest in json.loads(raw).items():
        blob = published(name)
        assert Path(name).name == name and H(blob) == digest
        files[name] = (digest, base64.b64encode(blob).decode('ascii'))
    assert H(Path(__file__).read_bytes()) == files['launch_once.py'][0]
    claim = json.loads(base64.b64decode(files['GPU_RESOURCE_CLAIM.json'][1]))
    assert claim['GPU_GO'] is True and claim['max_runs_per_arm'] == 1 and claim['arm_order'] == ['socket']
    assert claim['ROOT_GO'] is True and claim['Model_GO'] is True and claim['native_arms']['socket']=='ddfeae81526758fb06d68e4f688e9eeaa48e3b173018d0040a3de0c99ba8dfb5'
    assert not subprocess.check_output(git+['status','--porcelain','--untracked-files=all','--',PACKAGE],timeout=12)
    assert H(HELPER.read_bytes()) == HELPER_SHA
    command_base = runpy.run_path(str(HELPER))['ssh']
    local = G/'local-only'/('GLM474-depth-observer-model-'+claim['run_id']+'-original-dispatch')
    local.mkdir(mode=0o700)

    def save(name, value):
        with (local/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())

    request, nonce = uuid.uuid4().hex, uuid.uuid4().hex
    payload = (f"CLAIM={files['GPU_RESOURCE_CLAIM.json'][0]!r}\nRUN={claim['run_id']!r}\n"
               f"EPOCH={origin!r}\nREQUEST={request!r}\nCOMMIT={args.publication!r}\nFILES={files!r}\n").encode()
    payload += base64.b64decode(files['dispatch_host.py'][1])
    wire = json.dumps(dict(nonce=nonce, code=base64.b64encode(payload).decode(), sha256=H(payload))).encode()+b'\n'
    assert len(wire) <= 1048576
    bootstrap = ("import base64,hashlib,json,sys\n"
                 "print(json.dumps(dict(stage='READY',nonce=NONCE)),flush=True)\n"
                 "line=sys.stdin.buffer.readline(1048577)\n"
                 "assert len(line)<=1048576 and line.endswith(b'\\n')\n"
                 "wire=json.loads(line); assert wire['nonce']==NONCE\n"
                 "raw=base64.b64decode(wire['code'],validate=True)\n"
                 "assert hashlib.sha256(raw).hexdigest()==wire['sha256']\n"
                 "exec(compile(raw,'published-attention-dispatch','exec'),{'__name__':'__main__'})\n").replace('NONCE', repr(nonce))
    command = '/usr/bin/timeout --kill-after=7s 280s /usr/local/python3.12/bin/python3 -B -u -c '+shlex.quote(bootstrap)
    argv = [command_base[0], '-o', 'StrictHostKeyChecking=yes', *command_base[1:], command]
    assert len(subprocess.list2cmdline(argv)) < 30000
    intent = dict(publication=args.publication, request_id=request, run_id=claim['run_id'],
                  claim_SHA256=files['GPU_RESOURCE_CLAIM.json'][0], epoch=origin, deadline=origin+60000, wire_SHA256=H(wire), replay_allowed=False)
    sent, stdout, error = False, b'', None
    with (local/'stderr.private').open('xb') as stderr:
        with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr) as proc:
            ready = queue.Queue()
            threading.Thread(target=lambda: ready.put(proc.stdout.readline(4097)), daemon=True).start()
            try:
                assert json.loads(ready.get(timeout=18)) == dict(stage='READY', nonce=nonce)
                save('mutating-intent.json', intent)
                sent = True
                stdout, _ = proc.communicate(wire, timeout=285)
            except BaseException as exc:
                error = type(exc).__name__
                if proc.poll() is None:
                    proc.kill()
                tail, _ = proc.communicate(timeout=8)
                stdout += tail
            rc = proc.returncode
    (local/'stdout.private').write_bytes(stdout)
    result = dict(intent, payload_dispatch_attempted=sent, returncode=rc, error_type=error,
                  stdout_SHA256=H(stdout), remote_outcome='UNKNOWN' if sent else 'NO_PAYLOAD_SENT')
    if sent and rc == 0:
        value = json.loads(stdout)
        assert value['request_id'] == request and value['run_id'] == claim['run_id'] and value['claim_commit'] == args.publication
        result.update(remote_outcome='RETURNED', launcher=value)
    save('outcome.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
