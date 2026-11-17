"""One bounded installed-source capture; root publication before actual SSH."""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

G = Path(__file__).resolve().parents[3]
D = Path(__file__).resolve().parent
REL = D.relative_to(G).as_posix()
PYTHON = '/home/whj/nvidia-baseline/env-vlm/bin/python'
CONFIG = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
BOOTSTRAP = "import sys;h=sys.stdin.buffer.read(8);assert len(h)==8;n=int.from_bytes(h,'big');assert 0<n<=262144;b=sys.stdin.buffer.read(n);assert len(b)==n;exec(compile(b,'published-resume-provider-CPU','exec'))"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + str(G), *args], cwd=G, timeout=10)


def preserve_sources(stdout, context, plan, run):
    require(time.time() < context['deadline'], 'SOURCE_CLIENT_DEADLINE')
    require(0 < len(stdout) <= 4 * 1024**2, 'SOURCE_STDOUT_LIMIT')
    lines = stdout.decode('utf-8').splitlines()
    require(len(lines) == 2, 'SOURCE_TWO_STREAM_RECORDS')
    ready, reply = [json.loads(line) for line in lines]
    require(ready['kind'] == 'READY' and reply['kind'] == 'OUTCOME', 'SOURCE_RECORD_KINDS')
    result = reply['outcome']
    for key, value in context.items():
        require(ready[key] == value and result[key] == value, 'SOURCE_REQUEST_CONTEXT')
    require(result['owner'] == ready['owner'] == result['owner_end'] and result['actual_affinity'] == [2, 3], 'SOURCE_OWNER')
    canonical = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(sha(canonical) == reply['outcome_SHA256'], 'SOURCE_OUTCOME_HASH')
    expected = {row['name']: row for row in plan['files']}
    require(len(result['files']) == len(expected) and {r['name'] for r in result['files']} == set(expected), 'SOURCE_FILE_DENOMINATOR')
    (run / 'source-outcome.json').write_bytes(canonical)
    if 'compressed_bundle_base64' not in reply:
        return dict(source_passed=result['passed'], source_files_saved=0, source_outcome_SHA256=sha(canonical))
    packed = base64.b64decode(reply['compressed_bundle_base64'], validate=True)
    require(len(packed) == result['compressed_bundle_bytes'] <= 2 * 1024**2
            and sha(packed) == result['compressed_bundle_SHA256'], 'SOURCE_BUNDLE_BYTES')
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
        unpacked = stream.read(8 * 1024**2 + 1)
    require(len(unpacked) == result['uncompressed_bundle_bytes'] <= 8 * 1024**2
            and sha(unpacked) == result['uncompressed_bundle_SHA256'], 'SOURCE_BUNDLE_DECOMPRESSION')
    bundle = json.loads(unpacked)
    completed = {r['name']: r for r in result['files'] if r['state'] == 'SOURCE_READ_AND_AST_PARSED'}
    require(bundle['request_id'] == context['request_id'] and set(bundle['files']) == set(completed), 'SOURCE_PARTIAL_BUNDLE_SET')
    if result['passed']:
        require(set(completed) == set(expected), 'SOURCE_SUCCESS_DENOMINATOR')
    raw_bytes = 0
    for name, body in bundle['files'].items():
        require(time.time() < context['deadline'], 'SOURCE_CLIENT_DEADLINE')
        target = Path(name)
        require(not target.is_absolute() and '..' not in target.parts and '\\' not in name, 'SOURCE_LOCAL_MEMBER_PATH')
        raw = base64.b64decode(body['base64'], validate=True); pin = completed[name]
        require(len(raw) == body['bytes'] == pin['bytes'] <= 1024**2
                and sha(raw) == body['SHA256'] == pin['SHA256'], 'SOURCE_LOCAL_MEMBER_BYTES')
        original = expected[name].get('original_SHA256')
        require(original is None or original == sha(raw), 'SOURCE_ORIGINAL_SWIFT_PIN')
        raw_bytes += len(raw); require(raw_bytes <= 4 * 1024**2, 'SOURCE_LOCAL_TOTAL_BYTES')
        path = run / 'source-files' / target; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    (run / 'provider-source-bundle.json.gz').write_bytes(packed)
    return dict(source_passed=result['passed'], source_files_saved=len(completed), source_bytes_saved=raw_bytes,
                source_outcome_SHA256=sha(canonical), source_bundle_SHA256=sha(packed), provider_imported=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', required=True)
    args = parser.parse_args()
    epoch = time.time()
    deadline = epoch + 90
    require(len(args.publication) == 40 and all(x in '0123456789abcdef' for x in args.publication), 'PUBLICATION_FORMAT')
    require(not git('status', '--porcelain', '--untracked-files=all', '--', REL).strip(), 'OWN_SOURCE_DIR_DIRTY')
    require(not git('diff', '--cached', '--name-only').strip(), 'INDEX_NOT_EMPTY')
    manifest_raw = (D / 'SOURCE_FILES.json').read_bytes()
    require(git('show', args.publication + ':' + REL + '/SOURCE_FILES.json') == manifest_raw, 'PUBLISHED_SOURCE_MF_DIFFERS')
    manifest = json.loads(manifest_raw)
    for name, pin in manifest['files'].items():
        p = D / name
        require(p.is_file() and not p.is_symlink(), 'SOURCE_MEMBER_PATH')
        raw = p.read_bytes()
        require(len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'SOURCE_MEMBER_BYTES')
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    plan = json.loads((D / 'PLAN.json').read_bytes())
    require(claim['root_GO'] is True and claim['CPU_GO'] is True and claim['GPU_GO'] is False, 'CLAIM_GO_FALSE')
    require(claim['max_invocations'] == 1 and claim['whole_seconds'] == 90, 'CLAIM_SCOPE_DIFFERS')
    require(CONFIG.is_file() and not CONFIG.is_symlink(), 'EXISTING_SSH_CONFIG_MISSING')
    run = G / 'local-only/training-T03-T04-A100-resume-provider-CPU-v1'
    run.mkdir(mode=0o700, exist_ok=False)
    context = dict(request_id=uuid.uuid4().hex, epoch=epoch, deadline=deadline,
                   publication=args.publication, source_manifest_SHA256=sha(manifest_raw),
                   claim_SHA256=sha((D / 'CLAIM.json').read_bytes()))
    (run / 'intent.json').write_text(json.dumps(context, indent=2) + '\n', encoding='utf-8')
    script = ('CONTEXT=' + repr(context) + '\nPLAN=' + repr(plan) + '\nCLAIM=' + repr(claim) + '\n').encode() + (D / 'capture_sources.py').read_bytes()
    require(len(script) <= 262144, 'BOOTSTRAP_SOURCE_LIMIT')
    remote = PYTHON + ' -I -S -B -u -c ' + shlex.quote(BOOTSTRAP)
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o',
            'StrictHostKeyChecking=yes', '-F', str(CONFIG), 'a100-27', remote]
    require(len(subprocess.list2cmdline(argv)) < 30000, 'WIN32_COMMAND_LIMIT')
    payload = len(script).to_bytes(8, 'big') + script
    child = None
    out, err = b'', b''
    forced = False
    failure = None
    try:
        require(deadline - time.time() > 10, 'NO_REMAINING_EXECUTION_WINDOW')
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = child.communicate(payload, timeout=max(0.01, deadline - time.time()))
        except subprocess.TimeoutExpired:
            forced = True
            child.kill()
            out, err = child.communicate(timeout=5)
    except BaseException as exc:
        failure = type(exc).__name__
        if child is not None and child.poll() is None:
            forced = True
            child.kill()
            out, err = child.communicate(timeout=5)
    finally:
        (run / 'stdout').write_bytes(out)
        (run / 'stderr').write_bytes(err)
        rc = child.returncode if child is not None else None
        state = 'UNKNOWN' if forced or rc in (None, 255) else 'RETURNED'
        sources = {}
        if state == 'RETURNED' and rc in (0, 1) and out:
            try:
                sources = preserve_sources(out, context, plan, run)
            except BaseException as exc:
                failure = type(exc).__name__
        result = dict(**context, finished_epoch=time.time(), remote_outcome=state, returncode=rc,
                      forced_local_SSH_kill=forced, local_failure_type=failure,
                      stdout_bytes=len(out), stdout_SHA256=sha(out), stderr_bytes=len(err), stderr_SHA256=sha(err),
                      source_preservation=sources, GPU_GO=False, replay_allowed=False)
        (run / 'outcome.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result))
    return 0 if state == 'RETURNED' and rc == 0 and sources.get('source_passed') is True and failure is None else 1


if __name__ == '__main__':
    sys.exit(main())
