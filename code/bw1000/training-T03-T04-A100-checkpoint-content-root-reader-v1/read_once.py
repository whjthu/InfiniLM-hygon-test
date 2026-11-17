"""Root-only once CLI: fixed-Python -B -X utf8 read_once.py. No remote writes."""
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
import uuid

G = Path(__file__).resolve().parents[3]
D = Path(__file__).resolve().parent
CONFIG = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')
BOOT = "import sys;h=sys.stdin.buffer.read(8);assert len(h)==8;n=int.from_bytes(h,'big');assert 0<n<=262144;b=sys.stdin.buffer.read(n);assert len(b)==n;exec(compile(b,'original-content-JSON-root-audit','exec'))"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    epoch = time.time(); deadline = epoch + 30
    manifest = json.loads((D / 'SOURCE_FILES.json').read_bytes())
    for name, pin in manifest['files'].items():
        path = D / name; raw = path.read_bytes()
        require(not path.is_symlink() and len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'READER_SOURCE_BYTES')
    scope = json.loads((D / 'SCOPE.json').read_bytes())
    original = G / scope['original_local_package']
    raw = (original / 'SOURCE_FILES.json').read_bytes()
    require(sha(raw) == scope['source_manifest_SHA256'], 'ORIGINAL_LOCAL_SOURCE_MF')
    for name, pin in json.loads(raw)['files'].items():
        path = original / name; value = path.read_bytes()
        require(not path.is_symlink() and len(value) == pin['bytes'] and sha(value) == pin['SHA256'], 'ORIGINAL_LOCAL_SOURCE_BYTES')
    expected = {}
    for case in scope['cases']:
        path = G / case['local_binding_path']; raw = path.read_bytes(); binding = json.loads(raw)
        require(not path.is_symlink() and sha(raw) == case['native_binding_SHA256']
                and binding['spec']['case'] == case['case'], 'LOCAL_NATIVE_BINDING')
        rows = binding['trainable_manifest']
        require(len({r['name'] for r in rows}) == len(rows)
                and all(math.prod(r['shape']) == r['numel'] for r in rows), 'LOCAL_NATIVE_BINDING_SCHEMA')
        expected[case['case']] = dict(SHA256=sha(raw), bytes=len(raw), tensors=len(rows), elements=sum(r['numel'] for r in rows))
    context = dict(request_id=uuid.uuid4().hex, epoch=epoch, deadline=deadline)
    script = ('SCOPE=' + repr(scope) + '\nEXPECTED_BINDINGS=' + repr(expected)
              + '\nREADER_CONTEXT=' + repr(context) + '\n').encode() + (D / 'audit.py').read_bytes()
    require(len(script) <= 262144, 'READER_SCRIPT_SIZE')
    command = '/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -c ' + shlex.quote(BOOT)
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
            '-F', str(CONFIG), 'a100-27', command]
    require(CONFIG.is_file() and not CONFIG.is_symlink() and len(subprocess.list2cmdline(argv)) < 30000, 'EXISTING_SSH_CONFIG_OR_ARGV')
    run = G / 'local-only/training-T03-T04-A100-checkpoint-content-root-audit-v1'
    run.mkdir(mode=0o700, exist_ok=False)
    (run / 'intent.json').write_text(json.dumps(context, indent=2) + '\n', encoding='utf-8')
    child = None; stdout = stderr = b''; forced = False; error = None
    try:
        require(time.time() < deadline - 2, 'READER_DEADLINE_BEFORE_SSH')
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = child.communicate(len(script).to_bytes(8, 'big') + script, timeout=deadline - time.time())
        except subprocess.TimeoutExpired:
            forced = True; child.kill(); stdout, stderr = child.communicate(timeout=5)
    except BaseException as exc:
        error = type(exc).__name__
        if child is not None and child.poll() is None:
            forced = True; child.kill(); stdout, stderr = child.communicate(timeout=5)
    rc = child.returncode if child is not None else None
    result = None
    if rc == 0 and not forced:
        try:
            require(0 < len(stdout) <= 2 * 1024**2, 'ROOT_AUDIT_OUTPUT_BOUND')
            result = json.loads(stdout)
            require(result['saved_JSON_record_audit_passed'] is True and result['reader_context'] == context, 'ROOT_AUDIT_CONTEXT')
            (run / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        except BaseException as exc:
            error = type(exc).__name__
    (run / 'stdout').write_bytes(stdout); (run / 'stderr').write_bytes(stderr)
    receipt = dict(**context, returncode=rc, forced_local_SSH_kill=forced,
                   remote_outcome='UNKNOWN' if forced or rc in (None, 255) else 'RETURNED',
                   saved_JSON_record_audit_passed=result is not None and error is None,
                   local_error_type=error, stdout_bytes=len(stdout), stdout_SHA256=sha(stdout),
                   stderr_bytes=len(stderr), stderr_SHA256=sha(stderr),
                   original_outcome_SHA256=scope['original_outcome_SHA256'], actual_checkpoint_files_rehashed=False,
                   recovery_qualified=False, GPU_GO=False)
    (run / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))
    return 0 if receipt['saved_JSON_record_audit_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
