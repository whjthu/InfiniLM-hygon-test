"""Only new UID/durability/prior-terminal/bootstrap boundaries; no remote access."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import struct
import subprocess
import sys
import tempfile
import types

D = Path(__file__).resolve().parent


def extract(path, names, ns):
    tree = ast.parse(path.read_bytes())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)


ns = dict(json=json, hashlib=hashlib, re=re, os=os, Path=Path)
extract(D/'common.py', ['sha', 'canonical', 'require', 'save', 'durable_bundle'], ns)
extract(D/'receive.py', ['parent_identity'], ns)
good, bad = [], []


def reject(name, callback):
    try:
        callback()
    except (ValueError, OSError, EOFError):
        bad.append(name)
    else:
        raise AssertionError(name+' accepted')


fields = dict(JobId='1087656', JobState='RUNNING', NumNodes='40', NodeList='node[01-40]',
              UserId='authoritative-name(154459)', AllocTRES='cpu=2560,node=40,gres/dcu=320')
row = [' 1087656 ', ' RUNNING ', ' 40 ', ' node[01-40] ', ' alias-with-padding ']
assert ns['parent_identity']('1087656', row, fields, 154459)['queue_username'] == 'alias-with-padding'
good.append('strip_all_fields_accept_alias_when_numeric_UID_matches')
reject('foreign_numeric_UID', lambda: ns['parent_identity']('1087656', row, fields, 999))
for name, changes in [('malformed_UID', dict(UserId='authoritative-name')),
                       ('nonRUNNING', dict(JobState='COMPLETED')),
                       ('wrong40', dict(NumNodes='39')), ('wrong320', dict(AllocTRES='node=40,gres/dcu=319'))]:
    reject(name, lambda changes=changes: ns['parent_identity']('1087656', row, dict(fields, **changes), 154459))

with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    complete = base/'complete'; complete.mkdir()
    directory_sync_calls = []
    # Actual file flush/fsync runs on Windows. POSIX directory fsync itself is
    # not qualified here; verify the exact production call boundary separately.
    ns['sync_directory'] = lambda root: directory_sync_calls.append(root)
    receipt = ns['durable_bundle'](complete, b'compressed-fixture', b'{}\n', dict(request='CPU-fixture'))
    assert (complete/'bundle.jsonframe.gz').read_bytes() == b'compressed-fixture'
    assert (complete/'initial-bindings.json').read_bytes() == b'{}\n'
    assert json.loads((complete/'validation.json').read_bytes()) == receipt and directory_sync_calls == [complete]
    good.append('real_files_and_file_fsync_before_validation_and_directory_sync_call')
    reject('existing_source_bundle_no_overwrite', lambda: ns['durable_bundle'](complete, b'foreign', b'{}', {}))
    failure = base/'failed'; failure.mkdir()
    real_os = ns['os']
    ns['os'] = types.SimpleNamespace(fsync=lambda *_: (_ for _ in ()).throw(OSError('fixture-fsync-failure')))
    reject('file_fsync_failure', lambda: ns['durable_bundle'](failure, b'fixture', b'{}', {}))
    assert not (failure/'validation.json').exists() and not (failure/'initial-bindings.json').exists()
    ns['os'] = real_os
    final_failure = base/'failed-directory'; final_failure.mkdir()
    ns['sync_directory'] = lambda *_: (_ for _ in ()).throw(OSError('fixture-directory-fsync-failure'))
    reject('directory_fsync_failure_no_durable_success_return',
           lambda: ns['durable_bundle'](final_failure, b'fixture', b'{}', {}))

source = ast.parse((D/'source.py').read_bytes())
fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'run_source')
durability = [node.lineno for node in ast.walk(fn) if isinstance(node, ast.Call) and
              isinstance(node.func, ast.Name) and node.func.id == 'durable_bundle']
output = [node.lineno for node in ast.walk(fn) if isinstance(node, ast.Call) and
          ast.unparse(node.func) == 'sys.stdout.buffer.write']
assert len(durability) == len(output) == 1 and durability[0] < output[0]
good.append('source_durable_call_precedes_only_binary_write')

relay = runpy.run_path(str(D/'relay_once.py'))
plan = json.loads((D/'PLAN.json').read_bytes())
raw = (D.parents[2]/plan['prior_terminal']['path']).read_bytes()
assert relay['prior_terminal_gate'](raw, plan['prior_terminal'])['reconciled_remote_exit_code'] == 1
good.append('actual_published_v1_terminal_dependency')
reject('prior_SHA_differs', lambda: relay['prior_terminal_gate'](raw+b' ', plan['prior_terminal']))
value = json.loads(raw); value['target_written_members'] = 1
mutated = json.dumps(value).encode()
reject('nonzero_original_target_writes', lambda: relay['prior_terminal_gate'](
    mutated, dict(plan['prior_terminal'], SHA256=hashlib.sha256(mutated).hexdigest())))

script = b"import sys\nassert sys.stdin.buffer.read()==b'COMPRESSED_ONLY'\nprint('BOOTSTRAP_OK')\n"
argv = [sys.executable, '-I', '-S', '-B', '-c', relay['BOOTSTRAP']]
result = subprocess.run(argv, input=relay['frame'](script, b'COMPRESSED_ONLY'), capture_output=True, timeout=5)
assert result.returncode == 0 and result.stdout.splitlines() == [b'BOOTSTRAP_OK'] and not result.stderr
good.append('actual_child_bootstrap_exact_script_and_compressed_suffix')
for name, data in [('short_prefix', b'\x01'), ('short_script', struct.pack('<Q', 40)+b'abc')]:
    result = subprocess.run(argv, input=data, capture_output=True, timeout=5)
    assert result.returncode != 0 and not result.stdout
    bad.append(name)
print(json.dumps(dict(passed=True, positive=good, rejected=bad, remote_calls=0, GPU_calls=0,
                     POSIX_directory_fsync_runtime_qualified=False, actual_model_metadata_read=False)))
