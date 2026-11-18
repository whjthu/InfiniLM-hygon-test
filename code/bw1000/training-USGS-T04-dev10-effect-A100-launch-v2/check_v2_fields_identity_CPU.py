"""New v2 schema/minor/diagnostic/identity boundaries only; no device/query execute."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import gpu_admission as gpu
from install_and_run import parse_process_identity

D = Path(__file__).resolve().parent
SCHEMA_SHA = 'c590b509e11b5cf599ebda04ec096d2a2b5d0eee7db36f19c8242f0c9400e2b7'


def main(schema):
    body = schema.read_bytes(); assert len(body) == 1185 and hashlib.sha256(body).hexdigest() == SCHEMA_SHA
    actual = json.loads(body); raw = actual['minor_information']['body'].encode('ascii')
    assert len(raw) == 282 and hashlib.sha256(raw).hexdigest() == '02899b1a3c606edca7c2ec3abc6532652149223017b01b2dde495156494d4788'
    expected = json.loads((D / 'INSTALL_PLAN.json').read_bytes())['expected_GPU']; results = []
    value = gpu.parse_information(raw, expected)
    results.append(dict(name='actual282_minor_schema_UUID_PCI_bound', passed=value['Device_Minor'] == 0 and value['PCI'] == '0000:26:00.0'))
    other = re.sub(rb'(?m)^Device Minor:[^\n]*', b'Device Minor: \t 7', raw)
    results.append(dict(name='minor_is_actual_field_not_index', passed=gpu.parse_information(other, expected)['Device_Minor'] == 7 and expected['index'] == 0))
    bad = {
        'wrong_UUID': raw.replace(expected['UUID'].encode(), b'GPU-wrong'),
        'wrong_PCI': raw.replace(b'0000:26:00.0', b'0000:27:00.0'),
        'missing_minor': re.sub(rb'(?m)^Device Minor:[^\n]*\n', b'', raw),
        'duplicate_minor': raw + b'Device Minor: 0\n',
        'negative_minor': re.sub(rb'(?m)^Device Minor:[^\n]*', b'Device Minor: -1', raw),
        'nonnumeric_minor': re.sub(rb'(?m)^Device Minor:[^\n]*', b'Device Minor: nvidia0', raw),
        'oversized_minor': re.sub(rb'(?m)^Device Minor:[^\n]*', b'Device Minor: 256', raw),
        'NUL_body': raw + b'\0',
        'over4096_body': raw + b'x' * 4097,
    }
    for name, data in bad.items():
        try:
            gpu.parse_information(data, expected); rejected = False
        except (ValueError, UnicodeError):
            rejected = True
        results.append(dict(name=name, passed=rejected))
    saved = gpu.subprocess.run, gpu.driver_information, gpu.selected_fd_holders, gpu.socket.socket
    observed = []
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def bind(self, value): pass
    class Result:
        returncode = 0
        stdout = ''
        stderr = ''
    def fake_run(argv, **kwargs):
        observed.append(argv)
        result = Result()
        result.stdout = actual['query_stdout'] if argv[1].startswith('--query-gpu=') else ''
        return result
    try:
        gpu.subprocess.run = fake_run; gpu.socket.socket = Socket
        gpu.driver_information = lambda *args: dict(path=actual['minor_information']['path'], fields=gpu.parse_information(raw, expected))
        gpu.selected_fd_holders = lambda *args: dict(checked_own_UID_processes=0, selected_device_or_KFD_holders=[], incomplete_processes=0)
        snapshot = gpu.snapshot(expected, 29927, 10**20)
        results.append(dict(name='actual_snapshot_calls_supported5_only', passed=observed[0][1] ==
            '--query-gpu=index,uuid,pci.bus_id,memory.free,memory.total' and snapshot['devices'][0]['minor'] == 0))
        def failure_run(*args, **kwargs):
            value = Result(); value.returncode = 2; value.stdout = 'synthetic query diagnostic\n'; value.stderr = 'x' * 9000
            return value
        gpu.subprocess.run = failure_run
        try:
            gpu.query('index,uuid,pci.bus_id,memory.free,memory.total', 'gpu', 10**20); observed_failure = False
        except gpu.AdmissionFailure as error:
            diag = error.diagnostic
            observed_failure = (error.code == 'BOUNDED_NVIDIA_QUERY_FAILED' and diag['returncode'] == 2
                and diag['stderr_bytes'] == 9000 and len(diag['stderr_preview_private']) == 1024
                and diag['stderr_SHA256'] == hashlib.sha256(b'x' * 9000).hexdigest()
                and len(json.dumps(diag).encode()) <= 16384)
        results.append(dict(name='rc_hashes_and_bounded_private_stderr_preserved', passed=observed_failure))
    finally:
        gpu.subprocess.run, gpu.driver_information, gpu.selected_fd_holders, gpu.socket.socket = saved
    values = ['S', '11', '22'] + ['0'] * 16 + ['12345678']
    stat = '1234 (python worker) ' + ' '.join(values)
    results.append(dict(name='actual_identity_parser_PID_PPID_PGID_start', passed=parse_process_identity(stat, 1234) ==
        dict(PID=1234, PPID=11, PGID=22, startticks=12345678)))
    for name, text, pid in [('wrong_PID', stat, 4321), ('missing_startticks', '1234 (python) S 11 22', 1234),
                             ('zero_startticks', stat.rsplit(' ', 1)[0] + ' 0', 1234)]:
        try:
            parse_process_identity(text, pid); rejected = False
        except (ValueError, IndexError):
            rejected = True
        results.append(dict(name=name, passed=rejected))
    tree = ast.parse((D / 'bounded_eval.py').read_bytes())
    own = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'own_identity']
    saves = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'save'
             and len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and n.args[1].value == 'control-state.json']
    queries = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'gpu_snapshot']
    results.append(dict(name='controller_identity_and_saved_state_before_any_GPU_query', passed=min(own) < min(saves) < min(queries)))
    value = dict(passed=all(r['passed'] for r in results), cases=results, actual_schema_SHA256=SCHEMA_SHA,
        actual_GPU_calls=0, actual_nvidia_smi=0, actual_remote=0, actual_Torch=0, old_tests_repeated=False)
    print(json.dumps(value, sort_keys=True)); return 0 if value['passed'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--schema-original', type=Path, required=True); a = p.parse_args()
    raise SystemExit(main(a.schema_original))
