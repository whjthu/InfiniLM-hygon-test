"""Invoke the published independent Qwen target reader once after suffix completion."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

from verify_identity import check_local_receipts

TASK = 'coordination/tasks/training-qwen-target-identity-suffix-v1.json'


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('publication', type=Path)
    parser.add_argument('transfer_receipts', type=Path)
    parser.add_argument('connection', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    publication = json.loads(args.publication.read_bytes())
    commit = publication['commit']
    if (len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)
            or publication.get('pushed') is not True or publication.get('remote_head') != commit):
        raise ValueError('published claim confirmation is required')

    def blob(path):
        return subprocess.check_output(['git', '-c', 'safe.directory='+checkout.as_posix(),
                                        '-C', str(checkout), 'show', commit+':'+path], timeout=15)

    def pin(path, digest):
        raw = blob(path)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('published verifier dependency changed')
        return raw

    task = json.loads(blob(TASK))
    if (task['CPU_GO'] is not True or task['GPU_GO'] is not False
            or task['max_CPU_invocations'] != 1):
        raise ValueError('unexpected verification scope')
    sources = {name: pin(task['source']+'/'+name, digest)
               for name, digest in task['source_SHA256'].items()}
    for name, raw in sources.items():
        if Path(__file__).with_name(name).read_bytes() != raw:
            raise ValueError('local verifier source differs from the published bytes')
    binding = json.loads(pin(task['binding']['path'], task['binding']['SHA256']))
    helper = pin(task['helper']['path'], task['helper']['SHA256'])
    files = binding['files']
    if len(files) != task['exact_files'] or sum(row['bytes'] for row in files.values()) != task['file_bytes']:
        raise ValueError('published model denominator differs')
    proofs = check_local_receipts(args.transfer_receipts, task['transfer'], files)
    if hashlib.sha256(args.connection.read_bytes()).hexdigest() != task['connection_SHA256']:
        raise ValueError('existing cluster connection helper changed')
    spec = importlib.util.spec_from_file_location('cluster_connection', args.connection)
    connection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connection)

    seconds = task['deadline_seconds']
    body = (
        'import signal\n'
        'def timeout(*_):\n    raise TimeoutError("target verification deadline")\n'
        'signal.signal(signal.SIGALRM, timeout)\n'
        f'signal.alarm({seconds})\n'
        f'exec(compile({helper!r}, "indexed_model_identity", "exec"), globals())\n'
        + sources['verify_identity.py'].decode('utf8') + '\n'
        + f'print(json.dumps(run_remote({task["target_root"]!r}, {binding!r}, '
          f'{task["transfer"]!r}, {seconds - 5})))\n'
    ).encode('utf8')
    remote = '/usr/local/python3.12/bin/python3 -B -c ' + shlex.quote(
        'import sys\nexec(compile(sys.stdin.buffer.read(), "qwen_identity", "exec"))'
    )
    args.output.mkdir(mode=0o700)
    save(args.output / 'intent.json', dict(commit=commit, transfer=task['transfer'], **proofs,
                                          program_SHA256=hashlib.sha256(body).hexdigest()))
    result = dict(passed=False, remote_outcome='UNKNOWN')
    try:
        completed = subprocess.run(connection.ssh + [remote], input=body, capture_output=True,
                                   timeout=seconds + 25)
        result.update(returncode=completed.returncode,
                      stderr_SHA256=hashlib.sha256(completed.stderr).hexdigest())
        if completed.returncode == 0:
            value = json.loads(completed.stdout)
            reported = value.get('files', {})
            if (value.get('passed') is not True or reported.keys() != files.keys()
                    or value.get('index_tensors') != binding['index_tensors']
                    or value.get('total_file_bytes') != task['file_bytes']
                    or value.get('weight_bytes_read') != sum(
                        files[name]['bytes'] for name in binding['headers'])
                    or value.get('model_loaded') is not False or value.get('GPU_calls') != 0
                    or any(reported[name].get(key) != pin_value for name, row in files.items()
                           for key, pin_value in row.items())):
                raise ValueError('remote reader returned incomplete evidence')
            result.update(passed=True, remote_outcome='VERIFIED', value=value)
        elif completed.returncode == 1 and any(
                name in completed.stderr for name in (b'ValueError:', b'TimeoutError:', b'FileNotFoundError:')):
            result['remote_outcome'] = 'VERIFICATION_FAILED'
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result['error_type'] = type(exc).__name__
    save(args.output / 'outcome.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'value'}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
