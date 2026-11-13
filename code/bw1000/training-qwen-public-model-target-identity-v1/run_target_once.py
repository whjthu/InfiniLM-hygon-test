"""Run the published target identity verifier after the existing relay succeeds."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys


def main():
    base = Path(sys.argv[1]).resolve()
    checkout = Path(sys.argv[2]).resolve()
    publication = Path(sys.argv[3]).resolve()
    published = json.loads(publication.read_bytes())
    assert published['pushed'] and published['remote_head'] == published['commit']
    commit = published['commit']
    relay_dir = base / 'training-qwen-original-public-model-v1'
    launch = json.loads((relay_dir / 'transfer-intent.json').read_bytes())
    terminal = json.loads((relay_dir / 'transfer-outcome.json').read_bytes())
    relay = {key: launch[key] for key in ('claim', 'request_id', 'epoch', 'deadline')}
    assert terminal['passed'] and terminal['source_rc'] == terminal['destination_rc'] == 0
    assert all(terminal[key] == val for key, val in relay.items())

    def blob(path):
        return subprocess.check_output(['git', '-c', 'safe.directory='+checkout.as_posix(),
                                        '-C', str(checkout), 'show', commit+':'+path])

    def pin(path, digest):
        raw = blob(path)
        assert hashlib.sha256(raw).hexdigest() == digest
        return raw

    task = json.loads(blob('coordination/tasks/training-qwen-public-model-target-identity-v1.json'))
    assert relay['claim'] == task['original_relay_commit']
    assert task['CPU_GO'] and not task['GPU_GO'] and task['max_CPU_invocations'] == 1
    files = {n: pin(task['source']+'/'+n, v) for n,v in task['source_SHA256'].items()}
    assert Path(__file__).read_bytes() == files['run_target_once.py']
    binding = json.loads(files['BINDING.json'])
    assert terminal['receiver_result']['files'] == binding['files']
    helper = pin(task['helper']['path'], task['helper']['SHA256'])
    tree = ast.parse((base/'read_extra40_approval_original_handles_once.py').read_bytes())
    i = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Assign)
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'ssh')
    env = {'__file__': str(base/'qwen-target-identity.py')}
    exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'known-readonly-transport','exec'),env)
    root = str(Path(terminal['receiver_result']['model_directory']).parent.as_posix())
    assert root == '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-qwen-original-public-model-v1'
    payload = (b'import signal\n'
               b'signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("verification deadline")))\n'
               b'signal.alarm(180)\n'
               + f'exec(compile({helper!r}, "indexed_model_identity", "exec"), globals())\n'.encode()
               + files['verify_identity.py'] + b'\n'
               + f'print(json.dumps(run_remote({root!r}, {binding!r}, {relay!r}, 175)))\n'.encode())
    receipt_dir = publication.parent
    with (receipt_dir/'target-read-intent.json').open('x', encoding='utf8') as f:
        json.dump(dict(commit=commit,relay=relay,program_SHA256=hashlib.sha256(payload).hexdigest()),f)
    r = subprocess.run(env['ssh'],input=payload,capture_output=True,timeout=195)
    result = dict(returncode=r.returncode, stderr_SHA256=hashlib.sha256(r.stderr).hexdigest())
    if r.returncode == 0:
        result['value'] = json.loads(r.stdout)
    else:
        result['failure_types'] = [name for name in ('TimeoutError','ValueError','Connection closed','Permission denied') if name.encode() in r.stderr]
    with (receipt_dir/'target-read-outcome.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    main()
