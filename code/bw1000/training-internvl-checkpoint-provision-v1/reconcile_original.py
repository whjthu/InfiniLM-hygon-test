"""One future bounded readonly reconciliation of this exact request; no transfer."""
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import time
from relay_once import BOOTSTRAP, D, G, HELPER, HELPER_SHA, envelope, save


def main():
    plan = json.loads((D/'PLAN.json').read_bytes())
    local = G/'local-only'/plan['label']
    context = json.loads((local/'intent.json').read_bytes())
    context = {key:context[key] for key in ('publication', 'request_id', 'epoch', 'deadline', 'source_manifest_SHA256')}
    assert context['deadline'] == context['epoch']+21600
    mfraw = (D/'SOURCE_FILES.json').read_bytes()
    assert hashlib.sha256(mfraw).hexdigest() == context['source_manifest_SHA256']
    for name,pin in json.loads(mfraw)['files'].items():
        raw = (D/name).read_bytes()
        assert len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['SHA256']
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    base = runpy.run_path(str(HELPER))['ssh']
    out = local/'original-reconcile-once'
    out.mkdir(mode=0o700)  # A failed/unknown readonly request is not silently replayed.
    deadline = time.time()+90
    results = {}
    for mode in ('source', 'receiver'):
        code = ('MODE='+repr(mode)+'\nPLAN='+repr(plan)+'\nCONTEXT='+repr(context)
                +'\nREAD_DEADLINE='+repr(deadline)+'\n').encode()+(D/'reconcile_endpoint.py').read_bytes()
        command = ('/home/whj/nvidia-baseline/env-vlm/bin/python' if mode == 'source'
                   else '/usr/local/python3.12/bin/python3')+' -I -S -B -c '+shlex.quote(BOOTSTRAP)
        argv = (['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12',
                 '-F', 'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config', 'a100-27', command]
                if mode == 'source' else [base[0], '-o', 'StrictHostKeyChecking=yes', *base[1:], command])
        assert len(subprocess.list2cmdline(argv)) < 30000
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        save(out/(mode+'-owned-ssh.json'), dict(pid=child.pid, deadline=deadline, request_id=context['request_id']))
        try:
            stdout, stderr = child.communicate(envelope(code), timeout=max(.1, deadline-time.time()))
            assert len(stdout) <= 262144 and len(stderr) <= 65536
            pin = dict(returncode=child.returncode, stdout_bytes=len(stdout), stdout_SHA256=hashlib.sha256(stdout).hexdigest(),
                       stderr_bytes=len(stderr), stderr_SHA256=hashlib.sha256(stderr).hexdigest())
            value = json.loads(stdout) if child.returncode == 0 else None
            if value:
                assert value['context'] == context and value['mode'] == mode
                ready_path = local/(mode+'-remote-owned.json')
                if ready_path.exists() and value.get('records',{}).get('intent'):
                    ready = json.loads(ready_path.read_bytes())
                    identity = value['records']['intent']['value']
                    assert all(ready[key] == identity[key] for key in ('host','pid','startticks','uid','boot_id'))
            pin['result'] = value
        except BaseException as error:
            child.kill(); child.wait(timeout=3)
            pin = dict(error_type=type(error).__name__, execution_state='UNKNOWN', forced_local_SSH_kill=True)
        save(out/(mode+'.json'), pin)
        results[mode] = pin
        if time.time() >= deadline:
            break
    states = {name:row.get('result', {}).get('execution_state', 'UNKNOWN') if row.get('result') else 'UNKNOWN'
              for name,row in results.items()}
    passed = len(states) == 2 and all(state == 'TERMINAL_PASS' for state in states.values())
    result = dict(context=context, states=states, original_byte_provision_receipts_verified=passed,
                  target_body_rehashed=False, tensor_numeric_qualified=False, GPU_GO=False, Model_GO=False,
                  replay_allowed=False, source_payload_reads=0, remote_writes=0)
    save(out/'result.json', result)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
