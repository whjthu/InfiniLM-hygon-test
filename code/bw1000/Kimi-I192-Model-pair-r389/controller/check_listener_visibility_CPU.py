"""Exercise actual backend probe/account glue; no Native, weights or real Slurm."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest import mock

import slurm_backend as api
from controller import Controller

HERE = Path(__file__).resolve().parent


class Clock:
    def __init__(self): self.now = 1000000.; self.sleeps = 0
    def __call__(self): return self.now
    def sleep(self, value): self.now += value; self.sleeps += 1


def exercise_bound(mode, live_producer=None):
    with tempfile.TemporaryDirectory(prefix='Kimi351-listener-') as tmp:
        root = Path(tmp); (root/'control').mkdir()
        clock = Clock()
        backend = api.SlurmBackend(root, HERE, dict(run='recording-351-test'), clock=clock, sleep=clock.sleep)
        probes, accounts = [], []
        output = '/refactor/results/Kimi-long-draft-r334-recording-351-test.json'
        row = ['910371.641', backend.label('draft'), 'RUNNING', '0:0', api.DRAFT_NODE]
        plan = json.loads((HERE/'execution-plan.template.json').read_bytes())
        plan['run'] = backend.run
        cancel = Controller(root, plan, backend, 'recording-no-secret-'*3).cancel_marker
        assert cancel.name == 'Kimi-long-pair-cancel-r347-'+backend.run+'.json'
        class Owned:
            def poll(self): return 1 if mode == 'srun_exit' or mode == 'exit_wait' and clock.now >= 1000000.2 else None
        backend.processes['draft'] = Owned()
        def command(args, timeout=25):
            assert 0 < timeout <= 2
            if args[:3] == ['scontrol', 'show', 'hostnames']: return args[3]+'\n'
            assert args[0] == 'sacct'
            accounts.append(list(args))
            if mode == 'cancel_in_command': cancel.write_text('{}')
            if mode == 'deadline_in_command': clock.now += 91
            if mode == 'query_timeout': raise subprocess.TimeoutExpired(args, timeout)
            if mode in ['never', 'cancel_wait', 'exit_wait']: return ''
            if mode == 'delay' and len(accounts) <= 2: return ''
            value = list(row)
            if mode == 'terminal': value[2:4] = ['FAILED', '1:0']
            if mode == 'wrong_node': value[4] = 'foreign-node'
            if mode == 'wrong_parent': value[0] = '910374.641'
            raw = '|'.join(value)+'|\n'
            return raw+raw if mode == 'duplicate' else raw
        backend.command = command
        producer = dict(UTC='recording', node=api.DRAFT_NODE, CPU_only=True, GPU_initialized=False,
                        actual_step='910371.642', listener_ready=True, TCP_connections_made=0,
                        passive_proc_observation=True, worker_PID=1234)
        # This real subprocess produces the same JSON frame consumed by actual probe().
        # The Slurm boundary is recorded, never submitted.
        real_run = subprocess.run
        def transport(args, **kwargs):
            probes.append(args)
            assert args[0] == 'srun' and '--gres=none' in args
            assert args.count('--listener-output') == 1 and args[args.index('--listener-output')+1] == output
            assert '46947' == args[args.index('--port')+1]
            if live_producer:
                command, env = live_producer
                return real_run(command, env=env, capture_output=True, text=True, timeout=10)
            return real_run([sys.executable, '-B', '-c', 'import sys;print(sys.argv[1])', json.dumps(producer)],
                            capture_output=True, text=True, timeout=5)
        if mode == 'cancel_wait':
            original_sleep = clock.sleep
            def wait_sleep(value):
                original_sleep(value)
                if clock.now >= 1000000.2: cancel.write_text('{}')
            clock.sleep = wait_sleep
            backend.sleep = clock.sleep
        if mode == 'cancel_before': cancel.write_text('{}')
        if mode == 'deadline_before': clock.now += api.NATIVE_SECONDS-179
        error = None
        with mock.patch.object(api.subprocess, 'run', transport), \
             mock.patch.object(socket.socket, 'connect', side_effect=AssertionError('No active connection')), \
             mock.patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('No active connection')):
            try: result = backend.draft_listening(1000000.)
            except BaseException as exc: error = exc
        assert len(probes) == 1, 'Waiting must never re-probe or start'
        assert backend.processes.keys() == {'draft'}
        if mode in ['normal', 'delay']:
            assert error is None and result['actual_draft_step'] == '910371.641'
            assert backend.actual['draft'] == '910371.641'
            assert len(accounts) == (3 if mode == 'delay' else 1)
        else:
            assert error is not None, mode
            if mode in ['srun_exit', 'cancel_before', 'deadline_before']: assert not accounts
            if mode in ['duplicate', 'wrong_parent', 'wrong_node', 'terminal']: assert len(accounts) == 1
            if mode in ['never', 'query_timeout']: assert clock.now == 1000090. and len(accounts) == 18
            if mode in ['cancel_wait', 'exit_wait']: assert len(accounts) == 1 and clock.now < 1000001.
        assert backend.startup_diagnostics[0]['phase'] == 'probe'
        return dict(actual_backend_probe_and_account=True, actual_JSON_producer=True,
                    recording_Slurm=True, GPU=0, probe_calls=len(probes), accounting_calls=len(accounts),
                    actual_resource_probe_main_and_proc=bool(live_producer),
                    no_TCP_connections=True, no_retry_or_reload=True,
                    error_type=type(error).__name__ if error else None)


def exercise(mode):
    if mode not in ['normal', 'delay'] or os.name != 'posix' or not Path('/proc/net/tcp').exists():
        result = exercise_bound(mode)
        if mode in ['normal', 'delay']: result['Linux_producer_join_pending'] = True
        return result
    worker = '/refactor/incoming/Kimi-long-draft-memory-floor-r362b/worker.py'
    output = '/refactor/results/Kimi-long-draft-r334-recording-351-test.json'
    body = "import socket,sys,json;s=socket.socket();s.bind(('127.0.0.1',0));s.listen(1);print(json.dumps({'port':s.getsockname()[1]}),flush=True);sys.stdin.readline()"
    child = subprocess.Popen([sys.executable, '-u', '-c', body, worker, output], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        port = json.loads(child.stdout.readline())['port']
        env = dict(os.environ, SLURM_JOB_ID='910371', SLURM_STEP_ID='642')
        command = [sys.executable, '-B', str(HERE/'resource_probe.py'), '--listener-output', output,
                   '--port', str(port), '--deadline-unix', str(__import__('time').time()+400)]
        # Same real producer stdout is consumed by actual backend probe -> account.
        # Recording scheduler uses this CPU host; no draft allocation or GPU is created.
        with mock.patch.object(api, 'DRAFT_NODE', socket.gethostname().split('.')[0]):
            result = exercise_bound(mode, (command, env))
        assert result['actual_resource_probe_main_and_proc']
        child.stdin.write('close\n'); child.stdin.flush(); child.wait(timeout=5)
    finally:
        if child.poll() is None: child.kill(); child.wait(timeout=5)
    result.update(recording_worker=True, actual_r334_worker=False)
    return result


def run_cases():
    for name in ['normal', 'delay', 'never', 'terminal', 'duplicate', 'wrong_parent', 'wrong_node',
                 'srun_exit', 'cancel_before', 'cancel_in_command', 'cancel_wait', 'exit_wait',
                 'query_timeout', 'deadline_before', 'deadline_in_command']:
        yield 'actual_draft_listening_visibility_'+name, lambda n=name: exercise(n)
