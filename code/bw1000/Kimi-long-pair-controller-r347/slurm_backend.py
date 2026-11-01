"""Owned processes only; one target step and one deferred external draft step."""
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from contract import (TARGET_NODES, DRAFT_NODE, CPU_NODE, PORTS, TERMINAL,
                      GPU_SECONDS, NATIVE_SECONDS, WORKER_SHA, sha)


def clean_env():
    return {k:v for k,v in os.environ.items()
            if not k.startswith(('SLURM_', 'PMI_', 'PMIX_', 'OMPI_', 'MV2_',
                                 'CUDA_', 'HIP_', 'ROCR_', 'GPU_'))
            or k in ['SLURM_CONF', 'SLURM_CONF_SERVER']}


class SlurmBackend:
    def __init__(self, root, directory, plan):
        self.root, self.directory, self.plan = Path(root), Path(directory), plan
        self.env = clean_env()
        self.run = plan['run']
        self.processes = {}
        self.actual = {}
        self.ignored = set()
        self.forced = []

    def command(self, args, timeout=25):
        return subprocess.check_output(args, env=self.env, text=True, timeout=timeout)

    def expand(self, nodes):
        return self.command(['scontrol', 'show', 'hostnames', nodes], 15).splitlines()

    def label(self, kind):
        return 'Kimi-long-'+kind+'-r347-'+self.run

    def parent(self, number, expected):
        raw = self.command(['scontrol', 'show', 'job', str(number), '-o'])
        assert 'UserId=aclsylqidf(' in raw and 'JobState=RUNNING' in raw
        found = re.search(r'(?:^| )NodeList=([^ ]+)', raw)
        assert found and set(expected) <= set(self.expand(found.group(1)))

    def account_rows(self, parent):
        raw = self.command(['sacct', '-S', '2026-10-31', '-j', str(parent), '-n', '-P',
                            '-o', 'JobIDRaw,JobName%160,State,ExitCode,NodeList%120'])
        return [x.split('|') for x in raw.splitlines() if x.strip()]

    def account(self, kind):
        parent, nodes = (910380, TARGET_NODES) if kind == 'target' else (910371, [DRAFT_NODE])
        rows = [r for r in self.account_rows(parent) if r[1] == self.label(kind)]
        assert len(rows) <= 1, 'Existing duplicate label is not restart authority'
        if not rows: return None
        row = rows[0]
        assert re.fullmatch(str(parent)+r'[.]\d+', row[0])
        assert set(self.expand(row[4])) == set(nodes)
        assert self.actual.get(kind, row[0]) == row[0]
        self.actual[kind] = row[0]
        return row

    def no_foreign(self):
        for parent, nodes in [(910380, TARGET_NODES), (910371, [DRAFT_NODE])]:
            for row in self.account_rows(parent):
                if not re.fullmatch(str(parent)+r'[.]\d+', row[0]): continue
                if row[0] in self.ignored or row[0] in self.actual.values(): continue
                if row[2].split()[0].rstrip('+') in TERMINAL: continue
                assert not set(self.expand(row[4])) & set(nodes), 'Active foreign step on claimed node'

    def validate_resources(self):
        self.parent(910380, TARGET_NODES)
        self.parent(910371, [DRAFT_NODE])
        self.parent(910407, [CPU_NODE])
        assert self.account('target') is None and self.account('draft') is None
        self.no_foreign()

    def probe(self, kind, suffix, extra=(), timeout=170):
        parent, nodes = (910380, TARGET_NODES) if kind == 'target' else (910371, [DRAFT_NODE])
        args = ['srun', '--jobid='+str(parent), '--overlap', '--exact', '--immediate=30',
                '--cpu-bind=none', '-N'+str(len(nodes)), '-n'+str(len(nodes)),
                '--ntasks-per-node=1', '-c1', '--gres=none', '--mem=512M', '-w', ','.join(nodes),
                '-t', '00:03:00', '--job-name='+self.label(kind)+'-'+suffix,
                'python3', '-B', str(self.directory/'resource_probe.py'), *extra]
        proc = subprocess.run(args, env=self.env, text=True, capture_output=True, timeout=timeout)
        rows = [json.loads(x) for x in proc.stdout.splitlines() if x.startswith('{')]
        assert proc.returncode == 0 and len(rows) == len(nodes), 'Owned resource/listener probe failed'
        assert {r['node'] for r in rows} == set(nodes)
        assert len({r['actual_step'] for r in rows}) == 1
        assert all(r['CPU_only'] and not r['GPU_initialized'] for r in rows)
        self.ignored.update(r['actual_step'] for r in rows)
        return rows

    def idle(self, suffix):
        rows = self.probe('target', suffix) + self.probe('draft', suffix)
        assert all(r['passed'] and r['KFD_empty'] and not r['kfd_pids']
                   and all(r['ports_bindable'].values()) for r in rows)
        return rows

    def start(self, kind, env_values, epoch):
        assert kind not in self.processes, 'No retry or second residency'
        remaining = epoch+GPU_SECONDS-time.time()
        assert remaining > 180
        is_target = kind == 'target'
        parent, nodes = (910380, TARGET_NODES) if is_target else (910371, [DRAFT_NODE])
        inside = ('Kimi-long-target-bootstrap-r346/target_inside.sh' if is_target
                  else self.directory.name+'/draft_inside.sh')
        seconds = math.floor(remaining)
        wall = '%02d:%02d:%02d' % (seconds//3600, (seconds%3600)//60, seconds%60)
        env = dict(self.env, **env_values, WAVE_INSIDE_R2323=inside)
        args = ['srun', '--jobid='+str(parent), '--overlap', '--exact', '--immediate=30',
                '--cpu-bind=none', '--kill-on-bad-exit=1', '-N'+str(len(nodes)), '-n'+str(len(nodes)),
                '--ntasks-per-node=1', '-c'+('64' if is_target else '8'), '-w', ','.join(nodes),
                '--gres=dcu:'+('8' if is_target else '1'), '--mem='+('128000M' if is_target else '65536M'),
                '-t', wall, '--job-name='+self.label(kind),
                '--output='+str(self.root/'logs'/(self.label(kind)+'-%J-%t.log')), '--export=ALL',
                'bash', str(self.directory/'outer.sh'), 'probe']
        with (self.root/'logs'/(self.label(kind)+'-srun-client.log')).open('x') as log:
            proc = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes[kind] = proc
        return dict(srun_client_PID=proc.pid, label=self.label(kind), submitted=True)

    def observe(self, kind):
        row = self.account(kind)
        proc = self.processes[kind]
        return dict(row=row, srun_exit_code=proc.poll())

    def draft_listening(self, epoch):
        output = '/refactor/results/Kimi-long-draft-r334-'+self.run+'.json'
        rows = self.probe('draft', 'listener',
                          ['--listener-output', output, '--port', '46947',
                           '--deadline-unix', str(epoch+NATIVE_SECONDS)], timeout=170)
        assert rows[0]['listener_ready'] and rows[0]['TCP_connections_made'] == 0
        # Establish the exact owned step before issuing the passive-listener receipt.
        row = self.account('draft')
        assert row and row[2].split()[0].rstrip('+') not in TERMINAL
        return dict(actual_draft_step=row[0], listener_ready=True, probe=rows[0])

    def cancel_owned(self):
        """Always use recorded handles and original Popen groups, including exited leaders."""
        for kind in self.processes:
            try:
                row = self.account(kind)
                if row and row[2].split()[0].rstrip('+') not in TERMINAL:
                    subprocess.run(['scancel', '--signal=TERM', row[0]], env=self.env,
                                   check=True, timeout=25)
                    self.forced.append(dict(kind=kind, step=row[0], signal='TERM'))
            except BaseException as error:
                self.forced.append(dict(kind=kind, cancellation_error=type(error).__name__))
        for kind, proc in self.processes.items():
            for sig in [signal.SIGTERM, signal.SIGKILL]:
                try: os.killpg(proc.pid, sig)
                except ProcessLookupError: break
                self.forced.append(dict(kind=kind, owned_PGID=proc.pid, signal=int(sig)))
                try: proc.wait(timeout=20 if sig == signal.SIGTERM else 10)
                except subprocess.TimeoutExpired: continue
                if sig == signal.SIGTERM:
                    # A reaped srun leader need not imply its original group is empty.
                    try: os.killpg(proc.pid, 0)
                    except ProcessLookupError: break
                    continue
                break
        return self.forced

    def collect_success(self):
        rows, signatures = [], []
        for node in range(4):
            path = self.root/'results'/('Kimi-long-pair-r346-'+self.run+'-node%d.json'%node)
            q = json.loads(path.read_bytes())
            assert q['completed'] and q['all8_owned_workers_shutdown_completed']
            assert q['actual_step'] == self.actual['target']
            assert q['source_manifest_SHA256'] == self.plan['target_manifest_SHA256']
            assert q['Native_SHA256'] == self.plan['Native_SHA256']
            assert q['first_capture_validation']['passed'] is True
            pair = q['pair_report']; assert pair['completed'] and pair['normal_draft_protocol_close']
            assert len(pair['requests']) == 4
            signature = []
            for r, (name, mode) in zip(pair['requests'], self.plan['request_order']):
                assert r['name'] == name and r['mode'] == mode and r['completed'] and r['normal_reset']
                assert r['finish'] == 'eos' and r['quality']['passed'] is True
                assert r['output_budget'] == 128 and 1 <= r['tokens'] <= 128
                signature.append((r['name'], r['mode'], r['prompt_SHA256'], r['ids_SHA256'], r['tokens'], r['finish']))
            signatures.append(signature)
            rows.append(dict(node=node, path=str(path), SHA256=sha(path), normal8owners=True))
        assert all(x == signatures[0] for x in signatures)
        draft = self.root/'results'/('Kimi-long-draft-r334-'+self.run+'.json')
        q = json.loads(draft.read_bytes())
        assert q['completed'] and q['passed'] and q['normal_protocol_close']
        assert q['actual_step'] == self.actual['draft'] and q['source_manifest_SHA256'] == WORKER_SHA
        assert q['authenticated'] and q['ready'] and q['GPU_count'] == 1
        phases = q['phase_counts']
        assert phases['begin'] == phases['seal'] == phases['retire'] == 2
        assert phases['close'] == 1 and phases['append_prompt'] == 81 and phases['query'] >= 2
        assert q['commands'] == sum(phases.values()) <= 4096
        assert q['feature_bytes'] >= (8635+32638)*35840*2
        return dict(target_reports=rows, all32_normal=True, four_request_signatures_equal=True,
                    draft=dict(path=str(draft), SHA256=sha(draft), normal=True),
                    performance_requires_root_audit=True, client_SLO_qualified=False)
