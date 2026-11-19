"""Use only the two owned allocations, with no overlap or cancellation of existing steps."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from admission import parent_gate, step_gate
from qualified_sampler import node_map


ROOT = Path(__file__).resolve().parent
REF = ROOT.parents[1]
APPTAINER = '/public/software/apps/apptainer/1.3.4/bin/apptainer'


def save(name, data):
    with (ROOT / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def query(*args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=8, check=True)
    if len(r.stdout) > 131072:
        raise RuntimeError('SCHEDULER_OUTPUT_BOUND')
    return r.stdout.strip().splitlines()


def main(epoch, request):
    if not re.fullmatch('[a-f0-9]{32}', request) or not -2 <= time.time()-epoch < 35:
        raise ValueError('RUN_ID_OR_SINGLE_EPOCH')
    deadline = epoch + 470
    owners = node_map(json.loads((ROOT/'ledger.json').read_bytes()))
    parents = {job: sorted(n for n, (j, _) in owners.items() if j == job) for job in ('1087656', '1103606')}
    image = REF / 'containers/dtk26-sandbox-r1'
    if not image.is_dir() or not os.access(APPTAINER, os.X_OK):
        raise RuntimeError('QUALIFIED_CONTAINER_PATH')
    labels = {job: f'lora-iso-{request}-{job}' for job in parents}
    for job, nodes in parents.items():
        rows = query('scontrol','show','job',job,'-o')
        if len(rows) != 1:
            raise ValueError('EXACT_PARENT_ROW')
        owner = parent_gate(rows[0], job, os.getuid(), nodes,
                            lambda expression: query('scontrol','show','hostnames',expression))
        if time.mktime(time.strptime(owner['EndTime'], '%Y-%m-%dT%H:%M:%S')) <= deadline:
            raise ValueError('PARENT_LEASE_ENDS_TOO_SOON')
        step_gate(query('scontrol','show','step',job,'-o'),job)
        lo,hi = labels[job], job
        if any(row.split('|')[1:2] == [lo] for row in query('sacct','-n','-P','-j',hi,'--format=JobIDRaw,JobName%100')):
            raise ValueError('DUPLICATE_REQUEST')
    if time.time() >= epoch + 70:
        raise TimeoutError('ADMISSION_BUDGET')
    save('intent.json', dict(request=request, epoch=epoch, deadline=deadline, parents=parents, labels=labels, cards=640))
    watch = ROOT/'nodes'
    watch.mkdir(mode=0o700)
    children = {}
    errors = {}
    columns = 'JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,AllocTRES%160'

    def interrupted(_signal, _frame):
        raise InterruptedError('OWNED_DISPATCH_INTERRUPTED')

    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,interrupted)
    try:
        for job, nodes in parents.items():
            args = ['srun','--jobid='+job,'--job-name='+labels[job],'--exact','--nodes=40','--ntasks=40',
                    '--ntasks-per-node=1','--nodelist='+','.join(nodes),'--immediate=5','--kill-on-bad-exit=0',
                    '--cpus-per-task=24','--mem=48G','--gres=dcu:8','--time=00:06:00',
                    '--output='+str(ROOT/(job+'-%N.stdout')),'--error='+str(ROOT/(job+'-%N.stderr')),
                    APPTAINER,'exec','--rocm','--no-home','--bind',str(ROOT)+':'+str(ROOT),
                    '--bind','/opt/hyhal:/opt/hyhal:ro',str(image),'/bin/bash','--noprofile','--norc','-c',
                    'set -eo pipefail; source /opt/dtk/env.sh; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1; exec python3 -B -u "$1" "$2" "$3"',
                    'fleet-worker',str(ROOT/'worker.py'),str(watch),str(epoch)]
            children[job] = subprocess.Popen(args,start_new_session=True)
        save('submitted.json', dict(request=request,epoch=time.time(),owned_srun={j:c.pid for j,c in children.items()}))
        for job, child in children.items():
            child.wait(timeout=max(1, deadline-time.time()-25))
    except BaseException as exc:
        errors['dispatch'] = dict(type=type(exc).__name__,reason_SHA256=hashlib.sha256(str(exc).encode()).hexdigest())
    finally:
        for child in children.values():
            if child.poll() is None:
                try:
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=3)
                except ProcessLookupError:
                    child.poll()
    accounts = {}
    for job in children:
        try:
            for _ in range(3):
                rows = [r.split('|') for r in query('sacct','-n','-P','-j',job,'--format='+columns)
                        if r.split('|')[1:2] == [labels[job]]]
                accounts[job] = rows
                if len(rows) == 1 and rows[0][2] not in ('RUNNING','PENDING','COMPLETING'):
                    break
                if time.time() >= deadline - 10:
                    break
                time.sleep(1)
        except Exception as exc:
            errors[job] = type(exc).__name__
    result = dict(request=request, epoch=epoch, reported_epoch=time.time(), parents=labels, columns=columns.split(','),
                  accounts=accounts, returncodes={job:c.poll() for job,c in children.items()}, errors=errors)
    save('dispatch-terminal.json',result)
    print(json.dumps(result),flush=True)
    return int(bool(errors) or len(children) != 2 or any(c.poll()!=0 for c in children.values()))


if __name__ == '__main__':
    raise SystemExit(main(float(sys.argv[1]),sys.argv[2]))
