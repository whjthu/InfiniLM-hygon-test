"""Launch one owned CPU keeper after source-published concrete scope selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

from contract import CPU_SECONDS, publish_once, sha
from controller import ROOT, DIRECTORY, SlurmBackend, verify_sources


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args(); plan = json.loads(args.plan.read_bytes())
    verify_sources(ROOT, DIRECTORY, plan)
    backend = SlurmBackend(ROOT, DIRECTORY, plan); backend.validate_resources()
    run = plan['run']
    intent = ROOT/'results'/('Kimi-long-pair-r347-'+run+'-intent.json')
    watch = ROOT/'results'/('Kimi-long-pair-r347-'+run+'-watch.json')
    if watch.exists(): raise RuntimeError('Original controller already exists; follow it')
    # Irreversible local intent prevents duplicate submission even if SSH output is lost.
    publish_once(intent, dict(run=run, plan_SHA256=sha(args.plan), CPU_keeper_only=True,
                             target_GPU_submitted=False, retry=False, reload=False))
    env = dict(backend.env, KIMI_LONG_SESSION_R347=uuid.uuid4().hex+uuid.uuid4().hex)
    command = ['srun', '--jobid=910407', '--overlap', '--exact', '--immediate=30',
               '--cpu-bind=none', '-N1', '-n1', '-c1', '-w', 'a14r2n20', '--gres=none',
               '--mem=4096M', '-t', '09:00:00', '--job-name='+backend.label('keeper'),
               '--export=ALL', '--output='+str(ROOT/'logs'/(backend.label('keeper')+'-%J.log')),
               'python3', '-B', str(DIRECTORY/'controller.py'), '--plan', str(args.plan.resolve())]
    with (ROOT/'logs'/(backend.label('keeper')+'-srun-client.log')).open('x') as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(run=run, CPU_keeper_client_PID=proc.pid, watch=str(watch),
                          target_GPU_submitted_by_invoker=False)))


if __name__ == '__main__': main()
