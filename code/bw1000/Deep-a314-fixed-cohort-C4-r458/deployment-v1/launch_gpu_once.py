"""Single immutable intent -> original CPU keeper; GPU epoch starts after idle."""
import argparse
import json
import os
import re
import subprocess
import time
from support_gpu import D, LABEL, NODE, R, RUN, clean_environment, publish, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest-sha', required=True)
    p.add_argument('--publication-commit', required=True)
    p.add_argument('--source-only', action='store_true')
    a = p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', a.publication_commit)
    m = verify(a.manifest_sha, require_ready=not a.source_only)
    if a.source_only:
        print(json.dumps(dict(source_verified=True, Slurm=0, GPU=0, execution_READY=m['execution_READY'])))
        return
    for name in ['inside.sh', 'resource_wave_host_alias_outer.sh']:
        subprocess.run(['bash', '-n', str(D / name)], check=True, timeout=10)
    rows = [r.split('|') for r in subprocess.check_output(['sacct', '-S', '2026-10-31',
                '-j', '910374,910407', '-n', '-P', '-o', 'JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],
                text=True, timeout=30).splitlines() if '|' in r]
    assert not any(row[1].startswith(LABEL + '-' + RUN) for row in rows), 'Follow original handle; no new intent'
    for parent, node in [('910374', NODE), ('910407', 'a14r2n20')]:
        text = subprocess.check_output(['scontrol', 'show', 'job', parent, '-o'], text=True, timeout=25)
        assert 'JobState=RUNNING' in text
        assert re.search(r'(?:^| )UserId=' + re.escape(os.environ['USER']) + r'\(', text)
        members = subprocess.check_output(['scontrol', 'show', 'hostnames',
                    re.search(r'(?:^| )NodeList=([^ ]+)', text).group(1)], text=True, timeout=20).splitlines()
        assert node in members
    intent = R / 'results' / (LABEL + '-' + RUN + '-intent.json')
    assert not intent.exists()
    assert not (R / 'results' / (LABEL + '-' + RUN + '-watch.json')).exists()
    assert not (R / 'results/Deep458-fixed-cohort-private' / RUN).exists()
    assert not (R / 'results' / ('Deep458-fixed-cohort-' + RUN + '-actual-config.json')).exists()
    assert not list((R / 'logs').glob(LABEL + '-' + RUN + '*')), 'Existing logs prohibit resubmission'
    argv = ['srun', '--jobid=910407', '--overlap', '--exact', '--immediate=30', '--cpu-bind=none',
            '-N1', '-n1', '-c2', '--gres=none', '--mem=8192M', '-w', 'a14r2n20', '-t', '01:35:00',
            '--job-name=' + LABEL + '-' + RUN + '-keeper',
            '--output=' + str(R / 'logs' / (LABEL + '-' + RUN + '-keeper-%J.log')),
            'python3', '-B', '-u', str(D / 'controller_gpu.py'), '--manifest-sha', a.manifest_sha,
            '--publication-commit', a.publication_commit]
    publish(intent, dict(publication_commit=a.publication_commit, source_manifest_SHA256=a.manifest_sha,
            created_unix=time.time(), keeper_argv=argv, GPU_epoch_created=False, one_attempt=True,
            planned_cards=8, cohort_modes=['serial','cohort','serial','cohort','cohort','serial'], logical_requests=24, HTTP=0, retry=False, reload=False))
    with (R / 'logs' / (LABEL + '-' + RUN + '-keeper-client.log')).open('x') as log:
        proc = subprocess.Popen(argv, env=clean_environment(os.environ), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(keeper_submitted=True, srun_PID=proc.pid, owned_PGID=proc.pid,
            intent_path=str(intent), source_manifest_SHA256=a.manifest_sha,
            publication_commit=a.publication_commit, GPU_epoch_created=False)))


if __name__ == '__main__':
    main()
