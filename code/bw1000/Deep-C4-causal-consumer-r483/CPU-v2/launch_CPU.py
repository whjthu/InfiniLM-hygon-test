"""Launch the first guarded C4 HOST build on the current I03 allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

D = Path(__file__).resolve().parent
R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'Deep-C4-causal-consumer-r483-CPU-v2'
PREVIOUS_PARENT = '910407'
PARENT = '1103606'
NODE = 'f13r2n01'
I03_POOL = {'f13r2n01', 'f13r2n02', 'f13r2n03', 'f13r2n04'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query(*args):
    return subprocess.check_output(args, text=True, timeout=12).splitlines()


def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)


def expand(expression):
    rows = query('scontrol', 'show', 'hostnames', expression)
    hosts = set(rows)
    require(bool(rows) and len(hosts) == len(rows) and '' not in hosts, 'invalid Slurm host list')
    return hosts


def verify_membership():
    rows = query('scontrol', 'show', 'job', PARENT, '--oneliner')
    require(len(rows) == 1, 'missing or ambiguous current allocation')
    job = fields(rows[0])
    require(job.get('JobId') == PARENT and job.get('JobState') == 'RUNNING' and job.get('NumNodes') == '40'
            and 'gres/dcu=320' in job.get('AllocTRES', '').split(',') and job.get('NodeList'),
            'current allocation differs from the I03 ledger')
    hosts = expand(job['NodeList'])
    require(len(hosts) == 40 and I03_POOL <= hosts and NODE in I03_POOL, 'current I03 nodes differ')
    seen = set()
    for line in query('scontrol', 'show', 'step', PARENT, '--oneliner'):
        if not line.strip():
            continue
        step = fields(line)
        handle = step.get('StepId', '')
        if handle in (PARENT + '.batch', PARENT + '.extern'):
            continue
        require(re.fullmatch(PARENT + r'\.[0-9]+', handle) and handle not in seen
                and step.get('NodeList') and step.get('Nodes', '').isdigit(), 'invalid active step membership')
        seen.add(handle)
        members = expand(step['NodeList'])
        require(members <= hosts and len(members) == int(step['Nodes']), 'active step has wrong parent or nodes')
        require(NODE not in members, 'I03 build node already has an active compute step')


def verify_dedup():
    for parent, start in ((PREVIOUS_PARENT, '2026-11-06'), (PARENT, '2026-11-13')):
        rows = query('sacct', '-j', parent, '-S', start, '-n', '-P', '-o', 'JobIDRaw,JobName%100,State')
        fields_by_row = [row.split('|') for row in rows if row.strip()]
        require(all(len(row) == 3 for row in fields_by_row), 'unexpected Slurm accounting schema')
        require(all(row[1] != LABEL for row in fields_by_row), 'original build step already exists')
        if parent == PREVIOUS_PARENT:
            previous = [row for row in fields_by_row if row[0] == PREVIOUS_PARENT]
            require(len(previous) == 1 and previous[0][2].split()[0].rstrip('+') == 'CANCELLED',
                    'previous parent terminal accounting unavailable')


def main(publication):
    require(re.fullmatch('[0-9a-f]{40}', publication), 'invalid publication commit')
    subprocess.run([sys.executable, '-I', '-S', '-B', str(D / 'verify_package.py')],
                   capture_output=True, timeout=15, check=True)
    claim_path = D / 'CPU_RESOURCE_CLAIM.json'
    claim = json.loads(claim_path.read_bytes())
    require((claim['parent'], claim['node']) == (PARENT, NODE), 'published resource coordinates differ')
    ch = digest(claim_path)
    out = R / 'candidates' / LABEL
    intent = R / 'results' / (LABEL + '-intent.json')
    launch = R / 'results' / (LABEL + '-launch.json')
    log_path = R / 'logs' / (LABEL + '-srun.log')
    require(not any(p.exists() or p.is_symlink() for p in (out, intent, launch, log_path)),
            'original build namespace already exists')
    verify_dedup()
    verify_membership()
    epoch = time.time()
    deadline = epoch + 300
    args = ['srun', '--jobid=' + PARENT, '--exact', '--immediate=5', '--cpu-bind=none', '-N1', '-n1', '-c4',
            '--mem=8192M', '--gres=none', '-w', NODE, '-t', '00:05:00', '--job-name=' + LABEL,
            '--output=' + str(R / 'logs' / (LABEL + '-%J.log')), 'env', 'TUNING_CPU_CLAIM_SHA=' + ch,
            'TUNING_CPU_EPOCH=' + repr(epoch), 'TUNING_CPU_DEADLINE=' + repr(deadline),
            'WAVE_INSIDE_R2323=Deep-C4-causal-consumer-r483/CPU-v2/CPU_inside.sh',
            'bash', str(D / 'CPU_outer.sh'), 'probe']
    with intent.open('x') as f:
        json.dump(dict(publication=publication, source_manifest_SHA256=claim['source_manifest_SHA256'],
                       claim_SHA256=ch, once=True, HOST=socket.gethostname(), epoch=epoch, deadline=deadline,
                       argv=args, GPU=False, Native_dlopen=False), f)
    with log_path.open('xb') as log:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        stat = Path('/proc', str(proc.pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        identity = dict(owned_srun_PID=proc.pid, PGID=int(stat[2]), startticks=int(stat[19]))
    except FileNotFoundError:
        identity = dict(owned_srun_PID=proc.pid, exited_before_observation=True)
    receipt = dict(**identity, HOST=socket.gethostname(), epoch=epoch, deadline=deadline,
                   observed_at_epoch=time.time(), intent=str(intent), GPU=False)
    with launch.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', required=True)
    main(parser.parse_args().publication)
