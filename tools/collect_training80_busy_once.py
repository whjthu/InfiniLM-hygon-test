"""Run one coordinated CPU-only busy sample across both 40-node parent jobs."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
ASSIGNMENT = '36361b92773f8eda412132ea176c29e814e4620ea8392ff287e68d854d67f5c9'
ORIGINAL = 'af59ebe7e337905759a30a59acc01670a62766827e28de399932ae5a83d77c2d'
ADDED = '568acf1d431023f15c9824f0d5d04c7a7852a0d68471195ee4954c244c661823'
PYTHON = '/usr/local/python3.12/bin/python3'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def query(*args):
    return subprocess.check_output(args, timeout=9, text=True).strip().splitlines()


def hosts(value):
    return sorted(query('scontrol', 'show', 'hostnames', value))


def accounting(parent, label):
    rows = [line.split('|') for line in query('sacct', '-n', '-P', '-j', parent, '--format=' + COLUMNS)]
    matching = [row for row in rows if len(row) == 6 and row[1] == label]
    require(len(matching) <= 1, 'duplicate sample step')
    return matching[0] if matching else None


def check_parents(by_parent):
    rows = query('squeue', '-h', '-j', ','.join(by_parent), '-o', '%i|%T|%D|%N')
    seen = set()
    for row in rows:
        job, state, count, expr = row.split('|')
        require(job in by_parent and job not in seen and state == 'RUNNING'
                and count == '40' and hosts(expr) == by_parent[job], 'live allocation changed')
        seen.add(job)
    require(seen == set(by_parent), 'missing parent allocation')
    numbered = set()
    for job in by_parent:
        lines = query('scontrol', 'show', 'step', job, '-o')
        require(bool(lines), 'unavailable step inventory')
        for line in lines:
            step = re.search(r'(?:^|\s)StepId=(\S+)', line)
            require(step is not None and step[1].startswith(job + '.'), 'unrecognized step identity')
            suffix = step[1].rsplit('.', 1)[1]
            require(suffix in {'batch', 'extern'} or suffix.isdecimal(), 'unrecognized step kind')
            if suffix.isdecimal():
                numbered.add(step[1])
    require(numbered <= {'1103606.146'}, 'another numbered step is active')
    return numbered


def loading_snapshot():
    root = ROOT / 'GLM474-indexer-100cc6c7a0154cd392f4e0f2d694dce4-socket'
    for rank in range(16):
        with (root / f'owner-{rank:02d}.json').open('rb') as stream:
            data = stream.read(2097153)
        require(len(data) <= 2097152, 'owner growing past bounded load snapshot')
        r = json.loads(data)
        require(r['actual_GPU'] == '1103606.146' and r['global_rank'] == rank
                and r['phase'] == 'load.enter' and not r['Model_READY']
                and not r['requests'] and not r['completed'], 'GLM is outside its loading phase')


def main(run, files, controller_sha):
    require(re.fullmatch(r'[0-9a-f]{32}',run) is not None, 'invalid run identity')
    expected = {'ledger.json':ASSIGNMENT, 'original.json':ORIGINAL, 'added.json':ADDED}
    require(set(files) == {*expected, 'sampler.py'}, 'incomplete payload')
    content = {}
    for name, row in files.items():
        digest, encoded = row
        data = base64.b64decode(encoded,validate=True)
        require(sha(data) == digest == expected.get(name,digest), 'input bytes changed')
        content[name] = data
    namespace = {'__name__': 'fleet_sampler'}
    exec(compile(content['sampler.py'],'sampler.py','exec'),namespace)
    owners = namespace['node_map'](json.loads(content['ledger.json']))
    by_parent = {job:sorted(n for n,(p,_) in owners.items() if p==job) for job in sorted(namespace['PARENTS'])}
    baseline = json.loads(content['original.json'])['nodes']
    added = json.loads(content['added.json'])['independent']
    require(set(baseline)==set(by_parent['1087656']) and added['verified'] is True
            and set(added['node_pins'])==set(by_parent['1103606']), 'physical inventories differ')
    pci = {node: row['PCI'] for node,row in baseline.items()}
    for node,pin in sorted(added['node_pins'].items()):
        raw = (ROOT / 'training80-resource-ledger-v1' / 'nodes' / (node+'.json')).read_bytes()
        require(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'], 'added node receipt differs')
        row = json.loads(raw)
        require(row['node']==node and row['step']=='1103606.0', 'wrong added node provenance')
        pci[node] = [card['pci'] for card in row['cards']]
    namespace['require_pins'](owners,pci)
    active = check_parents(by_parent)
    path = ROOT / ('training80-busy-' + run)
    labels = {job:'training80-busy-'+run+'-'+job for job in by_parent}
    require(not path.exists() and all(accounting(job,label) is None for job,label in labels.items()),
            'sampling namespace already exists')
    if active:
        loading_snapshot()
    path.mkdir(mode=0o700)
    samples = path/'nodes'
    samples.mkdir()
    (path/'ledger.json').write_bytes(content['ledger.json'])
    (path/'sampler.py').write_bytes(content['sampler.py'])
    save(path/'pci.json',pci)
    start = time.time()+22
    save(path/'intent.json',dict(run_id=run,scheduled_epoch=start,controller_SHA256=controller_sha,
         sampler_SHA256=sha(content['sampler.py']),labels=labels,overlap_loading=bool(active),GPU_cards=0))
    children = {}
    failures = {}
    try:
        for job,nodes in by_parent.items():
            options = ['--overlap'] if job=='1103606' and active else []
            cmd=['srun','--jobid='+job,'--job-name='+labels[job],*options,'--exact','--immediate=5','--nodes=40',
                 '--ntasks=40','--ntasks-per-node=1','--nodelist='+','.join(nodes),'--cpus-per-task=1',
                 '--mem=256M','--gres=none','--time=1','--kill-on-bad-exit=0',PYTHON,'-B',str(path/'sampler.py'),
                 'sample',str(path/'ledger.json'),str(path/'pci.json'),str(start),str(samples)]
            with (path/(job+'.log')).open('xb') as stream:
                children[job]=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        for job,child in children.items():
            child.wait(timeout=max(1,start+16-time.time()))
            if child.returncode: failures[job]=child.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        failures['controller']=type(error).__name__
    finally:
        for child in children.values():
            if child.poll() is None:
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    child.wait(timeout=3)
    deadline = time.monotonic()+12
    while True:
        rows={job:accounting(job,label) for job,label in labels.items()}
        terminal=all(row and row[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT',
                   'NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'} for row in rows.values())
        if terminal or time.monotonic()>=deadline:
            break
        time.sleep(1)
    observed = {}
    pins = {}
    for node,(job,_) in owners.items():
        file=samples/(node+'.json')
        if file.is_file():
            raw=file.read_bytes()
            row=json.loads(raw)
            require(rows[job] and row['step']==rows[job][0], 'node file does not match its launched step')
            observed[node]=row
            pins[node]=sha(raw)
    summary=namespace['aggregate'](owners,pci,observed,start)
    qualified=(terminal and not failures and all(row[2:4]==['COMPLETED','0:0'] and hosts(row[5])==by_parent[job]
                                                for job,row in rows.items()) and summary['complete'])
    if not qualified:
        summary['fleet_mean_busy_percent']=None
    summary.update(qualified=qualified,actual_accounting=rows,accounting_columns=COLUMNS.split(','),
                   original_live_steps=sorted(active),run_id=run,no_GPU_work_started=True,failures=failures,
                   device_sample_files_SHA256=sha(json.dumps(pins,sort_keys=True).encode()))
    save(path/'outcome.json',summary)
    print(json.dumps(summary,allow_nan=False))


if __name__ == '__main__':
    main(RUN, FILES, CONTROLLER_SHA)
