"""Stage exact compiled-source bytes and submit the one owned CPU step."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'DeepC4-head-scale-fusion-gfx936-CPU-v1'
PARENT = '1103606'
NODE = 'f13r2n02'


def need(value, message):
    if not value:
        raise ValueError(message)


def query(*args):
    data = subprocess.check_output(args, text=True, timeout=8)
    need(len(data) <= 400000, 'SLURM_QUERY_BOUND')
    return data.splitlines()


def fields(row):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)', row))


def job_gate():
    line, = query('scontrol','show','job',PARENT,'-o')
    parent = fields(line)
    need(parent['JobState'] == 'RUNNING' and parent['UserId'].endswith(f'({os.getuid()})')
         and parent['NumNodes'] == '40' and 'gres/dcu=320' in parent['AllocTRES'].split(',')
         and NODE in query('scontrol','show','hostnames',parent['NodeList']), 'PARENT_OWNERSHIP_OR_PLACEMENT')
    steps = query('scontrol','show','step',PARENT,'-o')
    need(steps, 'STEP_VISIBILITY')
    for line in steps:
        row = fields(line)
        ident = row.get('StepId')
        if ident in {PARENT+'.extern', PARENT+'.batch'}:
            continue
        nodes = row.get('NodeList')
        need(ident and re.fullmatch(PARENT+r'\.\d+', ident) and nodes not in (None,'(null)','')
             and NODE not in query('scontrol','show','hostnames',nodes),'ACTIVE_OR_UNKNOWN_STEP_PLACEMENT')
    for parent_id in ('1087656', PARENT):
        rows = query('sacct','-nP','-j',parent_id,'-o','JobIDRaw,JobName%100')
        need(not any(row.split('|')[1:2] == [LABEL] for row in rows), 'ORIGINAL_LABEL_EXISTS')


def run(payload):
    files = {name: base64.b64decode(raw, validate=True) for name, raw in payload['files'].items()}
    manifest = files['SOURCE_FILES.json']
    mf = hashlib.sha256(manifest).hexdigest()
    man = json.loads(manifest)
    expected = man['files_SHA256']
    need(mf == payload['manifest_SHA256'] and set(files) == set(expected)|{'SOURCE_FILES.json'}, 'SOURCE_MANIFEST')
    need(all(Path(name).name == name and hashlib.sha256(files[name]).hexdigest() == pin
             for name, pin in expected.items()), 'SOURCE_BYTES')
    plan = json.loads(files['PLAN.json'])
    need((plan['label'],plan['parent'],plan['node']) == (LABEL,PARENT,NODE), 'PUBLISHED_PLACEMENT')
    need(re.fullmatch('[0-9a-f]{40}', payload['publication']) and re.fullmatch('[0-9a-f]{32}',payload['request']),
         'PUBLICATION_REQUEST_IDENTITY')
    paths = [ROOT/'incoming'/LABEL, ROOT/'candidates'/LABEL, ROOT/'results'/(LABEL+'-intent.json'),
             ROOT/'results'/(LABEL+'-launch.json'), ROOT/'logs'/(LABEL+'-srun.log')]
    need(not any(path.exists() for path in paths), 'ORIGINAL_NAMESPACE_EXISTS')
    job_gate()
    package = paths[0]
    package.mkdir(mode=0o700)
    for name, raw in files.items():
        with (package/name).open('xb') as handle:
            handle.write(raw)
    epoch = time.time()
    deadline = epoch + 300
    args = ['srun','--jobid='+PARENT,'--exact','--immediate=5','--cpu-bind=none',
            '-N1','-n1','-c4','--mem=8192M','--gres=none','-w',NODE,'-t','00:05:00',
            '--job-name='+LABEL,'--output='+str(ROOT/'logs'/(LABEL+'-%J.log')),
            'env','TUNING_CPU_MANIFEST_SHA='+mf,'TUNING_CPU_EPOCH='+str(epoch),'TUNING_CPU_DEADLINE='+str(deadline),
            'WAVE_INSIDE_R2323='+LABEL+'/CPU_inside.sh','/bin/bash',str(package/'CPU_outer.sh'),'probe']
    intent=dict(publication=payload['publication'],request=payload['request'],manifest_SHA256=mf,
                epoch=epoch,deadline=deadline,node=NODE,parent=PARENT,GRES='none',argv=args)
    with paths[2].open('x') as handle:
        json.dump(intent,handle)
        handle.flush()
        os.fsync(handle.fileno())
    with paths[4].open('xb') as log:
        process = subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True)
    stat = (Path('/proc')/str(process.pid)/'stat').read_text().rsplit(') ',1)[1].split()
    row = dict(request=payload['request'],publication=payload['publication'],manifest_SHA256=mf,
               epoch=epoch,deadline=deadline,host=socket.gethostname(),pid=process.pid,
               pgid=int(stat[2]),startticks=int(stat[19]),intent=str(paths[2]),GPU=False)
    with paths[3].open('x') as handle:
        json.dump(row,handle)
    print(json.dumps(row),flush=True)


if __name__ == '__main__':
    run(PAYLOAD)
