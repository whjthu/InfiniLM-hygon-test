"""Run the published Qwen decoder experiment on one owned Slurm accelerator."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time


D=Path(__file__).resolve().parent
R=D.parent.parent
P=json.loads((D/'PLAN.json').read_bytes())
CONTEXT=json.loads((D/'dispatch-context.json').read_bytes())
COLUMNS='JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%170'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}


def need(ok,reason):
    if not ok:raise ValueError(reason)


def query(*args):
    return subprocess.check_output(args,text=True,timeout=10).splitlines()


def save(name,value):
    with (D/name).open('x',encoding='utf-8') as f:json.dump(value,f,allow_nan=False,indent=2)


def account():
    labels={P['label']+suffix for suffix in ('','-pre','-post')}
    return [x.split('|') for x in query('sacct','-j',P['parent'],'-n','-P','--format='+COLUMNS)
            if x.split('|')[1] in labels]


def verify(path,expected,pin=None):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and (pin is None or before.st_size==pin),'Wrong frozen file envelope')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):
            need(time.time()<CONTEXT['epoch']+240,'Frozen asset verification deadline exhausted')
            h.update(chunk)
    ident=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(ident(before)==ident(path.lstat()) and h.hexdigest()==expected,'Frozen source or asset SHA256 differs')
    return dict(bytes=before.st_size,SHA256=h.hexdigest())


def admission():
    parent=P['parent']; node=P['node']
    rows=query('scontrol','show','job',parent,'-o')
    need(len(rows)==1,'Expected the owned long allocation')
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)',rows[0]))
    need(fields['JobId']==parent and fields['JobState']=='RUNNING' and fields['UserId'].endswith('('+str(os.getuid())+')'),
         'Long allocation is unavailable or has another owner')
    need(node in query('scontrol','show','hostnames',fields['NodeList']),'Expected Qwen node is not allocated')
    for line in query('squeue','-h','-s','-j',parent,'-o','%i|%N'):
        job,nodes=line.split('|')
        if job.rsplit('.',1)[-1] in ('extern','batch'):continue
        need(job.rsplit('.',1)[-1].isdigit() and nodes and node not in query('scontrol','show','hostnames',nodes),
             'Existing compute or unknown Slurm step conflicts')
    need(not account(),'A numbered step for this scope already exists')


def run(suffix,gpu,command,timeout):
    label=P['label']+suffix
    args=['srun','--jobid='+P['parent'],'--job-name='+label,'--nodes=1','--ntasks=1','--nodelist='+P['node'],
          '--cpus-per-task='+('8' if gpu else '2'),'--mem='+('16G' if gpu else '4G'),'--gres='+('dcu:1' if gpu else 'none'),
          '--time='+('00:04:00' if gpu else '00:00:40'),'--exact','--immediate=5',*command]
    result=subprocess.run(args,capture_output=True,timeout=min(timeout,CONTEXT['deadline']-time.time()-20))
    lines=result.stdout.splitlines()
    possible=[json.loads(line) for line in lines if line.startswith(b'{')]
    safe_errors=[]
    allow={'Frozen file changed from its verified SHA256','Original captured decoder envelope differs',
           'Original full decoder tensor inventory differs','Original decoder provider state names differ',
           'Original adapter named tensors differ','Rehydrated decoder failed original forward reproduction',
           'Rehydrated input-only VJP differs from original local VJP',
           'Native FP32 LoRA parameter bindings differ','Expected only the original 14 trainable LoRA parameters',
           'Unexpected native gradient scope','Unexpected captured operand type',
           'Unexpected keyword argument','Expected one runtime-visible accelerator'}
    for line in result.stderr.decode(errors='replace').splitlines():
        cls,sep,terr=line.partition(': ')
        if sep and cls in ('AttributeError','ValueError','TypeError','RuntimeError','KeyError','ImportError','NameError','UnpicklingError','AssertionError'):
            safe_errors.append(dict(type=cls,cause=terr if terr in allow else None,detail_SHA256=hashlib.sha256(terr.encode()).hexdigest()))
    save('receipt-'+(suffix[1:] if suffix else 'device')+'.json',dict(rc=result.returncode,stdout_bytes=len(result.stdout),
                 stderr_bytes=len(result.stderr),stderr_sha=hashlib.sha256(result.stderr).hexdigest(),safe_errors=safe_errors[-8:]))
    return result.returncode,possible[-1] if possible else None


def settle():
    until=min(time.time()+16,CONTEXT['deadline']-5)
    while True:
        rows=account()
        if rows and all(row[2].split()[0] in TERMINAL for row in rows) or time.time()>=until:
            return rows
        time.sleep(1)


def main():
    need(CONTEXT['request']==P['request'] and time.time()<CONTEXT['epoch']+40,'Original admitted epoch expired')
    verify(D/'SOURCE_FILES.json',CONTEXT['manifest_sha'])
    admission()
    assets={}
    for relative,pin in P['files'].items():
        assets[relative]=verify(R/'results'/relative,pin['SHA256'],pin['bytes'])
    save('asset-verification.json',assets)
    admission()
    snapshot=['/usr/local/python3.12/bin/python3','-B',str(D/'host_snapshot.py')]
    pre,before=run('-pre',False,snapshot,50)
    if not (pre==0 and before and before['node']==P['node'] and before['kfd_holder_count']==0):
        raise RuntimeError('Original Qwen preflight device cleanup or ownership is unavailable')
    save('before.json',before)
    swift=R/'results/training-vlm-swift-offline-runtime-v1'
    provider=R/'results/training-minicpm-hygon-processor-r2/cpu/runtime-site'
    image=R/'containers/dtk26-sandbox-r1'
    shell=('set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '
           'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 '
           'PYTHONPATH=/minicpm-transformers:/swift-runtime/site; exec python3 -B '+shlex.quote(str(D/'worker.py')))
    bind_paths={D:'rw',R/'results/training-qwen-original-public-model-v1/model':'ro',
                R/'results/training-qwen-math-sdpa-recovery-pair-v1/continuous/train/checkpoint-20':'ro',
                R/'results/training-qwen-local-decoder-vjp-v3/train/decoder-operands':'ro'}
    command=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--rocm','--no-home','--pwd','/']
    for path,mode in bind_paths.items():command+=['--bind',f'{path}:{path}:{mode}']
    command += ['--bind',f'{swift}:/swift-runtime:ro','--bind',f'{provider}:/minicpm-transformers:ro',
                '--bind','/opt/hyhal:/opt/hyhal:ro',str(image),'/bin/bash','--noprofile','--norc','-c',shell]
    save('step-intent.json',dict(request=P['request'],epoch=CONTEXT['epoch'],node=P['node'],actual_planned_device=1))
    rc,brief=run('',True,command,260)
    rows=settle()
    actual=[r for r in rows if r[1]==P['label']]
    after=None
    if len(actual)==1 and actual[0][2].split()[0] in TERMINAL:
        code,after=run('-post',False,snapshot,48)
        if not (code==0 and after and after['kfd_holder_count']==0 and before['boot_id']==after['boot_id']
                and before['pci']==after['pci']):
            after=None
        rows=settle()
        if after is not None:
            save('after.json',after)
    overall=bool(rc==0 and brief and brief['state']=='ACTUAL_REHYDRATED_DECODER_GRADIENT_TARGETS_COMPLETE'
                 and brief['actual_calls']==24 and after and len(rows)==3 and all(row[2:4]==['COMPLETED','0:0'] for row in rows))
    report=dict(request=P['request'],published=CONTEXT['publication'],actual_accounting=rows,accounting_columns=COLUMNS.split(','),
                original_passed=overall,actual_device_rc=rc,device_result_summary=brief,post_clean=bool(after),actual_asset_pins=assets)
    save('terminal.json',report)
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':main()
