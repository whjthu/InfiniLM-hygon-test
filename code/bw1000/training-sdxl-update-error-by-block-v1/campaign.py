"""Run exactly one zero-GPU Slurm step on the original SDXL node."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from slurm_accounting import wait_terminal


ROOT = Path(__file__).resolve().parent
REF = ROOT.parents[1]
PARENT = '1087656'
NODE = 'f11r4n11'
LABEL = 'training-sdxl-update-error-by-block-v1'
COLUMNS = 'JobIDRaw,JobName%100,State%40,ExitCode,ElapsedRaw,NodeList,AllocTRES%160'


def write(name, data):
    with (ROOT/name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main(epoch, req):
    clock_end = time.monotonic() + 216
    claim = json.loads((ROOT/'CLAIM.json').read_bytes())
    if not 0 <= time.time() - epoch < 16 or req != claim['request']:
        raise ValueError('EXACT_REQUEST_AND_HOST_EPOCH')
    def query(*args):
        remaining = min(clock_end-time.monotonic(),epoch+220-time.time())
        if remaining <= 0:
            raise TimeoutError('HOST_EPOCH_LIMIT')
        p = subprocess.run(args, text=True, capture_output=True, check=True, timeout=min(7,remaining))
        if len(p.stdout)>131072:
            raise ValueError('SCHEDULER_OUTPUT_BOUND')
        return p.stdout.strip().splitlines()
    def row():
        lines = [x.split('|') for x in query('sacct','-n','-P','-j',PARENT,'--format='+COLUMNS)
                 if '|'+LABEL+'|' in x]
        if len(lines)>1:
            raise ValueError('DUPLICATE_EXACT_LABEL')
        return lines[0] if lines else None
    details,=query('scontrol','show','job',PARENT,'-o')
    fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)',details))
    if (fields.get('JobId'),fields.get('JobState'),fields.get('NumNodes')) != (PARENT,'RUNNING','40'):
        raise ValueError('ORIGINAL_ALLOC_NOT_LIVE')
    if not fields['UserId'].endswith('('+str(os.getuid())+')') or NODE not in query(
            'scontrol','show','hostnames',fields['NodeList']):
        raise ValueError('OWNED_NODE_PLACEMENT')
    if row() is not None:
        raise ValueError('PRIOR_REQUEST_EXISTS')
    work = REF/'results'
    paths = dict(work=ROOT, reference=work/'training-t06-sdxl-first-reference-r2',
                 observed=work/'training-sdxl-backward-cancellation-v1/candidate/replay',
                 **{'training-runtime':work/'training-t06-sdxl-runtime-v1'})
    image=REF/'containers/dtk26-sandbox-r1'
    for path in (*paths.values(),image):
        if not path.is_dir():
            raise FileNotFoundError('PINNED_MOUNT')
    cell=['/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--no-home','--pwd','/']
    for dest,path in paths.items():
        cell.extend(('--bind',str(path)+':/'+dest+(':rw' if dest=='work' else ':ro')))
    cell.extend(('--bind','/opt/hyhal:/opt/hyhal:ro',str(image),'/bin/bash','--noprofile','--norc','-c',
                 'set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; '
                 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/training-runtime/site '
                 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1; '
                 'exec python3 -B -u /work/worker.py'))
    cmd=['srun','--jobid='+PARENT,'--job-name='+LABEL,'--exact','--nodes=1','--ntasks=1','--nodelist='+NODE,
         '--cpus-per-task=4','--mem=8192M','--gres=none','--immediate=5','--time=00:02:00',*cell]
    write('intent.json',dict(request=req,epoch=epoch,deadline=epoch+220,node=NODE,parent=PARENT,physical_GPU=0))
    rc=None
    failure=None
    with (ROOT/'srun.log').open('xb') as output:
        child=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        startticks=int(Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19])
        write('srun-owner.json',dict(pid=child.pid,pgid=child.pid,startticks=startticks))
        try:
            rc=child.wait(timeout=min(128,epoch+150-time.time(),clock_end-time.monotonic()-25))
        except (TimeoutError,subprocess.TimeoutExpired):
            failure='CPU_SRUN_DEADLINE'
            if child.poll() is None:
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=3)
            rc=child.returncode
    seen=wait_terminal(row,time.monotonic()+min(42,clock_end-time.monotonic()),interval=2,max_observations=21)
    actual=seen['row']
    success=bool(rc==0 and actual and actual[2:4]==['COMPLETED','0:0']
                 and actual[5]==NODE and 'gres/dcu' not in actual[6] and (ROOT/'result.json').is_file())
    record=dict(request=req,accounting=seen,columns=COLUMNS.split(','),rc=rc,failure=failure,passed=success)
    write('terminal.json',record)
    print(json.dumps(record),flush=True)
    return int(not success)


if __name__=='__main__':
    raise SystemExit(main(float(sys.argv[1]),sys.argv[2]))
