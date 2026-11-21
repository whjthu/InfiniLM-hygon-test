"""Build the read-only .678 audit command from the original DTK environment prefix."""

from hashlib import sha256
from pathlib import Path


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'Kimi461-wave-original678-provider-audit-CPU-v1'
PARENT, NODE = '1103606', 'f12r2n13'
ENV_PATH = Path('code/bw1000/Kimi-vocab-parallel-embedding-r461/output-residual-stage-diagnostic-CPU-v1/CPU_inside.sh')
ENV_PREFIX_SHA256 = '456e3ab53c263e6d01857974c2c0abdef21a0ea1aea06f0dfa588b13fe9e11b3'


def command(published_env, staged, out, epoch):
    """Return the bounded, exact one-node srun argv without invoking it."""
    prefix, marker, _ = published_env.partition(b'left=$(python')
    if not marker or sha256(prefix).hexdigest() != ENV_PREFIX_SHA256:
        raise ValueError('original DTK environment differs')
    if staged != ROOT / 'incoming' / LABEL or out != ROOT / 'results' / LABEL:
        raise ValueError('audit must use new canonical scope directories')
    replay = prefix.decode() + '''
export CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
budget=$(python -I -S -B -c 'import math,sys,time
epoch=float(sys.argv[1]); now=time.time()
if not math.isfinite(epoch) or not epoch <= now < epoch+270:
    raise SystemExit("original provider audit epoch expired")
print(int(epoch+280-now))' "${@: -1}")
exec /usr/bin/timeout --kill-after=5s "${budget}s" python -I -S -B "$@"
'''
    result = out / 'original-audit-result.json'
    return ['srun',f'--jobid={PARENT}',f'--job-name={LABEL}',f'--nodelist={NODE}','--exact',
            '--nodes=1','--ntasks=1','--cpus-per-task=4','--cpu-bind=threads',
            '--mem=8192M','--gres=none','--time=00:05:00','--immediate=5',
            '/public/software/apps/apptainer/1.3.4/bin/apptainer','exec','--pwd','/',
            '--home',str(ROOT.with_name('hygon-inference-20261013')/'build-deps/root')+':/root',
            '--bind',f'{ROOT}:/refactor:ro',
            '--bind',f'{ROOT.with_name("hygon-inference-20261013")}:/old:ro',
            '--bind','/opt/hyhal:/opt/hyhal:ro',
            '--bind',f'{out}:{out}:rw',
            '--bind',f'{staged}:{staged}:ro',
            str(ROOT/'containers/dtk26-sandbox-r1'),'/bin/bash','--noprofile','--norc','-c',replay,'original678-audit',
            str(staged/'audit_once.py'),'--result',str(result),'--epoch',repr(epoch)]
