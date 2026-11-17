"""One original CPU epoch; pure actual command/time-limit construction."""
import math
import shlex


def require(ok, why):
    if not ok:
        raise ValueError(why)


def left(deadline, monotonic_end, wall, monotonic, reserve=0):
    values=(deadline,monotonic_end,wall,monotonic,reserve)
    require(all(math.isfinite(x) for x in values) and reserve>=0, "invalid original CPU clock")
    value=min(deadline-wall,monotonic_end-monotonic)-reserve
    require(value>0,"original CPU epoch exhausted")
    return value


def limits(deadline, monotonic_end, wall, monotonic):
    available=left(deadline,monotonic_end,wall,monotonic)
    slurm=60*(int(min(480,available-40))//60)
    require(slurm>=60,"insufficient whole-minute Slurm budget")
    inside=int(min(450,available-75,slurm-10))
    require(inside>=1,"insufficient CPU work/kill budget")
    client=min(inside+20,available-40)
    require(client>inside,"insufficient CPU client cleanup budget")
    return dict(slurm_seconds=slurm,inside_seconds=inside,client_seconds=client,
                work_deadline=min(deadline-75,wall+inside),slurm_time=f"00:{slurm//60:02d}:00")


def terminal_timeout(deadline, monotonic_end, own_end, wall, monotonic):
    require(math.isfinite(own_end),"invalid terminal observation deadline")
    value=min(12,left(deadline,monotonic_end,wall,monotonic),own_end-monotonic)
    require(value>0,"original terminal window exhausted")
    return value


def command(container, inside_path, namespace, label, deadline, monotonic_end, wall, monotonic):
    budget=limits(deadline,monotonic_end,wall,monotonic)
    shell=("set -euo pipefail; ulimit -c 0; source /opt/dtk/env.sh; "
           "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 "
           "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=; "
           "export PYTHONPATH=/vision:/minicpm-transformers:/swift-runtime/site:"+namespace['framework']+"; "
           "exec /usr/bin/timeout --kill-after=10s "+str(budget['inside_seconds'])+" "+shlex.join([
               "python3","-B","-u",inside_path,"--deadline",str(deadline),
               "--work-deadline",str(budget['work_deadline'])]))
    argv=["srun","--jobid=1103606","--job-name="+label,"--exact","--nodes=1","--ntasks=1",
          "--nodelist=f13r1n11","--cpus-per-task=8","--mem=16384M","--gres=none",
          "--immediate=5","--time="+budget['slurm_time'],*container,namespace['container'],
          "/bin/bash","--noprofile","--norc","-c",shell]
    return argv,budget
