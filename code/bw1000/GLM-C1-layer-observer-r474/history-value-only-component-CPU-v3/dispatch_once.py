"""One new CPU step; prepare-only package, publication and root approval required."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from slurm_accounting import is_terminal, wait_terminal
from verify_package import verify
from datetime import datetime
import signal

ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'GLM474-history-value-only-component-CPU-v3'
RUN = ROOT / 'results' / LABEL
PARENT, NODE = '1103606', 'f13r1n05'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)


def remaining(deadline, monotonic_end, reserve=0):
    value = min(deadline - time.time(), monotonic_end - time.monotonic()) - reserve
    require(math.isfinite(value) and value > 0, 'original CPU epoch exhausted')
    return value


def numeric_parent_uid(value):
    match=re.fullmatch(r'(?:[^()\s]+\((\d+)\)|(\d+))',value or '')
    require(match is not None,'parent numeric UserId missing or invalid')
    return int(match.group(1) or match.group(2))


def mark_exec_attempt(install):
    path=RUN/'control-state.json'
    control=json.loads(path.read_bytes())
    require(all(control.get(k)==install[k] for k in ('request_id','publication','epoch','source_manifest_SHA256','claim_SHA256'))
            and control.get('exec_request_attempted') is False,'original control-state invalid')
    control.update(exec_request_attempted=True,phase='SRUN_POPEN_IMMINENT',updated_epoch=time.time())
    tmp=RUN/'control-state.next.json'
    with tmp.open('x',encoding='utf8') as f:
        json.dump(control,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    directory=os.open(str(RUN),os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    return control


def main(publication, expected_claim):
    install=json.loads((RUN/'install-intent.json').read_bytes())
    epoch, started = install['epoch'], time.monotonic()
    deadline=epoch+300
    monotonic_end=started+(deadline-time.time())
    require(epoch<=time.time()<deadline,'original dispatch epoch invalid')
    source = Path(__file__).resolve().parent
    require(source == RUN / 'src' and not RUN.is_symlink(), 'unexpected installed namespace')
    require(re.fullmatch(r'[0-9a-f]{40}', publication) is not None, 'invalid publication')
    require(re.fullmatch(r'[0-9a-f]{64}', expected_claim) is not None, 'invalid claim pin')
    verify(source)
    require(H(source / 'CLAIM.json') == expected_claim, 'claim differs from publication')
    require(not any((RUN / n).exists() for n in ('intent.json', 'outcome.json', 'owned-srun.json', 'step.log', 'build-result.json')), 'original namespace exists; reconcile without replay')
    terminal_deadline = None

    def query(*args):
        timeout = min(12, remaining(deadline, monotonic_end))
        if terminal_deadline is not None:
            timeout = min(timeout, terminal_deadline - time.monotonic())
        require(timeout > 0, 'terminal window exhausted')
        return subprocess.check_output(args, text=True, timeout=timeout).strip().splitlines()

    def accounting():
        rows = [line.split('|') for line in query('sacct', '-n', '-P', '-j', PARENT, '-o', COLUMNS)]
        require(all(len(row) == 7 for row in rows), 'accounting schema changed')
        matches = [row for row in rows if row[1] == LABEL]
        require(len(matches) <= 1, 'duplicate original step')
        return matches[0] if matches else None

    require(accounting() is None, 'original step already exists')
    jobs = query('scontrol', 'show', 'job', PARENT, '--oneliner')
    require(len(jobs) == 1, 'parent allocation unavailable')
    job = fields(jobs[0])
    require(numeric_parent_uid(job.get('UserId'))==os.getuid(),'parent numericUID differs from currentUID')
    require(job.get('JobState') == 'RUNNING' and job.get('NumNodes') == '40'
            and 'gres/dcu=320' in job.get('AllocTRES', '').split(','), 'parent allocation changed')
    require(NODE in query('scontrol', 'show', 'hostnames', job['NodeList']), 'node outside parent')
    require(job.get('EndTime') not in (None,'Unknown','N/A'),'parent end time unknown')
    parent_end=datetime.fromisoformat(job['EndTime']).timestamp()
    require(parent_end>=deadline+30,'parent allocation expires before whole new epoch')
    for line in query('scontrol', 'show', 'step', PARENT, '--oneliner'):
        row = fields(line)
        step = row.get('StepId', '')
        if step in (PARENT + '.extern', PARENT + '.batch'):
            continue
        require(re.fullmatch(PARENT + r'\.[0-9]+', step) is not None, 'unknown active step')
        require(NODE not in query('scontrol', 'show', 'hostnames', row['NodeList']), 'node carries active step')
    # Absolute offsets are from launch entry, including source/parent checks and queue time.
    work_deadline = epoch + 265
    inside_deadline = epoch + 270
    seconds = min(int(inside_deadline-time.time()), int(remaining(deadline, monotonic_end, 60)))
    slurm_seconds = 60 * (min(270, int(remaining(deadline, monotonic_end, 50))) // 60)
    seconds=min(seconds,slurm_seconds-15)
    require(slurm_seconds >= 60 and seconds > 10 and seconds + 10 <= slurm_seconds
            and work_deadline-time.time() > 10, 'CPU epoch too short')
    work_deadline=min(work_deadline,time.time()+seconds-30)
    require(work_deadline-time.time()>10,'insufficient work afterminute floor')
    save(RUN/'HOST-preflight.json',dict(passed=True,epoch=epoch,parent_end=parent_end,job=job,original_epoch_remaining=deadline-time.time(),node=NODE,active_step_conflicts=False))
    slurm_time = f'{slurm_seconds // 3600:02d}:{slurm_seconds // 60 % 60:02d}:00'
    target = '/refactor/results/' + LABEL
    container = ['/public/software/apps/apptainer/1.3.4/bin/apptainer', 'exec', '--rocm', '--no-home', '--pwd', '/',
        '--bind', str(ROOT)+':/refactor:ro', '--bind', str(RUN)+':'+target+':rw',
        '--bind', '/opt/hyhal:/opt/hyhal:ro', str(ROOT/'containers/dtk26-sandbox-r1')]
    shell = 'source "$1/ENV.sh"; exec /usr/bin/python3 -B -u "$1/inside_once.py"'
    container += ['/bin/bash', '--noprofile', '--norc', '-c', shell, 'build', target+'/src', str(seconds)]
    command = ['srun', '--jobid='+PARENT, '--job-name='+LABEL, '--exact', '--nodes=1', '--ntasks=1',
        '--nodelist='+NODE, '--cpus-per-task=4', '--mem=8192M', '--gres=none', '--immediate=5', '--time='+slurm_time, *container]
    save(RUN/'intent.json', dict(publication=publication, claim_SHA256=expected_claim, epoch=epoch,
        deadline=deadline, work_deadline=work_deadline, inside_deadline=inside_deadline, node=NODE,
        request_id=install['request_id'],source_manifest_SHA256=install['source_manifest_SHA256'],
        command=command, Slurm_seconds=slurm_seconds, inside_seconds=seconds, GPU_cards=0, maximum_invocations=1))
    rc, timed_out, child = None, False, None
    with (RUN/'step.log').open('xb') as stream:
        try:
            mark_exec_attempt(install)
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc = Path('/proc', str(child.pid), 'stat').read_text().rsplit(') ', 1)[1].split()
                save(RUN/'owned-srun.json', dict(PID=child.pid, PGID=int(proc[2]), startticks=int(proc[19]),host=__import__('socket').gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
            except FileNotFoundError:
                save(RUN/'owned-srun.json', dict(PID=child.pid, exited_before_identity_observation=True))
            rc = child.wait(timeout=min(seconds+20, remaining(deadline, monotonic_end, 40)))
        except BaseException:
            timed_out = True
        finally:
            # The held unreaped Popen child is the ownership anchor, even if
            # the original deadline throws between Popen and wait/receipt.
            if child is not None:
                if child.poll() is None:
                    require(os.getpgid(child.pid)==child.pid,'owned srun session changed')
                    if (RUN/'owned-srun.json').exists():
                        owned=json.loads((RUN/'owned-srun.json').read_bytes())
                        if owned.get('startticks') is not None:
                            now=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split()
                            require(owned['startticks']==int(now[19]) and owned['PGID']==int(now[2])==child.pid and owned['host']==__import__('socket').gethostname() and owned['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'owned srun identity changed')
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        require(os.getpgid(child.pid)==child.pid,'owned srun session changed');os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
                else:child.wait(timeout=0)
    errors = []

    def observe():
        try:
            return accounting()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(type(error).__name__)
            return None

    terminal_deadline = min(monotonic_end-10, time.monotonic()+35)
    visibility = wait_terminal(observe, terminal_deadline, interval=2, max_observations=18)
    row = visibility['row']
    build = json.loads((RUN/'build-result.json').read_bytes()) if (RUN/'build-result.json').is_file() else None
    passed = bool(rc == 0 and not timed_out and row and is_terminal(row) and row[2:4] == ['COMPLETED','0:0']
        and row[5] == NODE and 'gres/dcu=' not in row[6] and build and build['passed']
        and build['step'] == row[0] and build['actual'] == dict(freshM=1,HOST_compile=1,GPU_compile=0,objects=1,links=1))
    result = dict(passed=passed, exec_request_attempted=True, control_state_SHA256=H(RUN/'control-state.json'), epoch=epoch, deadline=deadline, observed_epoch=time.time(), rc=rc,
        client_timeout=timed_out, accounting_columns=COLUMNS, accounting=visibility, accounting_errors=errors,
        build=build, build_SHA256=H(RUN/'build-result.json') if build else None, log_SHA256=H(RUN/'step.log'),
        current_step_outcome='TERMINAL' if row and is_terminal(row) else 'UNKNOWN',
        GPU_cards=0, GPU_GO=False, Model_GO=False, replay_allowed=False)
    save(RUN/'outcome.json', result)
    print(json.dumps(dict(request_id=install['request_id'],passed=passed,exec_request_attempted=True,control_checkpoint_verified=True,control_state_SHA256=H(RUN/'control-state.json'),current_step_outcome=result['current_step_outcome'],rc=rc,actual=build['actual'] if build else None,build_SHA256=result['build_SHA256'],outcome_SHA256=H(RUN/'outcome.json'),GPU_GO=False,Model_GO=False,replay_allowed=False),allow_nan=False))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', required=True)
    parser.add_argument('--claim-sha256', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.publication, args.claim_sha256))
