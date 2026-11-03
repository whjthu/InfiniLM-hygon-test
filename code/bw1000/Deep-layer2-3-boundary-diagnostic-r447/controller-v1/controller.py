"""CPU keeper for one self-terminating fixed-prefix diagnostic, not a service."""
import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import time
import traceback
from support import D, LABEL, NODE, R, RUN, TERMINAL, clean_environment, publish, sha, verify
from owned_cleanup import close_owned


def gpu_command():
    return ['srun', '--jobid=910374', '--overlap', '--exact', '--immediate=30',
            '--cpu-bind=none', '--kill-on-bad-exit=1', '-N1', '-n1', '--ntasks-per-node=1',
            '-c32', '--gres=dcu:8', '--mem=128000M', '-w', NODE, '-t', '00:45:00',
            '--job-name=' + LABEL + '-' + RUN, '--export=ALL',
            '--output=' + str(R / 'logs' / (LABEL + '-' + RUN + '-%J-%t.log')),
            'bash', str(D / 'resource_wave_host_alias_outer.sh'), 'probe']


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--manifest-sha', required=True)
    p.add_argument('--publication-commit', required=True)
    p.add_argument('--source-only', action='store_true')
    a = p.parse_args(argv)
    assert re.fullmatch('[0-9a-f]{40}', a.publication_commit)
    m = verify(a.manifest_sha, require_ready=not a.source_only)
    if a.source_only:
        print(json.dumps(dict(source_verified=True, Slurm=0, GPU=0, Native_dlopen=False)))
        return 0
    assert os.environ['SLURM_JOB_ID'] == '910407' and socket.gethostname().split('.')[0] == 'a14r2n20'
    cleanup_limit = time.monotonic() + 2980
    watch = R / 'results' / (LABEL + '-' + RUN + '-watch.json')
    output = R / 'results/Deep447-boundary-private' / RUN
    assert not output.exists() and not watch.exists(), 'Reconcile original run; no replay'
    w = dict(completed=False, passed=False, actual_GPU=None,
             CPU_keeper=os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'],
             publication_commit=a.publication_commit, source_manifest_SHA256=a.manifest_sha,
             HTTP=0, max_sequences=2, max_forwards=184, GPU_cards=8,
             reload=False, retry=False, forced_cleanup=False)
    publish(watch, w)
    gpu = None;epoch = None;visibility_since = None

    def save():
        w['observed_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = watch.with_suffix('.tmp')
        tmp.write_text(json.dumps(w, indent=2, allow_nan=False) + '\n');tmp.replace(watch)

    def command(args, timeout=25):
        return subprocess.check_output(args, text=True, timeout=min(timeout, max(1, cleanup_limit - time.monotonic())))

    def rows():
        return [r.split('|') for r in command(['sacct', '-S', '2026-10-31', '-j', '910374', '-n', '-P',
                   '-o', 'JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines() if '|' in r]

    def hosts(value):
        return command(['scontrol', 'show', 'hostnames', value]).splitlines()

    def exact(label):
        found = [r for r in rows() if r[1] == label]
        assert len(found) <= 1, 'Duplicate diagnostic label'
        if found:
            assert re.fullmatch(r'910374\.\d+', found[0][0]) and hosts(found[0][4]) == [NODE]
        return found[0] if found else None

    def idle(phase):
        env = clean_environment(os.environ)
        env.update(CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
                   DEEP438_DIAG_POST_DEADLINE=str(min(time.time() + 90, epoch + 2880 if epoch else time.time() + 90)))
        cmd = ['srun', '--jobid=910374', '--overlap', '--exact', '--immediate=30', '--cpu-bind=none',
               '-N1', '-n1', '-c1', '--gres=none', '--mem=512M', '-w', NODE, '-t', '00:02:00',
               '--job-name=' + LABEL + '-' + RUN + '-' + phase, 'python3', '-B', str(D / 'idle.py')]
        z = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           timeout=min(115, max(1, cleanup_limit - time.monotonic())))
        w.setdefault('CPU_idle_observations', []).append(dict(phase=phase, rc=z.returncode, stdout=z.stdout, stderr=z.stderr));save()
        q = [json.loads(line) for line in z.stdout.splitlines() if line.startswith('{')]
        assert z.returncode == 0 and len(q) == 1 and q[0]['node'] == NODE
        q = q[0]
        assert q['CPU_only'] and not q['GPU_initialized'] and not q['kfd_pids'] and not q['matching_component_processes']
        assert q['ports_bindable'] == {'43882': True} and len(set(q['physical_PCI'])) == 8
        return q

    try:
        parent = command(['scontrol', 'show', 'job', '910374', '-o'])
        assert 'JobState=RUNNING' in parent and NODE in hosts(re.search(r'(?:^| )NodeList=([^ ]+)', parent).group(1))
        assert re.search(r'(?:^| )UserId=' + re.escape(os.environ['USER']) + r'\(', parent)
        w['parent_allocation'] = parent
        for row in rows():
            assert not row[1].startswith(LABEL + '-' + RUN), 'Existing label prohibits new submission'
            if re.fullmatch(r'910374\.\d+', row[0]) and row[2].split()[0].rstrip('+') not in TERMINAL:
                assert NODE not in hosts(row[4]), 'Live overlapping step'
        w['fresh_before'] = idle('before')
        epoch = time.time()
        w.update(original_GPU_start_unix=epoch, GPU_deadline_unix=epoch + 2700, keeper_deadline_unix=epoch + 3000)
        save()
        env = clean_environment(os.environ)
        env.update(WAVE_INSIDE_R2323='Deep-layer2-3-boundary-diagnostic-r447/controller-v1/inside.sh',
                   DEEP438_DIAG_START=str(epoch), DEEP438_DIAG_CONTROLLER_SHA=a.manifest_sha,
                   INFINILM_OWNER_CPU_WIDTH_R2392='4')
        with (R / 'logs' / (LABEL + '-' + RUN + '-srun.log')).open('x') as log:
            gpu = subprocess.Popen(gpu_command(), env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        w.update(GPU_srun_PID=gpu.pid, GPU_owned_PGID=gpu.pid);save()
        terminal_since = None
        while time.time() < epoch + 2700:
            try:
                row = exact(LABEL + '-' + RUN)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as error:
                if gpu.poll() not in (None, 0):
                    raise RuntimeError('Owned srun failed during accounting observation loss') from error
                w.setdefault('transient_accounting_errors', []).append(repr(error));save();time.sleep(15);continue
            if row:
                assert w['actual_GPU'] in (None, row[0])
                w.update(actual_GPU=row[0], actual_GPU_Slurm=row)
                if row[2].split()[0].rstrip('+') in TERMINAL:
                    assert row[2:4] == ['COMPLETED', '0:0'], ('Diagnostic original step failed', row)
            assert gpu.poll() in (None, 0), 'Owned GPU srun nonzero'
            if gpu.poll() == 0:
                terminal_since = time.monotonic() if terminal_since is None else terminal_since
                path = output / 'entry-report.json'
                try:
                    result = json.loads(path.read_bytes())
                except (FileNotFoundError, json.JSONDecodeError):
                    visibility_since = time.monotonic() if visibility_since is None else visibility_since
                    assert time.monotonic() - visibility_since <= 90, 'Terminal report visibility exceeded original bound'
                else:
                    assert result['completed'] and result['passed']
                    if w['actual_GPU'] is None:
                        assert time.monotonic() - terminal_since <= 90, 'Original step accounting not yet visible'
                        w['report_step_awaiting_accounting'] = result['actual_step'];save();time.sleep(15);continue
                    assert result['actual_step'] == w['actual_GPU']
                    assert result['gpu_start_epoch'] == epoch and result['HTTP'] == result['free_generation'] == 0
                    assert result['normal_Native_shutdown'] and result['engine_worker_shutdown_completed'] and result['observed_all8_threads_exited']
                    d = result['diagnostic']
                    typed_path = output / 'typed/report.json'
                    typed = json.loads(typed_path.read_bytes())
                    assert sha(typed_path) == result['diagnostic']['SHA256']
                    assert typed['planned_modes'] == [0, 0] and typed['max_sequences'] == 2 and typed['max_forwards'] == 184
                    assert 0 < len(typed['sequences']) <= 2 and typed['raw_budget_used'] <= 33554432
                    assert all(row['mode'] == 0 for row in typed['sequences'])
                    assert d['completed'] and d['passed'] and 0 < d['completed_forwards'] <= d['actual_forward_attempts'] <= 184
                    if not d['early_stop']:
                        assert d['completed_forwards'] == d['actual_forward_attempts'] == 184
                    if row and row[2:4] == ['COMPLETED', '0:0']:
                        w.update(entry_report_SHA256=sha(path), diagnostic=d, normal_Native_shutdown=True)
                        break
                assert time.monotonic() - terminal_since <= 90, 'srun/accounting/report propagation exceeded90s'
            save();time.sleep(min(15, max(0, epoch + 2700 - time.time())))
        else:
            raise TimeoutError('Original diagnostic GPU2700 deadline; no replay')
        w['post'] = idle('post')
        w.update(completed=True, passed=True, state='DIAGNOSTIC_FINISHED_NORMAL_AND_POST_CLEAN_NO_MODEL_QUALIFICATION')
    except BaseException as error:
        w.update(completed=True, passed=False, error=repr(error), traceback=traceback.format_exc(), state='FAILED_OR_UNKNOWN_NO_RETRY')
        if gpu is not None:
            w['forced_cleanup'] = True
            try:
                w['owned_cleanup'] = close_owned(LABEL + '-' + RUN, gpu, w['actual_GPU'], exact, cleanup_limit)
            except BaseException as cleanup:
                w['cleanup_error'] = repr(cleanup)
        try:
            w['post_after_failure'] = idle('failure-post')
        except BaseException as cleanup:
            w['post_error'] = repr(cleanup)
    finally:
        save()
    print(json.dumps({k: w.get(k) for k in ['completed', 'passed', 'actual_GPU', 'CPU_keeper', 'state', 'error']}))
    return 0 if w['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
