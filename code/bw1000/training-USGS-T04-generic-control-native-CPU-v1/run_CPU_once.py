"""One bounded CPU worker under the already qualified owned-Popen helpers."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from prepare_inputs import D, canonical, need, verify_source
from owned_processes import snapshot, refresh, cleanup
from held_child import cleanup_held


def schedule(epoch, now):
    need(all(math.isfinite(v) for v in [epoch, now]) and 0 < epoch <= now < epoch + 260,
         'original CPU epoch expired before worker')
    return dict(epoch=epoch, deadline=epoch + 300, work_deadline=epoch + 260,
                supervisor_stop=epoch + 270, held_cleanup_deadline=epoch + 282,
                registry_cleanup_deadline=epoch + 285, alarm_deadline=epoch + 290,
                client_deadline=epoch + 295)


def device_fds(pid):
    rows = []
    for item in (Path('/proc') / str(pid) / 'fd').iterdir():
        try: target = os.readlink(item)
        except FileNotFoundError: continue
        if target == '/dev/kfd' or target.startswith(('/dev/nvidia', '/dev/dri/renderD')):
            rows.append(int(item.name))
    return rows


def rss(pid):
    for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            value = line.split(); need(value[2] == 'kB', 'RSS units differ'); return int(value[1]) * 1024
    return 0  # A known zombie can lack VmRSS; process identity is checked separately.


def main(a):
    verify_source()
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    need(claim['root_GO'] is True and claim['CPU_GO'] is True, 'published CPU GO required')
    need(len(a.request_id) == 32 and not set(a.request_id) - set('0123456789abcdef')
         and len(a.publication) == 40 and not set(a.publication) - set('0123456789abcdef'), 'original CPU context malformed')
    clock = schedule(a.epoch, time.time()); mono = time.monotonic() + clock['deadline'] - time.time()
    scope = a.scope
    need(scope.is_absolute() and scope.is_dir() and not scope.is_symlink() and scope.resolve() == scope
         and scope.stat().st_uid == os.getuid(), 'root-staged own CPU scope required')
    controller = scope / 'controller'; native = scope / 'native'
    need(not controller.exists() and not native.exists(), 'original CPU namespace exists: reconcile, never replay')
    controller.mkdir(mode=0o700)
    def save(name, value):
        raw = canonical(value); need(len(raw) <= 2 * 1024**2, 'controller metadata cap exceeded')
        with (controller / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    affinity = sorted(os.sched_getaffinity(0) - {0, 1}); need(len(affinity) >= 2, 'two allowed CPU cores excluding0/1 unavailable')
    os.sched_setaffinity(0, set(affinity[:2])); need(os.sched_getaffinity(0) == set(affinity[:2]), 'actual CPU affinity differs')
    fields = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    owner = dict(PID=os.getpid(), PGID=os.getpgrp(), startticks=int(fields[19]), uid=os.getuid(),
                 host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    need(not device_fds(os.getpid()), 'controller already holds device FDs')
    argv = [sys.executable, '-B', str(D / 'native_template_CPU.py'), '--case', a.case,
            '--framework', str(a.framework), '--native-binding', str(a.native_binding), '--model-view', str(a.model_view),
            '--data', str(a.data), '--data-binding-sha', a.data_binding_sha, '--data-binding-bytes', str(a.data_binding_bytes),
            '--output', str(native), '--epoch', str(a.epoch), '--deadline', str(clock['work_deadline'])]
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    save('intent.json', dict(case=a.case, request_id=a.request_id, publication=a.publication, **clock, owner=owner,
        CPU_affinity=affinity[:2], GPU=0, command=argv, source_manifest_SHA256=hashlib.sha256((D / 'SOURCE_FILES.json').read_bytes()).hexdigest(),
        RSS_stop_threshold_bytes=8 * 1024**3, RSS_is_hard_limit=False))
    def expired(*_): raise TimeoutError('original CPU controller alarm exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, min(clock['alarm_deadline'] - time.time(), mono - time.monotonic() - 10)))
    child = None; anchor = None; registry = {}; failure = None; peak = 0; samples = 0; stopped = False
    cleanup_result = None; held = None; post = None
    try:
        need(time.time() < clock['work_deadline'] - 5, 'no remaining CPU worker budget')
        with (controller / 'worker.stdout').open('xb') as out, (controller / 'worker.stderr').open('xb') as err:
            child = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=True)
            anchor = dict(PID=child.pid, PGID=child.pid, start_new_session=True, supervisor=owner, request_id=a.request_id)
            save('held-child-anchor.json', anchor)  # Strong ownership before snapshot.
            row = snapshot().get(child.pid)
            need(row is not None, 'owned CPU child snapshot absent; held cleanup still mandatory')
            registry[child.pid] = {k: row[k] for k in ['PID', 'PPID', 'PGID', 'startticks', 'state']}
            save('child-owner.json', registry[child.pid])
            while child.poll() is None:
                live = refresh(registry, scope); samples += 1
                measured = rss(os.getpid())
                for pid in live:
                    try: measured += rss(pid)
                    except FileNotFoundError: continue
                peak = max(peak, measured)
                temp = controller / 'owned-registry.tmp'; temp.write_bytes(canonical(registry)); temp.replace(controller / 'owned-registry.json')
                left = min(clock['supervisor_stop'] - time.time(), mono - time.monotonic() - 30)
                if measured > 8 * 1024**3 or left <= 0:
                    stopped = True; failure = dict(type='RSS_OR_DEADLINE_STOP', sampled_group_RSS=measured); break
                try: child.wait(timeout=min(.25, left))
                except subprocess.TimeoutExpired: pass
    except BaseException as error:
        failure = dict(type=type(error).__name__, reason=str(error)[:200])
    finally:
        # Helper bytes are reused unchanged: TERM grace may be10 seconds,
        # always shrinking to the same origin's held282/registry285 reserves.
        held = cleanup_held(child, anchor, clock['held_cleanup_deadline'])
        cleanup_result = cleanup(registry, scope, grace=max(0, min(1.0, (clock['registry_cleanup_deadline'] - time.time()) / 2)))
        if child is not None and child.returncode is None:
            try: child.wait(timeout=max(0, min(1, clock['registry_cleanup_deadline'] - time.time())))
            except subprocess.TimeoutExpired: pass
        held['reaped'] = child is None or child.returncode is not None
        held['returncode'] = child.returncode if child else None
        try:
            live = refresh(registry, scope)
            post = dict(same_host=socket.gethostname() == owner['host'],
                        same_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip() == owner['boot_id'],
                        controller_device_FDs=device_fds(os.getpid()), owned_descendants_remaining=len(live),
                        held_child_reaped=held['reaped'], owned_device_FD_holders={str(pid): device_fds(pid) for pid in live})
            save('post.json', post)
        except BaseException as error:
            post = dict(state='POST_UNKNOWN', error_type=type(error).__name__)
        native_result = None; path = native / 'result.json'
        if path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 4 * 1024**2:
            raw = path.read_bytes(); value = json.loads(raw)
            native_result = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest(), passed=value.get('passed'),
                completed_train=value.get('completed_train', 0), completed_dev=value.get('completed_dev', 0))
        passed = bool(child and child.returncode == 0 and failure is None and not stopped and held['reaped']
            and not held['termination_needed'] and cleanup_result['owned_processes_remaining'] == 0
            and not cleanup_result['termination_needed'] and post.get('same_host') and post.get('same_boot')
            and post.get('controller_device_FDs') == [] and post.get('owned_descendants_remaining') == 0
            and native_result and native_result['passed'] is True
            and native_result['completed_train'] == 45 and native_result['completed_dev'] == 0)
        outcome = dict(case=a.case, request_id=a.request_id, publication=a.publication, **clock, finished=time.time(),
            owner=owner, worker_returncode=child.returncode if child else None, failure=failure, stopped=stopped,
            observed_group_RSS_peak_bytes=peak, RSS_samples=samples, RSS_is_hard_limit=False,
            held_cleanup=held, registry_cleanup=cleanup_result, post=post, native_result=native_result,
            passed=passed, actual_updates=0, GPU=0, training_GO=False, GPU_GO=False, Model_GO=False)
        save('outcome.json', outcome); signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(case=a.case, request_id=a.request_id, passed=passed, GPU=0,
                         worker_returncode=outcome['worker_returncode'], native_result=native_result)))
    return 0 if passed else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=['T03.sft', 'T04.sft'], required=True)
    for name in ['framework', 'native-binding', 'model-view', 'data', 'scope']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--data-binding-sha', required=True); p.add_argument('--data-binding-bytes', type=int, required=True)
    p.add_argument('--request-id', required=True); p.add_argument('--publication', required=True)
    p.add_argument('--epoch', type=float, required=True)
    raise SystemExit(main(p.parse_args()))
