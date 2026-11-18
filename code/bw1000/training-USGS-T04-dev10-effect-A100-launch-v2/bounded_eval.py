"""Single original-epoch evaluator; execute qualified RSS-v2 ownership functions."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from gpu_admission import snapshot as gpu_snapshot, AdmissionFailure

D = Path(__file__).resolve().parent
QUALIFIED = '2307123f57756f3a7a42153176b2e3acb068f6e0a51e5756730468236bbf0e2f'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fresh_before(before_monotonic, now_monotonic):
    need(0 <= now_monotonic-before_monotonic <= 60, 'SAME_HOST_MONOTONIC_ADMISSION_STALE')


def qualified():
    path = D / 'qualified/bounded_GPU_supervisor.py'
    need(sha(path.read_bytes()) == QUALIFIED, 'ACTUAL_QUALIFIED_RSS_V2_SOURCE')
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('qualified_RSS_v2', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Same tested RSS classifier and sampler; only the declared soft threshold changes.
    module.RSS_SOFT_BYTES = 64 * 1024**3
    return module


def main(a):
    q = qualified(); clock = q.schedule(a.epoch, time.time())
    raw, mfpin = q.stable(D / 'SOURCE_FILES.json', 256 * 1024)
    need(sha(raw) == a.source_manifest_sha256, 'PUBLISHED_OUTER_SOURCE')
    members = json.loads(raw)['files']
    for name, pin in members.items():
        q.stable(D / name, 2 * 1024**2, pin)
    claim = json.loads((D / 'CLAIM.json').read_bytes()); plan = json.loads((D / 'INSTALL_PLAN.json').read_bytes())
    need(sha((D / 'CLAIM.json').read_bytes()) == a.claim_sha256 and claim['root_GO'] is claim['GPU_GO'] is True
         and claim['max_invocations'] == 1, 'ROOT_PUBLISHED_ONE_SCOPE')
    need(a.scope == Path(plan['namespace']) / 'run' and not a.scope.exists(), 'EXCLUSIVE_EXACT_RUN_NAMESPACE')
    need(time.time() < a.epoch + 760, 'RESERVE_NATIVE_AND_CLEANUP_BUDGET')
    root = a.scope; root.mkdir(mode=0o700)
    registry = {}; child = None; anchor = None; failure = None; post = None; native = None
    clean = dict(owned_processes_remaining=None, termination_needed=False)
    resources = dict(inherited_affinity=sorted(os.sched_getaffinity(0)), samples=0,
        sampled_group_peak_RSS_bytes=0, soft_limit_bytes=q.RSS_SOFT_BYTES,
        soft_limit_exceeded=False, hard_limit=False, last_sample=None,
        method='Qualified RSS-v2 PID/startticks VmRSS classifier; sampled lower bound, not cgroup peak')
    def expired(*_):
        raise TimeoutError('SAME900_EPOCH_OR_SIGNAL')
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired); signal.signal(signal.SIGINT, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, a.epoch + 890 - time.time()))
    phase = 'controller_identity'; released = False; timed_out = False
    try:
        owner = q.own_identity(a.request_id)
        control = dict(request_id=a.request_id, publication=a.publication,
            epoch=a.epoch, whole_deadline=clock['deadline'], native_deadline=clock['native_deadline'],
            source_manifest_SHA256=a.source_manifest_sha256, supervisor=owner,
            exec_request_attempted=False, entry_released=False)
        q.save(root, 'control-state.json', control)
        allowed = sorted(set(os.sched_getaffinity(0)) - {0, 1})
        need(len(allowed) >= 4, 'FOUR_AVAILABLE_CPU_EXCLUDING_ZERO_ONE')
        os.sched_setaffinity(0, set(allowed[:4])); resources['inherited_affinity'] = allowed[:4]
        q.affinity_contract(os.sched_getaffinity(0))
        q.save(root, 'resource-samples.json', resources)
        core_raw, _ = q.stable(D / 'core/SOURCE_FILES.json', 256 * 1024)
        need(sha(core_raw) == plan['core_manifest_SHA256'], 'EXACT_INSTALLED_CORE_MANIFEST')
        core_plan = json.loads((D / 'core/PLAN.json').read_bytes())
        core_claim = json.loads((D / 'core/CLAIM.json').read_bytes())
        need(all(x is True for x in [plan['root_GO'], plan['GPU_GO'], core_plan['root_GO'], core_plan['GPU_GO'],
             core_claim['root_GO'], core_claim['GPU_GO']]), 'ALL_EIGHT_ADOPTION_FLAGS')
        need(not (root / 'native').exists(), 'WORKER_OUTPUT_INITIALLY_ABSENT')
        phase = 'passive_admission'
        model_admission = q.verify_model(dict(model_admission=plan['model_admission']),
            lambda _: need(time.time() < clock['native_deadline'] - 30, 'MODEL_ADMISSION_DEADLINE'))
        q.save(root, 'original-model-inventory-admission.json', model_admission)
        before = gpu_snapshot(plan['expected_GPU'], plan['master_port'], min(a.epoch + 760, time.time() + 20))
        q.save(root, 'before.json', before)
        env = os.environ.copy()
        need(not any(key in env for key in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK']), 'UNRANKED_PARENT_REQUIRED')
        env.update(CUDA_VISIBLE_DEVICES=plan['expected_GPU']['UUID'], PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline')
        admission_path = root / 'worker-admission.json'
        command = [sys.executable, '-c', q.BOOTSTRAP, str(D / 'worker_entry.py'), '--framework', plan['framework_root'],
            '--data', core_plan['data_root'], '--identities', core_plan['native_identities']['path'],
            '--admission', str(admission_path), '--admission-bytes', 'PENDING_BYTES', '--admission-sha', 'PENDING_SHA',
            '--epoch', str(a.epoch), '--deadline', str(clock['native_deadline']), '--output', str(root / 'native')]
        old_exec = 'os.execv(sys.executable,[sys.executable]+sys.argv[1:])'
        need(q.BOOTSTRAP.count(old_exec) == 1, 'ONE_QUALIFIED_HELD_EXEC_REPLACEMENT')
        command[2] = q.BOOTSTRAP.replace(old_exec,
            "import json; pin=json.loads(sys.stdin.buffer.readline(256)); args=sys.argv[1:]; args[args.index('--admission-bytes')+1]=str(pin['bytes']); args[args.index('--admission-sha')+1]=pin['SHA256']; os.execv(sys.executable,[sys.executable]+args)")
        phase = 'held_worker_start'
        with (root / 'native.stdout').open('xb') as out, (root / 'native.stderr').open('xb') as err:
            control['exec_request_attempted'] = True
            q.save(root, 'control-state.json', control, replace=True)
            child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True)
            anchor = dict(PID=child.pid, PGID=child.pid, start_new_session=True, request_id=a.request_id)
            q.save(root, 'held-child-anchor.json', anchor)
            row = q.snapshot().get(child.pid)
            need(row is not None and row['PPID'] == os.getpid() and row['PGID'] == child.pid, 'EXACT_NEW_HELD_CHILD')
            registry[child.pid] = {k: row[k] for k in ['PID','PPID','PGID','startticks','state']}
            owned = dict(**anchor, startticks=row['startticks'], UID=os.getuid(), host=owner['host'], boot_id=owner['boot_id'])
            q.save(root, 'child-owner.json', dict(child=owned, supervisor=owner))
            admission = dict(passed=True, request_id=a.request_id, epoch=a.epoch, deadline=clock['native_deadline'],
                whole_deadline=clock['deadline'], physical_GPU_count=1, unique_owned_PCI_KFD_PID_port_checked=True,
                source_manifest_SHA256=plan['core_manifest_SHA256'], selected_GPU_UUID=plan['expected_GPU']['UUID'],
                CPU_affinity=allowed[:4], supervisor=owner, owned_child=owned,
                parent_source_SHA256=sha((D / 'bounded_eval.py').read_bytes()), before=before)
            pin = q.save(root, 'worker-admission.json', admission)
            q.save(root, 'exec-intent.json', dict(request_id=a.request_id, epoch=a.epoch,
                native_deadline=clock['native_deadline'], admission=pin, child=owned, supervisor=owner,
                planned_arms=3, planned_real_generations=30, actual_training_updates=0))
            need(time.time() < a.epoch + 760, 'DO_NOT_RELEASE_LATE_WORKER')
            fresh_before(before['observed_monotonic'], time.monotonic())
            child.stdin.write(b'GO\n' + json.dumps(dict(bytes=pin['bytes'], SHA256=pin['SHA256'])).encode() + b'\n')
            child.stdin.flush(); child.stdin.close(); released = True; phase = 'native_inference'
            control['entry_released'] = True
            q.save(root, 'control-state.json', control, replace=True)
            while child.poll() is None:
                q.refresh(registry, root); q.save(root, 'owned-registry.json', registry, replace=True)
                try:
                    below = q.sample_group_rss(registry, resources)
                finally:
                    q.save(root, 'resource-samples.json', resources, replace=True)
                need(below, 'OWNED_GROUP_RSS64_SOFT_STOP')
                need(all((root / name).stat().st_size <= 16 * 1024**2 for name in ['native.stdout','native.stderr']), 'NATIVE_LOG_BOUND')
                left = clock['supervisor_stop'] - time.time()
                if left <= 0:
                    timed_out = True; break
                try:
                    child.wait(timeout=min(1, left))
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as error:
        failure = dict(phase=phase, error_type=type(error).__name__, message_SHA256=sha(str(error).encode()))
        if isinstance(error, AdmissionFailure):
            failure['admission_error_code'] = error.code
            failure['query_returncode'] = error.diagnostic.get('returncode')
            failure['query_stdout_SHA256'] = error.diagnostic.get('stdout_SHA256')
            failure['query_stderr_SHA256'] = error.diagnostic.get('stderr_SHA256')
            try:
                failure['admission_diagnostic_private'] = q.save(root, 'admission-query-failure-private.json', error.diagnostic)
            except BaseException:
                failure['admission_diagnostic_private'] = None
        try:
            failure['private_file'] = q.save(root, 'failure-private.json', dict(phase=phase,
                error_type=type(error).__name__, message=str(error)[:4096], trace=traceback.format_exc()[-4096:]))
        except BaseException:
            failure['private_file'] = None
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            clean['held_child'] = q.cleanup_held(child, anchor, a.epoch + 880)
        except BaseException as error:
            clean['held_child'] = dict(reaped=False, error_type=type(error).__name__)
        try:
            clean.update(q.cleanup(registry, root, grace=max(0, min(10, (a.epoch + 880-time.time())/2))))
        except BaseException as error:
            clean['tree_error_type'] = type(error).__name__
        clean['termination_needed'] |= clean['held_child'].get('termination_needed', False)
        try:
            post = gpu_snapshot(plan['expected_GPU'], plan['master_port'], min(a.epoch + 889, clock['deadline']))
            q.save(root, 'after.json', post)
        except BaseException as error:
            post = dict(state='POST_UNKNOWN', error_type=type(error).__name__)
            if isinstance(error, AdmissionFailure):
                post.update(admission_error_code=error.code, query_returncode=error.diagnostic.get('returncode'),
                    query_stdout_SHA256=error.diagnostic.get('stdout_SHA256'), query_stderr_SHA256=error.diagnostic.get('stderr_SHA256'))
                try:
                    post['admission_diagnostic_private'] = q.save(root, 'post-query-failure-private.json', error.diagnostic)
                except BaseException:
                    post['admission_diagnostic_private'] = None
        try:
            body, pin = q.stable(root / 'native/generation-finalized.json', 2 * 1024**2)
            value = json.loads(body)
            need(value['passed'] is True and value['all_predictions_finalized_before_gold'] is True
                 and value['all_prediction_files_closed_before_gold'] is True
                 and set(value['arm_results']) == {'initial','domain25','generic25'}, 'THREE_FINALIZED_ARMS')
            for arm, record in value['arm_results'].items():
                need(record['completed'] == record['attempted'] == 10 and record['missing'] == 0, 'THIRTY_REAL_PREDICTIONS')
                q.stable(root / 'native' / record['file'], 2 * 1024**2, record)
            native = dict(file=pin, arms=3, predictions=30, original_domain_training_accepted=False,
                          dev_gold_read=False, blind_opened=False)
        except BaseException as error:
            native = dict(state='NATIVE_PREDICTIONS_INCOMPLETE_OR_FAILED', error_type=type(error).__name__, planned_arms=3, planned_predictions=30)
        clean_exit = bool(clean['held_child'].get('reaped') is True and clean['owned_processes_remaining'] == 0
            and post.get('active_compute_processes') == 0 and post.get('master_port_bindable') is True)
        passed = bool(child is not None and child.returncode == 0 and released and failure is None and not timed_out
            and not clean['termination_needed'] and clean_exit and native.get('predictions') == 30)
        outcome = dict(request_id=a.request_id, publication=a.publication, **clock,
            source_manifest_SHA256=a.source_manifest_sha256, root_owned_scope=str(root),
            native_returncode=child.returncode if child else None, native_released=released, failure=failure,
            timed_out=timed_out, cleanup=clean, post=post, resources=resources, native=native,
            engineering_completed=passed, release_permitted=clean_exit, actual_training_updates=0,
            planned_generations=30, labels_read=False, blind_opened=False,
            original_domain_training_accepted=False, effect_qualified=False, formal17=False, finished_epoch=time.time())
        q.save(root, 'outcome.json', outcome); signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(request_id=a.request_id, engineering_completed=passed,
        release_permitted=clean_exit, native_returncode=outcome['native_returncode'], planned_generations=30)))
    return 0 if passed else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source-manifest-sha256','claim-sha256','request-id','publication']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--scope', type=Path, required=True); p.add_argument('--epoch', type=float, required=True)
    raise SystemExit(main(p.parse_args()))
