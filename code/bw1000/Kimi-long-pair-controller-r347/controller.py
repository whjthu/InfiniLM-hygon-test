"""Single target residency, two AR arms, then one authenticated draft residency."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

from contract import (CPU_NODE, TARGET_NODES, DRAFT_NODE, GPU_SECONDS, NATIVE_SECONDS,
                      TERMINAL, NATIVE_SHA, WORKER_SHA, marker_identity, publish_once,
                      read_start, sha, validate_plan, verify_bundle)
from slurm_backend import SlurmBackend

ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
DIRECTORY = Path(__file__).resolve().parent
DEPENDENCIES = {
    'Kimi-long-draft-worker-r334b': WORKER_SHA,
    'Kimi-long-draft-RPC-CPU-r333': '358016b6b9e334817d2d4929ed613d67548dcd6764a20f2372b7748be3c99a2e',
    'Kimi-long-target-publisher-r335': '5e14dfd4945987141b9da42b3733fbae7e75aae277f9854b0c755e7ffd8e4ab8',
    'Kimi-long-pair-driver-r337': '445f96b37c1ceab9e62c3599879834b8be21da9b9585384c942c12ffef86644f',
}


def verify_sources(root, directory, plan):
    validate_plan(plan, execution=True)
    manifest = verify_bundle(directory, plan['controller_manifest_SHA256'])
    for name, expected in DEPENDENCIES.items():
        verify_bundle(root/'incoming'/name, expected)
    verify_bundle(root/'incoming/Kimi-long-target-bootstrap-r346', plan['target_manifest_SHA256'])
    target_cpu = root/'candidates/Kimi-long-target-bootstrap-r346/CPU-result.json'
    assert sha(target_cpu) == plan['target_CPU_receipt_SHA256']
    target_check = json.loads(target_cpu.read_bytes())
    assert target_check['completed'] and target_check['passed'] and target_check['GPU'] == 0
    assert target_check['Native_dlopen'] is False
    assert target_check['source_manifest_SHA256'] == plan['target_manifest_SHA256']
    assert sha(directory/'outer.sh') == manifest['outer_wrapper_SHA256']
    receipt = root/'candidates/Kimi-long-pair-controller-r347/CPU-result.json'
    assert sha(receipt) == plan['CPU_receipt_SHA256']
    q = json.loads(receipt.read_bytes())
    assert q['completed'] and q['passed'] and q['GPU'] == 0 and q['Native_dlopen'] is False
    assert q['controller_manifest_SHA256'] == plan['controller_manifest_SHA256']
    artifact = root/'candidates/Kimi-long-capture-link-CPU-r327/CPU-build.json'
    assert sha(artifact) == 'e60be9f7305c7024affd82a662c4d11b7732fa4eb0acea497d57ca6cae964664'
    bound = json.loads(artifact.read_bytes())
    assert bound['completed'] and bound['passed'] and bound['native_SHA256'] == NATIVE_SHA
    # Container path maps to the fixed host workspace; no arbitrary filename execution.
    assert bound['native'] == '/refactor/candidates/Kimi-long-capture-link-CPU-r327/libKimi_long_capture_r327.so'
    assert sha(root/'candidates/Kimi-long-capture-link-CPU-r327/libKimi_long_capture_r327.so') == NATIVE_SHA
    return manifest


def state_of(observation):
    row = observation.get('row')
    return row[2].split()[0].rstrip('+') if row else None


def terminal_success(observation):
    row = observation.get('row')
    return (state_of(observation) == 'COMPLETED' and row[3] == '0:0'
            and observation['srun_exit_code'] == 0)


def terminal_failure(observation):
    row = observation.get('row')
    if observation['srun_exit_code'] not in (None, 0): return True
    return state_of(observation) in TERMINAL and (state_of(observation) != 'COMPLETED' or row[3] != '0:0')


class Controller:
    """The real orchestration is also exercised with a recording Slurm boundary."""
    def __init__(self, root, plan, backend, nonce, *, clock=time.time, sleep=time.sleep):
        self.root, self.plan, self.backend = Path(root), plan, backend
        self.clock, self.sleep = clock, sleep
        assert isinstance(nonce, str) and len(nonce) >= 32
        self.nonce = nonce
        self.nonce_hash = hashlib.sha256(nonce.encode()).hexdigest()
        run = plan['run']
        self.watch = self.root/'results'/('Kimi-long-pair-r347-'+run+'-watch.json')
        self.start_marker = self.root/'control'/('Kimi-long-draft-start-r347-'+run+'.json')
        self.listening_marker = self.root/'control'/('Kimi-long-draft-listening-r347-'+run+'.json')
        self.cancel_marker = self.root/'control'/('Kimi-long-pair-cancel-r347-'+run+'.json')
        self.q = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      completed=False, passed=False, target_submitted=False, draft_submitted=False,
                      run=run, nonce_SHA256=self.nonce_hash, scope='Four engine-level requests; no clientSLO',
                      publication_commit=plan['publication_commit'], target_cards=32, draft_cards=1,
                      retry=False, reload=False, source_manifest_SHA256=plan['controller_manifest_SHA256'])
        self.epoch = None
        self.exit_seen = {}
        self.terminal_seen = {}

    def save(self):
        self.q['observed_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temp = self.watch.with_suffix('.tmp')
        temp.write_text(json.dumps(self.q, indent=2)+'\n', encoding='utf8')
        temp.replace(self.watch)

    def observation(self, kind):
        value = self.backend.observe(kind)
        self.q[kind+'_observation'] = value
        if value['row']:
            self.q['actual_'+kind] = value['row'][0]
        if value['srun_exit_code'] is not None and state_of(value) not in TERMINAL:
            self.exit_seen.setdefault(kind, self.clock())
            if self.clock()-self.exit_seen[kind] > 90:
                raise RuntimeError('Owned srun exited but accounting unresolved; no replay')
        if state_of(value) in TERMINAL and value['srun_exit_code'] is None:
            self.terminal_seen.setdefault(kind, self.clock())
            if self.clock()-self.terminal_seen[kind] > 90:
                raise RuntimeError('Owned accounting terminal but srun exit unresolved')
        return value

    def abort_owned(self, reason):
        if not self.cancel_marker.exists():
            publish_once(self.cancel_marker, dict(run=self.plan['run'], nonce_SHA256=self.nonce_hash,
                                                reason=reason, cancelled=True))
        self.q['owned_forced_cleanup'] = self.backend.cancel_owned()

    def run(self):
        validate_plan(self.plan, execution=True)
        # Even a pre-existing empty marker means another or failed original attempt.
        if any(p.exists() for p in [self.watch, self.start_marker, self.listening_marker, self.cancel_marker]):
            raise RuntimeError('Existing exact run must be followed; never overwrite or replay')
        publish_once(self.watch, self.q)
        try:
            self.backend.validate_resources()
            before = self.backend.idle('before')
            self.q['fresh_before'] = before
            self.backend.no_foreign()
            heads = {r['node']: r['ib_ipv4'][0] for r in before}
            assert set(heads) == set(TARGET_NODES+[DRAFT_NODE])
            self.epoch = self.clock()
            assert math.isfinite(self.epoch)
            self.q['original_GPU_start_unix'] = self.epoch
            expected = marker_identity(self.plan['run'], self.nonce_hash, self.epoch)
            environment = dict(KIMI_LONG_PAIR_R346='1', KIMI_LONG_SESSION_R347=self.nonce,
                KIMI_LONG_RUN_R347=self.plan['run'], KIMI_LONG_GPU_START_UNIX_R347=str(self.epoch),
                KIMI_LONG_NATIVE_DEADLINE_UNIX_R347=str(self.epoch+NATIVE_SECONDS),
                KIMI_LONG_CANCEL_FILE_R347='/refactor/control/'+self.cancel_marker.name,
                KIMI_LONG_DRAFT_HOST_R347=heads[DRAFT_NODE], KIMI_LONG_DRAFT_PORT_R347='46947',
                KIMI_LONG_CAPTURE_PORT_R347='45947', KIMI_LONG_TARGET_MANIFEST_SHA_R347=self.plan['target_manifest_SHA256'],
                KIMI_ENGINE_HEAD=heads[TARGET_NODES[0]], KIMI_ENGINE_NONCE=self.nonce,
                GLOBAL_PORT_R2354='32947', GLOBAL_RUN_R2354=self.plan['run'])
            self.q['target_submission'] = self.backend.start('target', environment, self.epoch)
            self.q['target_submitted'] = True; self.save()
            while self.clock() < self.epoch+GPU_SECONDS:
                if self.cancel_marker.exists(): raise RuntimeError('Original run cancellation requested')
                target = self.observation('target')
                if terminal_failure(target):
                    raise RuntimeError('Target terminal failure; no later arms or reload')
                if not self.q['draft_submitted'] and read_start(self.start_marker, expected):
                    if state_of(target) in TERMINAL: raise RuntimeError('Target ended before draft submission')
                    if self.epoch+NATIVE_SECONDS-self.clock() <= 180:
                        raise TimeoutError('Original Native budget exhausted before draft; no load')
                    self.backend.no_foreign()
                    self.q['draft_submission'] = self.backend.start('draft', environment, self.epoch)
                    self.q['draft_submitted'] = True; self.save()
                    listening = self.backend.draft_listening(self.epoch)
                    # No socket is opened here; node0 performs the sole authenticated connect.
                    publish_once(self.listening_marker,
                                 dict(expected, actual_draft_step=listening['actual_draft_step'], listener_ready=True))
                    self.q['passive_listener'] = listening
                draft = self.observation('draft') if self.q['draft_submitted'] else None
                if draft and terminal_failure(draft):
                    raise RuntimeError('Draft terminal failure; no reconnect or restart')
                if terminal_success(target):
                    if not draft: raise RuntimeError('Target completed without four-arm deferred draft')
                    if terminal_success(draft): break
                self.save(); self.sleep(5)
            else: raise TimeoutError('Original GPU bound exceeded')
            # A successful process exit alone does not imply quality or normal ownership.
            self.q['post'] = self.backend.idle('post')
            self.q['post_empty'] = True
            self.q['final_evidence'] = self.backend.collect_success()
            self.q.update(passed=True, completed=True, state='FOUR_REQUEST_NORMAL_CLEAN_PENDING_ROOT_COST_AUDIT')
        except BaseException as error:
            self.q.update(error_type=type(error).__name__, error=str(error), passed=False)
            if self.q['target_submitted'] or self.q['draft_submitted']:
                try: self.abort_owned(type(error).__name__)
                except BaseException as cleanup: self.q['cleanup_error_type'] = type(cleanup).__name__
                try:
                    self.q['post'] = self.backend.idle('failure-post')
                    self.q['post_empty'] = True
                except BaseException as cleanup: self.q['post_error_type'] = type(cleanup).__name__
            self.q.update(completed=True, state='FAILED_NO_RETRY_RELOAD')
        finally: self.save()
        return self.q


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args(); plan = json.loads(args.plan.read_bytes())
    verify_sources(ROOT, DIRECTORY, plan)
    assert os.environ['SLURM_JOB_ID'] == '910407' and socket.gethostname().split('.')[0] == CPU_NODE
    backend = SlurmBackend(ROOT, DIRECTORY, plan)
    controller = Controller(ROOT, plan, backend, os.environ['KIMI_LONG_SESSION_R347'])
    result = controller.run()
    print(json.dumps({k:result.get(k) for k in ['state', 'passed', 'actual_target', 'actual_draft']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
