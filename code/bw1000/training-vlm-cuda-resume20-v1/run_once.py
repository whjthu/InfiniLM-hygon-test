"""New bounded A100 recovery supervisor; common framework owns all training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from deadline_contract import schedule
from owned_processes import snapshot, refresh, cleanup
from held_child import cleanup_held
from prepare_launch import D, prepare, require, sha


def save(root, name, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    require(len(raw) <= 2 * 1024**2, 'bounded supervisor record exceeded')
    with (root / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['T03.sft', 'T04.sft'])
    parser.add_argument('--framework', required=True)
    parser.add_argument('--canonical-reference', required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--publication', required=True)
    parser.add_argument('--epoch', required=True, type=float)
    args = parser.parse_args()
    clock = schedule(args.epoch, time.time())
    claim = json.loads((D / 'CLAIM.json').read_bytes())
    require(claim['root_GO'] is True and claim['GPU_GO'] is True, 'root source review/publication required')
    root = Path(claim['cases'][args.case]['output_root'])
    require(root.is_absolute() and not root.is_symlink() and not root.exists(),
            'new recovery namespace exists: reconcile original request, never replay')
    require(time.time() < clock['native_deadline'] - 60, 'original recovery epoch has no admission reserve')
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    registry = {}; child = None; anchor = None; failure = None; timed_out = False; post = None; planned = None
    def expired(*_):
        raise TimeoutError('original recovery supervisor epoch exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, args.epoch + 890 - time.time()))
    cleaned = dict(owned_processes_remaining=None, termination_needed=False)
    try:
        planned = prepare(args.case, framework=args.framework, scope=root,
                          canonical_reference=args.canonical_reference, request_id=args.request_id,
                          epoch=args.epoch, native_deadline=clock['native_deadline'], publication=args.publication)
        save(root, 'intent.json', {k: v for k, v in planned.items() if k != 'environment'})
        sys.path.insert(0, str(Path(args.framework)))
        from vlm.runtime.cuda_resume_contract import bounded_bytes, fresh_gpu_admission, verify_file, remaining
        config = json.loads(Path(planned['config_path']).read_bytes())
        # Reuse historically whole-hashed shards only while their exact recorded stat identity remains stable.
        model = json.loads((D / 'cases' / (args.case + '-model-files.json')).read_bytes())
        source_root = Path(model['model_root']); verified = []
        for name, pin in sorted(model['files'].items()):
            remaining(config); path = source_root / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == pin['bytes'], 'original model member differs')
            if name.endswith('.safetensors'):
                current = path.stat()
                require(dict(bytes=current.st_size, mtime_ns=current.st_mtime_ns, inode=current.st_ino, device=current.st_dev)
                        == model['weight_stats'][name], 'whole-qualified original shard stat identity differs')
            else:
                raw = bounded_bytes(path, 16 * 1024**2)
                require(sha(raw) == pin['SHA256'], 'model metadata identity differs')
            verified.append(name)
        inputs = json.loads((D / 'cases' / (args.case + '-input-files.json')).read_bytes())
        for pin in inputs['files']:
            verify_file(Path(pin['source']), pin, lambda: remaining(config))
        save(root, 'assets-admission.json', dict(model_members=verified, original_whole_shard_SHA_reuse=True,
             input_files=len(inputs['files']), model_tensor_payload_read_before_native=False, formal_performance=False))
        port = 29911 if args.case == 'T03.sft' else 29913
        save(root, 'before.json', fresh_gpu_admission(config, port))
        with (root / 'native.stdout').open('xb') as out, (root / 'native.stderr').open('xb') as err:
            child = subprocess.Popen(planned['command'], env=planned['environment'], stdout=out, stderr=err,
                                     start_new_session=True)
            anchor = dict(PID=child.pid, PGID=child.pid, start_new_session=True,
                          supervisor=planned['supervisor'], request_id=args.request_id)
            # Persist the strong held-Popen/new-session root before any /proc
            # snapshot. Even a missing snapshot still enters held cleanup.
            save(root, 'held-child-anchor.json', anchor)
            row = snapshot().get(child.pid)
            require(row is not None, 'new common-entry child identity unavailable; preserve unknown')
            registry[child.pid] = {k: row[k] for k in ('PID', 'PPID', 'PGID', 'startticks', 'state')}
            save(root, 'child-owner.json', dict(supervisor=planned['supervisor'], child=registry[child.pid],
                                                request_id=args.request_id, epoch=args.epoch))
            while child.poll() is None:
                refresh(registry, root)
                temp = root / 'owned-registry.tmp'; temp.write_text(json.dumps(registry)); temp.replace(root / 'owned-registry.json')
                left = clock['supervisor_stop'] - time.time()
                if left <= 0:
                    timed_out = True; break
                try:
                    child.wait(timeout=min(1.0, left))
                except subprocess.TimeoutExpired:
                    pass
    except BaseException as error:
        failure = dict(type=type(error).__name__, reason=str(error)[:240])
    finally:
        # The registry starts with this new common-entry child, then follows only its descendants.
        held = cleanup_held(child, anchor, args.epoch + 880)
        cleaned = cleanup(registry, root, grace=max(0, min(10, (args.epoch + 880 - time.time()) / 2)))
        cleaned['held_child'] = held
        cleaned['termination_needed'] = cleaned['termination_needed'] or held['termination_needed']
        if not held['reaped']:
            failure = failure or dict(type='OWNED_ROOT_REMAINS', reason='held child not reaped after bounded TERM/KILL')
        if planned is not None:
            try:
                config['deadline'] = clock['deadline']  # Same origin; post reserve only, never extends the scope.
                post = fresh_gpu_admission(config, 29911 if args.case == 'T03.sft' else 29913)
                save(root, 'after.json', post)
            except BaseException as error:
                post = dict(state='POST_UNKNOWN', error_type=type(error).__name__)
        rank_receipts = {}
        for rank in range(8):
            path = root / 'resume-witness' / ('rank-%02d' % rank) / 'receipt.json'
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * 1024**2:
                raw = path.read_bytes(); row = json.loads(raw)
                rank_receipts[str(rank)] = dict(bytes=len(raw), SHA256=sha(raw),
                    microbatches=row.get('microbatches'), updates=row.get('updates'),
                    own_restore_engineering_verified=row.get('own_restore_engineering_verified'),
                    original25_typed_state_exact=row.get('original25_typed_state_exact'))
        passed = bool(child is not None and child.returncode == 0 and failure is None and not timed_out
                      and cleaned['owned_processes_remaining'] == 0 and not cleaned['termination_needed']
                      and post and post.get('active_compute_processes') == 0 and len(rank_receipts) == 8
                      and all(row['own_restore_engineering_verified'] is True for row in rank_receipts.values()))
        result = dict(case=args.case, request_id=args.request_id, **clock, finished_epoch=time.time(),
                      source_publication=args.publication, native_returncode=child.returncode if child else None,
                      actual_GPU_launched=child is not None, timed_out=timed_out, failure=failure, cleanup=cleaned,
                      planned_ranks=8, planned_microbatches=160, planned_optimizer_updates_per_rank=5,
                      rank_receipts=rank_receipts, post=post, engineering_recovery_completed=passed,
                      original25_all_rank_exact=(len(rank_receipts) == 8 and all(r['original25_typed_state_exact'] is True for r in rank_receipts.values())),
                      original_tail_per_micro_qualified=False, formal_numerics=False, formal_performance=False)
        save(root, 'outcome.json', result)
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(case=args.case, request_id=args.request_id, passed=passed,
                         original25_all_rank_exact=result['original25_all_rank_exact'], GPU_launched=child is not None)))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
