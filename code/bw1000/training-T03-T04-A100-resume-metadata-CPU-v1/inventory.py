"""Original checkpoint metadata only. No framework imports or tensor-body reads."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pwd
import resource
import signal
import socket
import stat
import sys
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def identity():
    fields = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    return dict(host=socket.gethostname(), uid=os.getuid(), PID=os.getpid(),
                startticks=int(fields[19]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def stable(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def safe_path(path):
    require(path.is_absolute(), 'PATH_NOT_ABSOLUTE')
    for p in [path, *path.parents]:
        require(not p.is_symlink(), 'SYMLINK_PATH_REJECTED')
    require(path.stat().st_uid == os.getuid(), 'ORIGINAL_NOT_OWNED_UID')


def small(path, cap, budget):
    budget()
    safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'SMALL_JSON_SIZE_OR_TYPE')
        with os.fdopen(fd, 'rb', buffering=0, closefd=False) as stream:
            raw = stream.read(before.st_size + 1)
        after = os.fstat(fd)
        require(stable(before) == stable(after) == stable(path.stat()), 'SMALL_JSON_CHANGED')
        require(len(raw) == before.st_size, 'SMALL_JSON_SHORT_READ')
        value = json.loads(raw)
        require(isinstance(value, dict), 'SMALL_JSON_NOT_OBJECT')
        budget(len(raw))
        return value, dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
    finally:
        os.close(fd)


SCALARS = {'global_step', 'max_steps', 'gradient_accumulation_steps', 'per_device_train_batch_size',
           'world_size', 'seed', 'data_seed', 'epoch', 'save_steps', 'save_total_limit', 'bf16',
           'fp16', 'gradient_checkpointing', 'r', 'lora_alpha', 'lora_dropout', 'bias',
           'inference_mode', 'peft_type', 'task_type', 'is_local_process_zero', 'is_world_process_zero'}
INPUT_MARKERS = {'input_SHA256', 'input_fingerprint', 'input_fingerprints', 'microbatch_fingerprints',
                 'microbatch_SHA256', 'RNG', 'rng', 'rng_SHA256', 'data_cursor', 'sampler_epoch',
                 'gradient_accumulation_boundary', 'accumulation_boundary', 'optimizer_step'}


def scalar_summary(value):
    result = {}
    for k, v in value.items():
        if k in SCALARS and (v is None or type(v) in (int, bool) or type(v) is float and math.isfinite(v)):
            result[k] = v
        elif k in {'peft_type', 'task_type', 'bias'} and v in {'LORA', 'CAUSAL_LM', 'none', 'all', 'lora_only'}:
            result[k] = v
    return result


def reference_presence(native):
    """Report fields, never payload text or guessed input/RNG equality."""
    rows = []
    def walk(v, depth=0):
        require(depth <= 12, 'NATIVE_METADATA_DEPTH')
        if isinstance(v, dict):
            step = v.get('global_step', v.get('step'))
            if type(step) is int and 21 <= step <= 25:
                present = sorted(INPUT_MARKERS.intersection(v))
                rows.append(dict(global_step=step, present_reference_fields=present,
                                 loss_present=type(v.get('loss')) in (int, float)))
            for x in v.values():
                if isinstance(x, (dict, list)):
                    walk(x, depth + 1)
        elif isinstance(v, list):
            require(len(v) <= 4096, 'NATIVE_METADATA_LIST_LIMIT')
            for x in v:
                if isinstance(x, (dict, list)):
                    walk(x, depth + 1)
    walk(native)
    return rows[:160]


def declared_checkpoint_paths(native, train):
    found = set()
    def walk(v, depth=0, declared_step=None):
        require(depth <= 12, 'NATIVE_PATH_METADATA_DEPTH')
        if isinstance(v, dict):
            step = v.get('global_step', v.get('step'))
            for key, x in v.items():
                walk(x, depth + 1, step if key in {'path', 'checkpoint', 'checkpoint_path', 'checkpoint_dir'} else None)
        elif isinstance(v, list):
            require(len(v) <= 4096, 'NATIVE_PATH_METADATA_LIST_LIMIT')
            for x in v:
                if isinstance(x, (dict, list, str)):
                    walk(x, depth + 1, declared_step)
        elif isinstance(v, str) and len(v) <= 4096 and not any(c in v for c in ('\x00', '\n', '\r', '\\')):
            p = Path(v)
            if (p.is_absolute() and '..' not in p.parts and p.is_relative_to(train)
                    and p.name in {'checkpoint-20', 'checkpoint-25'}):
                if type(declared_step) is int and declared_step in (20, 25):
                    require(p.name == f'checkpoint-{declared_step}', 'DECLARED_CHECKPOINT_STEP_PATH_DIFFERS')
                found.add(p)
    walk(native)
    return found


def rank_coverage(row):
    coverage = []
    for step in range(21, 26):
        records = [r for r in row['step21_to25_reference_fields'] if r['global_step'] == step]
        coverage.append(dict(global_step=step, metadata_records=len(records),
                             loss_present=any(r['loss_present'] for r in records),
                             present_reference_fields=sorted({f for r in records for f in r['present_reference_fields']})))
    return dict(rank=row['rank'], steps=coverage)


def summary_cases(cases):
    rows = []
    for case in cases:
        checkpoints = []
        for cp in case.get('checkpoints', []):
            checkpoints.append(dict(path=cp.get('path'), expected_global_step=cp['expected_global_step'],
                                    actual_global_step=cp.get('actual_global_step'),
                                    state=cp.get('state', 'INVENTORIED'), step_matches=cp.get('step_matches', False),
                                    members={k: {x: v[x] for x in ('present', 'bytes') if x in v}
                                             for k, v in cp.get('members', {}).items()}))
        rows.append(dict(case=case['case'], state=case['state'], checkpoints=checkpoints,
                         unexpanded_output_directory_candidates=case.get('unexpanded_output_directory_candidates', []),
                         rank_step21_to25_reference_field_coverage=[
                             rank_coverage(r) for r in case['native_rank_metadata']], recovery_qualified=False))
    return rows


def checkpoint(path, expected_step, required, budget):
    safe_path(path)
    require(path.is_dir(), 'CHECKPOINT_NOT_DIRECTORY')
    budget()
    entries = list(itertools.islice(path.iterdir(), 65))
    require(len(entries) <= 64, 'CHECKPOINT_MEMBER_LIMIT')
    rows = {}
    for name in required:
        p = path / name
        if not p.exists() and not p.is_symlink():
            rows[name] = dict(present=False)
            continue
        safe_path(p)
        st = p.stat()
        require(stat.S_ISREG(st.st_mode), 'CHECKPOINT_MEMBER_NOT_REGULAR')
        rows[name] = dict(present=True, bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    docs = {}
    for name in ['trainer_state.json', 'args.json', 'adapter_config.json']:
        if rows[name]['present']:
            value, pin = small(path / name, 262144, budget)
            docs[name] = dict(**pin, scalars=scalar_summary(value))
    actual_step = docs.get('trainer_state.json', {}).get('scalars', {}).get('global_step')
    return dict(path=str(path), expected_global_step=expected_step, actual_global_step=actual_step,
                step_matches=actual_step == expected_step, members=rows, small_JSON=docs,
                all_14_present=all(r['present'] for r in rows.values()),
                tensor_body_bytes_read=0, tensor_SHA_verified=False,
                RNG_optimizer_scheduler_restoration_qualified=False)


def main(context, plan, claim):
    start = time.monotonic()
    epoch, deadline = context['epoch'], context['deadline']
    require(type(epoch) in (float, int) and deadline == epoch + 90 and epoch <= time.time() < deadline,
            'ORIGIN_EPOCH_INVALID_OR_EXPIRED')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('ORIGINAL_DEADLINE_EXHAUSTED')))
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    require(claim['root_GO'] is True and claim['CPU_GO'] is True and claim['GPU_GO'] is False, 'CLAIM_NOT_AUTHORIZED')
    require(pwd.getpwuid(os.getuid()).pw_name == 'whj', 'A100_ORIGINAL_UID_NAME_DIFFERS')
    require({2, 3}.issubset(os.sched_getaffinity(0)), 'CPU2_3_NOT_ALLOWED')
    os.sched_setaffinity(0, {2, 3})
    require(os.sched_getaffinity(0) == {2, 3}, 'CPU_AFFINITY_DIFFERS')
    owner = identity()
    run = Path(plan['new_namespace'])
    safe_path(run.parent)
    run.mkdir(mode=0o700, exist_ok=False)
    def save(name, value):
        with (run / name).open('xb') as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    intent = dict(**context, owner=owner, actual_affinity=[2, 3], resources=dict(CPU=2, GPU=0, AS_bytes=256 * 1024**2))
    save('intent.json', intent)
    print(json.dumps(dict(kind='READY', **intent)), flush=True)
    consumed = 0
    def budget(used=0):
        nonlocal consumed
        consumed += used
        require(time.time() < deadline and consumed <= 16 * 1024**2, 'DEADLINE_OR_JSON_BYTES_EXCEEDED')
    result = dict(**context, owner=owner,
                  cases=[dict(case=c['case'], state='NOT_STARTED', native_rank_metadata=[], checkpoints=[])
                         for c in plan['cases']], passed=False, GPU_GO=False, recovery_qualified=False)
    failure = None
    try:
        for case, case_result in zip(plan['cases'], result['cases']):
            root = Path(case['original_root']); train = root / 'train'
            safe_path(root); safe_path(train)
            case_result.update(state='READING_ORIGINAL_METADATA', original_root=str(root))
            _, source_pin = small(root / 'src/SOURCE_FILES.json', 262144, budget)
            require(source_pin['SHA256'] == case['original_source_manifest_SHA256'], 'ORIGINAL_SOURCE_MANIFEST_DIFFERS')
            case_result['original_source_manifest'] = source_pin
            native_rows, candidates = case_result['native_rank_metadata'], set()
            for rank in range(8):
                relative = 'train/nvidia-baseline/rank-%02d.json' % rank
                native, pin = small(root / relative, 524288, budget)
                native_rows.append(dict(rank=rank, path=str(root / relative), **pin,
                                        scalars=scalar_summary(native), step21_to25_reference_fields=reference_presence(native)))
                candidates.update(declared_checkpoint_paths(native, train))
            children = list(itertools.islice(train.iterdir(), 65)); require(len(children) <= 64, 'ORIGINAL_TRAIN_CHILD_LIMIT')
            output_dirs = []
            for p in children:
                if not p.is_symlink():
                    st = p.stat()
                    if stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid():
                        output_dirs.append(str(p))
                if p.name in {'checkpoint-20', 'checkpoint-25'}:
                    candidates.add(p)
            case_result['unexpanded_output_directory_candidates'] = sorted(output_dirs)
            checkpoints = case_result['checkpoints']
            for step in (20, 25):
                matching = sorted(p for p in candidates if p.name == f'checkpoint-{step}')
                require(len(matching) <= 1, 'AMBIGUOUS_ORIGINAL_CHECKPOINT')
                checkpoints.append(checkpoint(matching[0], step, plan['required_members'], budget)
                                   if matching else dict(expected_global_step=step, path=None, state='NOT_LOCATED_IN_ALLOWED_SCOPE'))
            case_result.update(state='INVENTORY_COMPLETED', train_single_level_entry_count=len(children),
                               recovery_qualified=False, GPU_GO=False)
        result['passed'] = all(c['state'] == 'INVENTORY_COMPLETED' for c in result['cases'])
    except BaseException as exc:
        failure = exc
        result.update(error_type=type(exc).__name__, error_reason=str(exc)[:180])
    finally:
        result.update(finished_epoch=time.time(), wall_seconds=time.monotonic() - start,
                      logical_JSON_bytes_read=consumed, tensor_body_bytes_read=0,
                      actual_affinity=sorted(os.sched_getaffinity(0)), owner_identity_end=identity())
        save('outcome.json', result)
        print(json.dumps(dict(kind='OUTCOME', passed=result['passed'], cases=len(result['cases']),
                              logical_JSON_bytes_read=consumed, GPU_GO=False,
                              summary_cases=summary_cases(result['cases']),
                              outcome_SHA256=hashlib.sha256(canonical(result)).hexdigest())), flush=True)
    if failure is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main(CONTEXT, PLAN, CLAIM)
