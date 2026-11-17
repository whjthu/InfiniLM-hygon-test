"""Read a fixed list of A100 Python sources; AST only, never import providers."""
import ast
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import pwd
import resource
import signal
import socket
import stat
import time


def require(ok, why):
    if not ok:
        raise ValueError(why)


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def identity():
    return dict(host=socket.gethostname(), uid=os.getuid(), PID=os.getpid(),
                startticks=int(Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def stable(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def resolve_source_path(row, record):
    """One declared logical file; resolve never authorizes an arbitrary new root."""
    logical = Path(row['path']); suffix = Path(row['package_relative_path'])
    require(logical.is_absolute() and '..' not in logical.parts
            and not suffix.is_absolute() and '..' not in suffix.parts, 'SOURCE_DECLARED_PATH')
    canonical = logical.resolve(strict=True)
    record['canonical_path_before'] = str(canonical)
    matches = [Path(root) for root in row['allowed_canonical_roots'] if canonical == Path(root) / suffix]
    require(len(matches) == 1, 'SOURCE_CANONICAL_SUFFIX_OR_ROOT')
    root = matches[0]
    resolved_root = root.resolve(strict=True)
    record['canonical_root_before'] = str(resolved_root)
    require(root.is_absolute() and resolved_root == root, 'REGISTERED_ROOT_HAS_UNKNOWN_ALIAS')
    return logical, canonical, root


def finish_source_path(logical, canonical, root, record):
    after = logical.resolve(strict=True)
    after_root = root.resolve(strict=True)
    record.update(canonical_path_after=str(after), canonical_root_after=str(after_root))
    require(after == canonical and after_root == root, 'SOURCE_CANONICAL_RESOLUTION_CHANGED')


def validate_registered_paths(files):
    env = '/home/whj/nvidia-baseline/env/lib/python3.12/site-packages'
    system = '/usr/local/lib/python3.12/dist-packages'
    logical_env = '/home/whj/nvidia-baseline/env-vlm/lib/python3.12/site-packages'
    require(len(files) == 15 and len({r['name'] for r in files}) == 15, 'FIXED_SOURCE_DENOMINATOR')
    for row in files:
        suffix = row['package_relative_path']; roots = row['allowed_canonical_roots']
        if row['name'].startswith('site-packages/'):
            require(row['name'] == 'site-packages/' + suffix and suffix.endswith('.py'), 'SOURCE_MEMBER_SUFFIX')
            package = suffix.split('/')[0]
            if package == 'swift':
                require(roots == [env] and row['path'] == env + '/' + suffix, 'SWIFT_REGISTERED_CANONICAL_PATH')
            elif package == 'torch':
                require(roots == [system] and row['path'] == system + '/' + suffix, 'TORCH_REGISTERED_CANONICAL_PATH')
            else:
                require(package in {'transformers', 'accelerate'} and set(roots) == {env, system}
                        and len(roots) == 2 and row['path'] == logical_env + '/' + suffix,
                        'DECLARED_LOGICAL_PACKAGE_PATH')
        else:
            require(row['name'] == 'original-callback/swift_callback_nvidia_baseline_v2.py'
                    and suffix == 'swift_callback_nvidia_baseline_v2.py'
                    and roots == ['/home/whj/nvidia-baseline']
                    and row['path'] == '/home/whj/nvidia-baseline/' + suffix, 'ORIGINAL_CALLBACK_PATH')


def read_source(row, deadline, remaining, consume):
    require(time.time() < deadline, 'ORIGINAL_EPOCH_EXPIRED')
    path, canonical, root = resolve_source_path(row, row['_record'])
    fd = os.open(canonical, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.getuid())
                and not before.st_mode & stat.S_IWOTH and 0 < before.st_size <= min(1024**2, remaining),
                'SOURCE_TYPE_OWNER_SIZE')
        with os.fdopen(fd, 'rb', buffering=0, closefd=False) as stream:
            raw = stream.read(before.st_size + 1)
        consume(len(raw))
        finish_source_path(path, canonical, root, row['_record'])
        require(len(raw) == before.st_size and stable(before) == stable(os.fstat(fd))
                == stable(canonical.stat()) == stable(path.stat()),
                'SOURCE_STABLE_FD')
    finally:
        os.close(fd)
    digest = hashlib.sha256(raw).hexdigest()
    if row.get('original_SHA256') is not None:
        require(digest == row['original_SHA256'] and len(raw) == row['original_bytes'], 'ORIGINAL_SWIFT_PIN_CHANGED')
    return raw, dict(bytes=len(raw), SHA256=digest, owner_UID=before.st_uid, mtime_ns=before.st_mtime_ns)


def relevant_definitions(raw):
    text = raw.decode('utf-8-sig'); tree = ast.parse(text); lines = text.splitlines(keepends=True)
    result = []
    selected = {'__iter__', '__len__', 'set_epoch', 'get_train_dataloader', '_get_train_sampler',
                '_inner_training_loop', '_load_rng_state', '_save_rng_state', '_load_from_checkpoint',
                '_load_optimizer_and_scheduler', '_save_checkpoint', '_maybe_log_save_evaluate',
                'training_step', 'prepare_data_loader', 'skip_first_batches', 'load_accelerator_state',
                'save_accelerator_state', 'synchronize_rng_states', 'synchronize_rng_state',
                'on_save', 'on_step_begin', 'on_step_end', 'on_train_begin', 'on_train_end'}
    def walk(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in selected:
                    require(len(prefix + node.name) <= 512, 'AST_QUALNAME_LIMIT')
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    source = ''.join(lines[start - 1:node.end_lineno]).encode('utf-8')
                    result.append(dict(qualname=prefix + node.name, start_line=start, end_line=node.end_lineno,
                                       definition_SHA256=hashlib.sha256(source).hexdigest()))
                # Nested functions remain in the captured complete file, not separately executed.
    walk(tree.body)
    require(len(result) <= 512, 'AST_DEFINITION_LIMIT')
    return result


def main(context, plan, claim):
    start = time.monotonic(); epoch, deadline = context['epoch'], context['deadline']
    require(deadline == epoch + 90 and epoch <= time.time() < deadline, 'ORIGIN_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('ORIGINAL_EPOCH_EXPIRED')))
    signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    require(claim['root_GO'] is True and claim['CPU_GO'] is True and claim['GPU_GO'] is False, 'CLAIM_GO_FALSE')
    validate_registered_paths(plan['files'])
    require(pwd.getpwuid(os.getuid()).pw_name == 'whj' and {2, 3}.issubset(os.sched_getaffinity(0)), 'ORIGINAL_ACCOUNT_CPUS')
    os.sched_setaffinity(0, {2, 3})
    who = identity(); run = Path(plan['new_namespace'])
    require(not any(p.is_symlink() for p in [run.parent, *run.parent.parents])
            and run.parent.stat().st_uid == os.getuid(), 'NEW_NAMESPACE_PARENT')
    run.mkdir(mode=0o700, exist_ok=False)
    def save(name, raw):
        with (run / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    save('intent.json', encode(dict(**context, owner=who, source_file_denominator=len(plan['files']), actual_affinity=[2, 3])))
    print(json.dumps(dict(kind='READY', **context, owner=who)), flush=True)
    records = [dict(name=r['name'], path=r['path'], state='NOT_READ') for r in plan['files']]
    result = dict(**context, owner=who, files=records, passed=False, GPU_GO=False,
                  provider_imported=False, Torch_imported=False, model_or_checkpoint_loaded=False)
    bodies = {}; read_bytes = 0; failure = None; packed = None
    def persist_bundle():
        nonlocal packed
        bundle = encode(dict(files=bodies, request_id=context['request_id']))
        require(len(bundle) <= 8 * 1024**2 and time.time() < deadline, 'BUNDLE_OR_EPOCH_LIMIT')
        packed = gzip.compress(bundle, compresslevel=6, mtime=0)
        require(len(packed) <= 2 * 1024**2 and time.time() < deadline, 'COMPRESSED_BUNDLE_OR_EPOCH_LIMIT')
        save('provider-source-bundle.json.gz', packed)
        result.update(compressed_bundle_bytes=len(packed), compressed_bundle_SHA256=hashlib.sha256(packed).hexdigest(),
                      uncompressed_bundle_bytes=len(bundle), uncompressed_bundle_SHA256=hashlib.sha256(bundle).hexdigest(),
                      bundled_source_files=len(bodies))
    try:
        for source, record in zip(plan['files'], records):
            record['state'] = 'READING_DECLARED_SOURCE'
            def consume(size):
                nonlocal read_bytes
                read_bytes += size
                record['returned_read_bytes'] = size
                require(read_bytes <= 4 * 1024**2 and time.time() < deadline, 'SOURCE_READ_BUDGET')
            request = dict(source, _record=record)
            raw, pin = read_source(request, deadline, 4 * 1024**2 - read_bytes, consume)
            definitions = relevant_definitions(raw)
            candidate = dict(record, state='SOURCE_READ_AND_AST_PARSED', **pin, definitions=definitions,
                             original_byte_pin_reused=source.get('original_SHA256') is not None)
            require(len(encode(candidate)) <= 32768, 'PER_SOURCE_METADATA_LIMIT')
            record.update(candidate)
            bodies[source['name']] = dict(**pin, base64=base64.b64encode(raw).decode('ascii'))
        persist_bundle()
        result['passed'] = True
    except BaseException as exc:
        failure = exc; result.update(error_type=type(exc).__name__, error_reason=str(exc)[:180])
    finally:
        if packed is None and bodies and time.time() < deadline - 1:
            try:
                persist_bundle()
            except BaseException as exc:
                result['partial_bundle_error_type'] = type(exc).__name__
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(finished_epoch=time.time(), wall_seconds=time.monotonic() - start, logical_source_bytes_read=read_bytes,
                      actual_affinity=sorted(os.sched_getaffinity(0)), owner_end=identity(),
                      self_user_seconds=usage.ru_utime, self_system_seconds=usage.ru_stime,
                      self_maxrss_bytes=usage.ru_maxrss * 1024,
                      historical_identity_of_unpinned_base_sources_qualified=False,
                      sampler_runtime_or_loader_length_qualified=False, recovery_qualified=False)
        save('outcome.json', encode(result))
        response = dict(kind='OUTCOME', outcome=result, outcome_SHA256=hashlib.sha256(encode(result)).hexdigest())
        if packed is not None and 'compressed_bundle_SHA256' in result:
            response['compressed_bundle_base64'] = base64.b64encode(packed).decode('ascii')
        print(json.dumps(response, separators=(',', ':')), flush=True)
    if failure is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main(CONTEXT, PLAN, CLAIM)
