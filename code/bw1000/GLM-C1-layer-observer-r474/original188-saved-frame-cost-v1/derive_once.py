"""One CPU-only derivation from exact existing report and saved owner JSONs."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time

from costs import derive_owner, need, summarize

R = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
RUN_ID = '5f162445c8e04a399b29a8b47a555d0a'
OLD_REQUEST = '72ceb384e76441e5816ad51fbb1471ae'
OLD_PUB = 'c405c6af35a111d740af6ae121b614857b0aad5a'
REPORT = R / 'GLM474-original188-root-independent-terminal-v1.json'
REPORT_PIN = (22337, '3c884152bb0f86377ceb896b665218a650615a62ae7116a644b6ea04563c0baa')
OUT = R / 'GLM474-original188-saved-frame-cost-v1'
PER_FILE = 128 * 1024**2
TOTAL = 2 * 1024**3
H = lambda b: hashlib.sha256(b).hexdigest()


def identity(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    need(len(raw) <= 32*1024**2, 'DERIVED_OUTPUT_MAX_32MIB')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(bytes=len(raw), SHA256=H(raw))


def read_stable(path, expected, limit, remaining, charge):
    remaining()
    need(not path.is_symlink(), 'READ_SYMLINK')
    prior = path.stat()
    need(stat.S_ISREG(prior.st_mode) and prior.st_uid == os.getuid() and 0 < prior.st_size <= limit, 'READ_FILE_SCOPE')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb', buffering=0) as stream:
        before = os.fstat(stream.fileno())
        # One full explicit read, capped before allocation; no second payload hash pass.
        parts, done, hasher = [], 0, hashlib.sha256()
        while done < prior.st_size:
            remaining()
            piece = stream.read(min(1024**2, prior.st_size - done))
            need(piece, 'READ_TRUNCATED')
            parts.append(piece)
            hasher.update(piece)
            done += len(piece)
            charge(len(piece))
        extra = stream.read(1)
        charge(len(extra))
        need(not extra, 'READ_GREW')
        after = os.fstat(stream.fileno())
    need(identity(prior) == identity(before) == identity(after) == identity(path.stat()), 'READ_UNSTABLE')
    need(hasher.hexdigest() == expected, 'READ_SHA')
    raw = b''.join(parts)
    del parts
    remaining()
    return json.loads(raw), dict(bytes=len(raw), SHA256=hasher.hexdigest(), file_identity=identity(after))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epoch', type=float, required=True)
    p.add_argument('--request-id', required=True)
    p.add_argument('--publication', required=True)
    a = p.parse_args()
    import re
    need(re.fullmatch('[0-9a-f]{32}', a.request_id) and re.fullmatch('[0-9a-f]{40}', a.publication), 'NEW_CONTEXT')
    need(math.isfinite(a.epoch) and 0 <= time.time() - a.epoch < 150, 'ORIGIN_EPOCH')
    deadline = a.epoch + 180
    mono_deadline = time.monotonic() + deadline - time.time()

    def remaining(reserve=20):
        left = min(deadline - time.time(), mono_deadline - time.monotonic()) - reserve
        need(left > 0, 'ORIGINAL_DEADLINE')
        return left

    def hard_exit(_s, _f):
        os._exit(124)

    def stop(_s, _f):
        # Tail save/stdout remains bounded by the same original deadline.
        signal.signal(signal.SIGALRM, hard_exit)
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(deadline-time.time(), mono_deadline-time.monotonic())))
        raise TimeoutError('ORIGINAL_WORK_DEADLINE')

    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, remaining())
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    allowed = sorted(os.sched_getaffinity(0) - {0, 1})
    need(len(allowed) >= 2, 'TWO_ALLOWED_CPU_EXCLUDING_0_1')
    os.sched_setaffinity(0, allowed[:2])
    inherited = os.getpriority(os.PRIO_PROCESS, 0)
    requested = max(inherited, 10)
    os.nice(requested - inherited)
    actual = os.getpriority(os.PRIO_PROCESS, 0)
    need(requested <= actual <= 19, 'ACTUAL_NICE')
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    need(OUT.is_dir() and not OUT.is_symlink() and OUT.stat().st_uid == os.getuid(), 'OWNED_INSTALLED_SCOPE')
    src = Path(__file__).resolve().parent
    from verify_package import verify
    verify(src, adopted=True)
    ticks = int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    context = dict(request_id=a.request_id, publication=a.publication, epoch=a.epoch, deadline=deadline,
                   host=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   PID=os.getpid(), startticks=ticks, uid=os.getuid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                   inherited_nice=inherited, requested_nice=requested, actual_nice=actual,
                   RLIMIT_AS_bytes=4*1024**3, original_run=RUN_ID, original_request=OLD_REQUEST)
    save(OUT/'intent.json', context)
    start = time.monotonic()
    result = dict(context=context, planned_owners=16, planned_rank_frames=48, planned_AR_steps=3,
                  actual_AR_steps=None, independence='NOT_ESTABLISHED_CONTIGUOUS_DEPENDENT_AR_STEPS',
                  owner_results=[], errors={}, input_pins={}, explicit_read_bytes=0,
                  missing_owners=list(range(16)), GPU_GO=False, Model_qualified=False,
                  performance_qualified=False, exclusive_GPU_time=False, current_GPU_busy=None)
    def charge(size):
        result['explicit_read_bytes'] += size
        need(result['explicit_read_bytes'] <= TOTAL, 'TOTAL_EXPLICIT_IO_CAP')
    try:
        report, pin = read_stable(REPORT, REPORT_PIN[1], REPORT_PIN[0], remaining, charge)
        need(pin['bytes'] == REPORT_PIN[0] and report['run_id'] == RUN_ID
             and report['original_request_id'] == OLD_REQUEST and report['claim_commit'] == OLD_PUB
             and report['independent_post_clean'] is True and report['valid_attention_detail'] is True,
             'ORIGINAL_REPORT_IDENTITY')
        pins = report['raw_artifact_SHA256']
        need(len(pins) == 31, 'ORIGINAL_31_PINS')
        result['input_pins'][str(REPORT)] = pin
        for rank in range(16):
            try:
                relative = 'GLM474-history-value-model-%s-socket/owner-%02d.json' % (RUN_ID, rank)
                need(relative in pins and re.fullmatch('[0-9a-f]{64}', pins[relative]), 'EXACT_OWNER_PIN')
                path = R/relative
                # Pre-stat is not a read; enforce whole-job limit before read_stable.
                remaining()
                need(result['explicit_read_bytes'] + path.stat().st_size <= TOTAL, 'TOTAL_EXPLICIT_IO_CAP')
                owner, owner_pin = read_stable(path, pins[relative], PER_FILE, remaining, charge)
                result['input_pins'][str(path)] = owner_pin
                derived = derive_owner(owner, rank)
                result['owner_results'].append(derived)
                result['missing_owners'].remove(rank)
                del owner
                gc.collect()
            except (ValueError, KeyError, TypeError, OSError, MemoryError) as exc:
                result['errors'][str(rank)] = dict(type=type(exc).__name__, reason=str(exc)[:160])
            remaining()
        # Identity equality is intra-step across partitions, not cross-host clock subtraction.
        reference = result['owner_results'][0]['capture_identities'] if result['owner_results'] else None
        for owner in result['owner_results']:
            need(owner['capture_identities'] == reference, 'PARTITION_CAPTURE_TOKEN_IDENTITY')
        if len(result['owner_results']) == 16 and not result['errors']:
            result['actual_AR_steps'] = len(reference)
        result['cost_groups'] = summarize(result['owner_results'])
    except (ValueError, KeyError, TypeError, OSError, TimeoutError, MemoryError) as exc:
        result['errors']['derivation'] = dict(type=type(exc).__name__, reason=str(exc)[:160])
    # Keep a hard deadline while final files and stdout are persisted.
    signal.signal(signal.SIGALRM, hard_exit)
    signal.setitimer(signal.ITIMER_REAL, max(.001, min(deadline-time.time(), mono_deadline-time.monotonic())))
    frames = sum(len(o['frames']) for o in result['owner_results'])
    complete = not result['errors'] and not result['missing_owners'] and frames == 48
    result.update(observed_rank_frames=frames, state='DERIVED_SAVED48_ENVELOPES' if complete else 'UNKNOWN_INCOMPLETE_SAVED_COST',
                  derived_complete=complete, wall_s=time.monotonic()-start,
                  limitations=['48 rank-frames are 3 contiguous dependent AR steps partitioned over16 ranks; independence is not established',
                               'Nested host/event envelopes overlap; no exclusiveGPU/FLOPs/busy or cross-host makespan',
                               'NoV exposure derives from original branch+tuple markers, not a direct call counter',
                               'First-to-last marker span is not necessarily complete AR wall time',
                               'No Model/Native/SDK/provider/repeated numerical qualification; saved events only'])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result['system_cost'] = dict(user_s=usage.ru_utime, sys_s=usage.ru_stime, maxrss_bytes=usage.ru_maxrss*1024,
                                 physical_IO_bytes='UNKNOWN_OS_READAHEAD_AND_CACHE')
    private_pin = save(OUT/'cost-private.json', result)
    # Public candidate: duration ranges, no raw markers/capture tokens/owner rows.
    summary = {k: result[k] for k in ('state','derived_complete','planned_owners','planned_rank_frames',
               'observed_rank_frames','missing_owners','errors','explicit_read_bytes','system_cost','limitations')}
    summary.update(private_report=private_pin, units='seconds', planned_AR_steps=3,
                   actual_AR_steps=result['actual_AR_steps'], independence=result['independence'],
                   source_derived_exposure_layer_frames=sum(e['source_derived_exposure'] for o in result['owner_results']
                       for f in o['frames'] for e in f['source_exposure']),
                   cost_group_count=len(result.get('cost_groups', [])), performance_qualified=False, exclusive_GPU_time=False)
    # Selected detail and first/last bounds fit below32KiB; all78layer ranges remain private.
    selected = [r for r in result.get('cost_groups', []) if r['name'] == 'first_to_last_observed_marker'
                or r['layer'] in (0, 3)]
    summary['cost_groups'] = selected
    raw = json.dumps(summary, allow_nan=False).encode()
    if len(raw) > 30*1024:
        summary['cost_groups'] = [r for r in selected if r['name'] in ('decoder_layer','FFN.enclosing','first_to_last_observed_marker')
                                 or r['name'] in ('attention.history.selected_value','attention.output.forward','indexer.features','indexer.scored_selector')]
    need(len(json.dumps(summary).encode()) <= 32768, 'BOUNDED_SUMMARY')
    summary_pin = save(OUT/'summary.json', summary)
    save(OUT/'outcome.json', dict(context=context, derived_complete=complete, state=result['state'],
                                 private_report=private_pin, summary=summary_pin, finished=time.time()))
    # Same invocation returns the bounded cost summary, never the private owner/frame JSON.
    print(json.dumps(dict(summary=summary, summary_file_pin=summary_pin, GPU_GO=False),
                     separators=(',', ':'), allow_nan=False), flush=True)
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
