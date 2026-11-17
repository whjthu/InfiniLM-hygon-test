"""Future metadata-only audit; S is injected from exact original context, never guessed."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import time
P=S['producer_plan']
ROOT=Path(P['namespace'])
READ=0
EVIDENCE={}

def require(ok, why):
    if not ok:
        raise ValueError(why)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def stamp(st):
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]

def unique(raw):

    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs)

def take(relative, limit=131072, pin=None):
    global READ
    path = ROOT / relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'relative path')
    require(not any((p.is_symlink() for p in (path, *path.parents))), 'linked metadata/source')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        before = os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, 'metadata bound')
        require(READ + before.st_size <= S['read_limit_bytes'], 'total read bound')
        raw = f.read(limit + 1)
        READ += len(raw)
        require(len(raw) == before.st_size and stamp(os.fstat(f.fileno())) == stamp(before) == stamp(path.lstat()), 'stable metadata FD')
    identity = dict(bytes=len(raw), SHA256=sha(raw))
    EVIDENCE[relative] = identity
    if pin:
        require(identity == pin, 'original pin mismatch')
    return raw

def obj(relative, limit=131072, pin=None):
    return unique(take(relative, limit, pin))

def digest(v):
    return type(v) is str and re.fullmatch('[0-9a-f]{64}', v) is not None

def context(v):
    for key in ('publication', 'epoch', 'deadline', 'source_manifest_SHA256'):
        expected = S['source_manifest']['SHA256'] if key == 'source_manifest_SHA256' else S[key]
        require(v[key] == expected, 'original context ' + key)
    require(v.get('request_id') == S['request_id'], 'original request')
    require(v['deadline'] - v['epoch'] == 900, 'original epoch900')

def owner_shape(v):
    require(type(v) is dict and set(v) == {'PID', 'PPID', 'PGID', 'startticks', 'UID', 'host', 'boot_id'}, 'owner schema')
    require(all((type(x) is int and x >= 0 for x in (v[k] for k in ('PID','PPID','PGID','startticks','UID')))) and v['PID'] > 0 and (v['startticks'] > 0), 'owner identity')

def original_owner_absent(v):
    owner_shape(v)
    require(v['host'] == socket.gethostname() and v['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'ORIGINAL_PROCESS_HOST_OR_BOOT_UNKNOWN')
    path = Path('/proc') / str(v['PID']) / 'stat'
    try:
        with path.open('rb') as stream:
            raw = stream.read(4097)
    except FileNotFoundError:
        return dict(PID=v['PID'], original_startticks=v['startticks'], original_absent=True, state='PID_ABSENT')
    require(len(raw) <= 4096, 'known PID stat bound')
    parts = raw.decode().rsplit(') ', 1)[1].split()
    ticks = int(parts[19])
    require(ticks != v['startticks'], 'original owner still exists')
    return dict(PID=v['PID'], original_startticks=v['startticks'], observed_startticks=ticks, original_absent=True, state='PID_REUSED_ORIGINAL_ABSENT')

def main():
    started = time.time()

    def expired(*unused):
        raise TimeoutError('reader60s')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, S['reader_seconds'])
    source = obj('source/SOURCE_FILES.json', 131072, S['source_manifest'])
    require(source['files'] == S['source_files'], 'source inventory')
    for name, pin in S['source_files'].items():
        take('source/' + name, 256 * 1024, pin)
    install = obj('install-intent.json')
    intent = obj('output/intent.json')
    outcome = obj('output/outcome.json', 16384, S['original_outcome'])
    worker_owner = obj('output/worker-owner.json', 4096)
    worker = obj('output/worker-result.json', 16384)
    resolved = obj('output/resolved-input-pins.json')
    doc = obj('output/input-fingerprints.json', 131072, worker['output'])
    for v in (install, intent, outcome):
        context(v)
    require(install['owner'] == intent['owner'], 'supervisor original identity')
    require(install['source_files'] == len(S['source_files']) and install['source_bytes'] == sum((v['bytes'] for v in S['source_files'].values())) + S['source_manifest']['bytes'], 'installed source denominator')
    require(intent['work_deadline'] == S['epoch'] + 780 and intent['files'] == 64 and (intent['max_workers'] == 1), 'work scope')
    require(len(intent['CPU_affinity']) == 2 and len(set(intent['CPU_affinity'])) == 2 and not set(intent['CPU_affinity']) & {0,1}, 'CPU scope')
    cpu_ready=obj('output/worker-cpu-ready.json',4096)
    require(cpu_ready['CPU_affinity']==intent['CPU_affinity'] and cpu_ready['excluded_CPU_cores']==[0,1]
            and cpu_ready['epoch']==S['epoch'] and cpu_ready['deadline']==S['deadline']
            and cpu_ready['before_Torch_and_payload'] is True,'actual worker nonstream CPU_READY')
    require(intent['worker_RSS_sampled_limit_bytes'] == 8 * 1024 ** 3 and intent['hard_memory_cgroup_enforced'] is False and (intent['RLIMIT_AS_enforced'] is False), 'memory terminology')
    require(outcome['passed'] is True and outcome['worker_returncode'] == 0 and (outcome['own_worker_reaped'] is True) and (outcome['failure_type'] is None) and (outcome['cleanup_error_type'] is None), 'terminal success')
    require(outcome['completed_files'] == outcome['file_denominator'] == 64 and outcome['worker_result'] == worker and (outcome['worker'] == worker_owner), 'worker chain')
    owner_shape(worker_owner)
    owner_shape(intent['owner'])
    require(worker_owner['PPID'] == intent['owner']['PID'] and worker_owner['PGID'] == worker_owner['PID'], 'owned worker parent/session')
    require(outcome['observed_worker_peak_RSS_bytes'] == S['known_observed_peak_RSS_bytes'] and outcome['RSS_sampling_seconds'] == 0.2, 'original observed RSS')
    require(outcome['device_compute_calls'] == 0 and outcome['driver_query_calls'] == 1 and (outcome['target_GO'] is False) and (outcome['other136_payloads_read'] == 0) and (outcome['RNG_equivalence'] is False), 'producer boundaries')
    require(worker['passed'] is True and worker['completed_files'] == worker['file_denominator'] == 64 and (worker['input_payload_only'] is True) and (worker['other136_payloads_read'] == 0) and (worker['models'] == 0) and (worker['device_compute_calls'] == worker['driver_query_calls'] == 0), 'worker success boundaries')
    required = {'schema', 'case', 'reference_request_id', 'original_binding_SHA256', 'original_value_audit_SHA256', 'fingerprint_source_SHA256', 'projection', 'world_size', 'accumulation', 'source_receipts', 'rows'}
    require(type(doc) is dict and set(doc) == required, 'document schema')
    require(doc['schema'] == 'native_first_two_inputs_v1' and doc['case'] == 'T04.sft' and (doc['projection'] == ['args', 'kwargs']) and (doc['world_size'] == 8) and (doc['accumulation'] == 4), 'input protocol')
    for key in ('reference_request_id', 'original_binding_SHA256', 'original_value_audit_SHA256', 'fingerprint_source_SHA256', 'source_receipts'):
        require(doc[key] == P[key], 'document provenance ' + key)
    require(resolved['case'] == 'T04.sft' and resolved['source_receipts'] == P['source_receipts'] and (resolved['all_pins_resolved_before_Torch'] is True), 'resolved receipt provenance')
    require(set(resolved['receipt_stamps']) == set((str(n) for n in range(8))), 'eight receipt stamps')
    for rank, identity in resolved['receipt_stamps'].items():
        require(type(identity) is list and len(identity) == 5 and all((type(x) is int for x in identity)) and (identity[2] == P['source_receipts'][rank]['bytes']), 'typed original receipt stamps')
    require(type(doc['rows']) is list and len(doc['rows']) == 64 and (type(resolved['rows']) is list) and (len(resolved['rows']) == 64), 'full64 denominator')
    total = 0
    for index, (row, original) in enumerate(zip(doc['rows'], resolved['rows'])):
        rank, micro = divmod(index, 8)
        micro += 1
        require(type(row) is dict and set(row) == {'rank', 'micro_ordinal', 'global_step', 'path', 'bytes', 'file_SHA256', 'input_fingerprint_SHA256'}, 'row schema')
        require(all((type(row[k]) is int for k in ('rank', 'micro_ordinal', 'global_step', 'bytes'))), 'typed row integers')
        require((row['rank'], row['micro_ordinal'], row['global_step']) == (rank, micro, (micro - 1) // 4), 'rank/micro/step ordered64')
        require(row['path'] == P['reference_root'] + '/capture/rank-%02d/forward-%02d-input.pt' % (rank, micro), 'exact original input path')
        require(0 < row['bytes'] <= P['per_input_file_limit_bytes'] and digest(row['file_SHA256']) and digest(row['input_fingerprint_SHA256']), 'typed SHA/file bounds')
        require(type(original) is dict and set(original) == set(row) - {'input_fingerprint_SHA256'} | {'identity'}, 'resolved row schema')
        require({k: v for k, v in row.items() if k != 'input_fingerprint_SHA256'} == {k: v for k, v in original.items() if k != 'identity'}, 'fullfile pin cross-resolved')
        require(type(original['identity']) is list and len(original['identity']) == 5 and all((type(x) is int for x in original['identity'])) and (original['identity'][2] == row['bytes']), 'resolved stable identity')
        total += row['bytes']
    require(total == resolved['total_input_bytes'] == worker['input_bytes'] and total <= P['all_input_file_limit_bytes'], 'total bytes')
    posts = [original_owner_absent(intent['owner']), original_owner_absent(worker_owner)]
    result = dict(passed=True, request_id=S['request_id'], publication=S['publication'], original_context=dict(epoch=S['epoch'], deadline=S['deadline']), original_outcome=S['original_outcome'], source_manifest=S['source_manifest'], metadata_file_pins=EVIDENCE, metadata_bytes_read=READ, rank_denominator=8, file_denominator=64, input_bytes=total, input_reference_pin=worker['output'], input_reference=doc, known_owner_posts=posts, observed_producer_peak_RSS_bytes=S['known_observed_peak_RSS_bytes'], reader_elapsed_seconds=time.time() - started, independent_metadata_audit=True, producer_canonical_source_qualified=True, independent_math_digest_recomputed=False, target_must_recompute_actual_inputs=True, target_GO=False, GPU_GO=False, reader_Torch_imports=0, reader_device_compute_calls=0, reader_driver_query_calls=0, original64_payloads_read=0, other136_payloads_read=0, replay_allowed=False)
    raw = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(len(raw) <= 131072, 'reader output cap')
    print(raw.decode(), flush=True)
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, metadata_file_pins=EVIDENCE, metadata_bytes_read=READ, request_id=S['request_id'], replay_allowed=False, target_GO=False)), flush=True)
        raise SystemExit(1)
