"""Root injects exact CONTEXT/READ_DEADLINE. Read saved JSON only after original terminal gate."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time

ROOT = Path('/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-fresh-control-A100-v1')


def need(value, reason):
    if not value:
        raise ValueError(reason)


def main():
    remaining = READ_DEADLINE-time.time()
    need(0 < remaining <= 60, 'READONLY_EPOCH')
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))  # Original <=60s includes bounded stdout tail.
    signal.setitimer(signal.ITIMER_REAL, remaining)
    cores = sorted(set(os.sched_getaffinity(0))-{0,1})
    need(len(cores) >= 2, 'READONLY_TWO_CPU')
    os.sched_setaffinity(0, cores[:2])
    os.nice(10); need(os.getpriority(os.PRIO_PROCESS, 0) == 10, 'READONLY_NICE10')
    result = dict(context=CONTEXT, files={}, stage='TERMINAL_GATE', original_execution_state='UNKNOWN',
                  remote_writes=0, Torch_GPU_calls=0, state_body_reads=0, replay_allowed=False,
                  CPU_affinity=cores[:2], actual_nice=10, source_cost_semantics='metadata read only; no GPU or busy claim')
    total = 0

    def read(key, path, cap=2*1024**2):
        nonlocal total
        path = Path(path)
        for item in [*reversed(path.parents), path]:
            need(not item.is_symlink(), 'REGISTERED_PATH_SYMLINK')
        if not path.exists():
            result['files'][key] = dict(exists=False)
            return None
        before = path.stat()
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
             and 0 < before.st_size <= cap and total+before.st_size <= 6*1024**2, 'SMALL_ORIGINAL_BOUND')
        with path.open('rb', buffering=0) as stream:
            raw = stream.read(cap+1); after = os.fstat(stream.fileno())
        stable = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(stable(before) == stable(after) == stable(path.stat()) and len(raw) == before.st_size, 'ORIGINAL_CHANGED')
        value = [json.loads(line) for line in raw.splitlines()] if path.suffix in ('.jsonl',) else json.loads(raw)
        result['files'][key] = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest(), value=value)
        total += len(raw)
        return value

    outcome = read('run/outcome.json', ROOT/'run/outcome.json')
    if outcome is None:
        return result
    need(outcome['request_id'] == CONTEXT['request_id'] and outcome['source_publication'] == CONTEXT['publication']
         and outcome['epoch'] == CONTEXT['epoch'] and outcome['deadline'] == CONTEXT['epoch']+900,
         'EXACT_ORIGINAL_REQUEST')
    config = read('run/domain-config.json', ROOT/'run/domain-config.json')
    owner = read('run/child-owner.json', ROOT/'run/child-owner.json')
    read('run/after.json', ROOT/'run/after.json')
    read('run/failure-private.json', ROOT/'run/failure-private.json', 65536)
    if config is None or owner is None:
        result.update(missing_original_owned_identity=True, total_read_bytes=total)
        return result
    original_host = config['supervisor']['host']
    observer_host = socket.gethostname()
    observation = dict(original_host=original_host, observer_host=observer_host, processes=[])
    result['original_process_observation'] = observation
    if observer_host != original_host:
        result.update(total_read_bytes=total, cross_host_original_process_state='UNKNOWN')
        return result
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    any_live = False
    for identity in (config['supervisor'], owner['child']):
        need(identity['host'] == original_host and identity['uid'] == os.getuid()
             and identity['boot_id'] == config['supervisor']['boot_id']
             and type(identity['PID']) is int and identity['PID'] > 0
             and type(identity['startticks']) is int and identity['startticks'] > 0, 'ORIGINAL_OWNED_IDENTITY')
        live = False
        if boot == identity['boot_id']:
            try:
                fields = (Path('/proc')/str(identity['PID'])/'stat').read_text().rsplit(')', 1)[1].split()
                live = fields[0] not in ('Z', 'X') and int(fields[19]) == identity['startticks']
            except FileNotFoundError:
                pass
        any_live |= live
        observation['processes'].append(dict(PID=identity['PID'], startticks=identity['startticks'], original_process_live=live,
                                            same_boot=boot == identity['boot_id']))
    if any_live:
        result.update(total_read_bytes=total)
        return result
    result['original_execution_state'] = 'TERMINAL_PASS' if outcome['engineering_completed'] else 'TERMINAL_FAIL'
    result['stage'] = 'READ_SAVED_METADATA_AFTER_TERMINAL'
    for name in ('control-state.json', 'exec-intent.json', 'held-child-anchor.json', 'owned-registry.json',
                 'resource-samples.json', 'common-entry-receipt.jsonl'):
        read('run/'+name, ROOT/'run'/name)
    for name in ('SOURCE_FILES.json', 'CLAIM.json', 'CONFIG.template.json', 'INSTALL_PLAN.json',
                 'MERGED_FRAMEWORK_FILES.json', 'supervisor/SOURCE_FILES.json', 'supervisor/CLAIM.json'):
        read('src/'+name, ROOT/'src'/name)
    for rank in range(8):
        prefix = 'run/witness/rank-%02d/' % rank
        for name in ('owner.json', 'domain-plan.json', 'progress.json', 'receipt.json', 'initial-state.json',
                     'loader.json', 'loader-set-epoch.json', 'trainer-method-providers.json',
                     'native-recipe-providers.json', 'microbatches.json', 'global800-occurrences.json',
                     'first-update-1-actual-delta.json', 'first-update-2-actual-delta.json',
                     'update-25-after.json', 'update-25-scheduler.json'):
            read(prefix+name, ROOT/'run'/('witness/rank-%02d'%rank)/name)
    for step in (10, 20, 25):
        key = 'run/witness/rank-00/checkpoint-%d-files.json' % step
        read(key, ROOT/key)
    for key in ('native45', 'training800', 'occurrences', 'pair_initial_reference'):
        pin = config[key]
        # Exact declared existing small references only; never follow arbitrary config paths.
        known = {
            'native45': '/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-generic-control-native-CPU-v1/run/native/native-identities-private.json',
            'training800': '/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-paired-training-views-CPU-v1/control/train-800.jsonl',
            'occurrences': '/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-paired-training-views-CPU-v1/control/exposure-ledger-private.json',
            'pair_initial_reference': '/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-fresh-domain-A100-v1/run/witness/rank-00/initial-state.json'}
        need(pin['path'] == known[key], 'EXACT_REFERENCE_PATH')
        read('reference/'+key, pin['path'])
    initial = config['initial_state']; path = Path(initial['path'])
    need(path.as_posix() == '/home/whj/nvidia-baseline/alignment-assets/training-USGS-T04-fresh-domain-A100-v1/run/witness/initial.pt'
         and not path.is_symlink(), 'EXACT_INITIAL_STAT_ONLY')
    st = path.stat()
    result['initial_body_stat_only'] = dict(bytes=st.st_size, regular=stat.S_ISREG(st.st_mode), uid=st.st_uid,
                                         body_read=False, SHA256_not_independently_rehashed=True)
    result['total_read_bytes'] = total
    return result


if __name__ == '__main__':
    print(json.dumps(main(), separators=(',', ':'), allow_nan=False))
