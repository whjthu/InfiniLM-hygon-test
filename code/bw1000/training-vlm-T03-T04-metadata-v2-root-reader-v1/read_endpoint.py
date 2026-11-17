"""Read only the new staged artifacts/receipts; never open original model/run files."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import struct
import time

H = lambda raw: hashlib.sha256(raw).hexdigest()
PINS = {}


def require(test, message):
    if not test:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8') + b'\n'


def stable(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def regular(path, maximum, expected_bytes=None, expected_SHA=None, retain=False):
    path = Path(path)
    for item in [*reversed(path.parents), path]:
        require(not item.is_symlink(), 'read-only path contains symlink')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb', buffering=0) as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
                and 0 < before.st_size <= maximum, 'artifact type/owner/size differs')
        require(expected_bytes is None or before.st_size == expected_bytes, 'artifact byte count differs')
        digest, kept, consumed = hashlib.sha256(), [], 0
        while consumed < before.st_size:
            block = stream.read(min(1024**2, before.st_size-consumed))
            require(bool(block), 'artifact truncated')
            consumed += len(block)
            digest.update(block)
            if retain:
                kept.append(block)
        require(stable(before) == stable(os.fstat(stream.fileno())) == stable(path.stat()),
                'artifact changed during read')
    require(expected_SHA is None or digest.hexdigest() == expected_SHA, 'artifact SHA differs')
    pin = dict(bytes=consumed, SHA256=digest.hexdigest())
    PINS[path.relative_to(ROOT).as_posix()] = pin
    return b''.join(kept) if retain else pin


def record(name, maximum=262144):
    return json.loads(regular(ROOT/name, maximum, retain=True))


def context(value):
    require(all(value.get(key) == expected for key, expected in CONTEXT.items()), 'original context differs')


def post(intent):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(intent['boot_id'] == boot and intent['uid'] == os.getuid(), 'observer host boot/UID differs')
    path = Path('/proc')/str(intent['pid'])/'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return dict(original_owned_process_live=False, original_PID_absent=True,
                    original_startticks=intent['startticks'], boot_matches=True)
    ticks = int(fields[19])
    return dict(original_owned_process_live=ticks == intent['startticks'], original_PID_absent=False,
                current_startticks=ticks, original_startticks=intent['startticks'], boot_matches=True)


def costs(value):
    require(value['CPU_affinity'] and len(value['CPU_affinity']) == len(set(value['CPU_affinity'])) == 2
            and value['nice'] == 10 and value['RLIMIT_AS_bytes'] == 512*1024**2
            and value['actual_GPU_cards'] == 0, 'original endpoint resource contract differs')
    cost = value['system_cost']
    require(cost['nice'] == 10 and cost['self_maxrss_native_unit'] == 'Linux_KiB'
            and cost['RLIMIT_AS_bytes'] == 512*1024**2, 'original cost unit/limit differs')
    for key in ['endpoint_wall_seconds', 'self_user_CPU_seconds', 'self_system_CPU_seconds', 'self_maxrss_bytes']:
        require(type(cost[key]) in (int, float) and math.isfinite(cost[key]) and cost[key] >= 0,
                'invalid original process cost')
    require(0 < cost['endpoint_wall_seconds'] <= 300 and cost['self_maxrss_bytes'] > 0
            and cost['GPU_busy'] == 'NOT_MEASURED'
            and cost['kernel_and_filesystem_physical_readahead_IO'] == 'UNKNOWN_NOT_MEASURED',
            'cost qualification boundary differs')
    return cost


def initial_binding(raw):
    value = json.loads(raw)
    require(canonical(value) == raw and value['schema'] == 'T03_T04_initial_dependencies_v1',
            'initial binding encoding differs')
    context(value['context'])
    require(value['initial_tensor_bytes_read'] == 0 and value['cross_platform_RNG_or_dropout_qualified'] is False,
            'initial qualification inflated')
    require(set(value['cases']) == set(PLAN['cases']), 'initial case denominator differs')
    summary = {}
    for key, original in PLAN['cases'].items():
        case = value['cases'][key]
        rows = case['rows']
        require(case['rank_denominator'] == 8 and [x['rank'] for x in rows] == list(range(8)), 'rank denominator differs')
        output = []
        for rank, row in enumerate(rows):
            for field, relative in [('source_native', 'train/nvidia-baseline/rank-%02d.json'%rank),
                                    ('source_capture_receipt', 'capture/rank-%02d/receipt.json'%rank)]:
                require(row[field] == dict(path=original['run_root']+'/'+relative, **original['records'][relative]),
                        'initial dependency original metadata pin differs')
            require(row['initial_path'] == original['run_root']+'/capture/rank-%02d/initial.pt'%rank
                    and type(row['initial_file_bytes']) is int and 0 < row['initial_file_bytes'] < 2*1024**3
                    and re.fullmatch('[0-9a-f]{64}', row['initial_file_SHA256']) is not None
                    and re.fullmatch('[0-9a-f]{64}', row['named_initial_digest']) is not None
                    and row['native_global_step'] == 25 and row['captured_optimizer_updates'] == 2
                    and row['initial_tensor_read'] is False, 'initial dependency differs')
            output.append({name:row[name] for name in ['rank', 'initial_path', 'initial_file_bytes',
                                                      'initial_file_SHA256', 'named_initial_digest']})
        named = len({x['named_initial_digest'] for x in output}) == 1
        serialized = len({x['initial_file_SHA256'] for x in output}) == 1
        require(case['named_digest_all8_agree'] == named and case['file_SHA_all8_agree'] == serialized,
                'initial agreement differs')
        summary[key] = dict(rows=output, named_digest_all8_agree=named, file_SHA_all8_agree=serialized)
    return summary


def reservation(value, intent):
    require(value == intent['parent_reservations'] and value['reserved_nodes'] == 80
            and value['reserved_cards'] == 640 and value['cards_are_reservation_not_busy'] is True
            and value['GPU_busy'] == 'NOT_MEASURED', 'reservation receipt differs')
    rows, nodes = value['parents'], []
    require(len(rows) == 2 and {x['job_id'] for x in rows} == set(PLAN['parent_jobs']), 'parent denominator differs')
    for row in rows:
        tres = dict(x.split('=', 1) for x in row['AllocTRES'].split(','))
        require(row['state'] == 'RUNNING' and row['node_count'] == 40 and row['reserved_DCU'] == 320
                and tres['node'] == '40' and tres['gres/dcu'] == '320' and row['numeric_owner_matches'] is True
                and row['numeric_owner_uid'] == intent['uid'] and len(row['nodes']) == 40,
                'actual parent reservation/owner differs')
        nodes.extend(row['nodes'])
    require(len(nodes) == len(set(nodes)) == 80
            and CONTEXT['epoch'] <= value['observed_epoch'] <= CONTEXT['deadline'], 'parent80/epoch differs')
    return dict(parent_jobs=PLAN['parent_jobs'], actual_parent_cards=[320, 320], unique_nodes=80,
                reserved_cards=640, reserved_not_busy=True, observed_epoch=value['observed_epoch'],
                original_receipt_SHA256=H(canonical(value)))


def source(intent, outcome):
    require(outcome['passed'] is True and outcome['metadata_only'] is True and outcome['full_weight_identity'] is False,
            'source original failed or inflated')
    require(outcome['source_files_read'] == 67 and outcome['source_bytes_read'] == outcome['source_logical_bytes_consumed'] == 32327445
            and outcome['member_count'] == 36 and outcome['header_python_buffering'] == 0
            and outcome['physical_IO'] == 'UNKNOWN_NOT_MEASURED', 'source IO boundary differs')
    validation = record('validation.json', 65536)
    context(validation)
    require(validation['validated'] is True and validation['metadata_only'] is True
            and validation['source_validation_not_receiver_success'] is True,
            'source validation boundary differs')
    bundle = ROOT/'bundle.jsonframe.gz'
    require(not bundle.is_symlink(), 'source bundle symlink rejected')
    st = bundle.stat()  # Intentionally no open/read of the compressed bundle body.
    require(stat.S_ISREG(st.st_mode) and st.st_uid == intent['uid']
            and 0 < st.st_size <= PLAN['max_compressed_bytes'], 'source durable bundle stat differs')
    require(validation['bundle']['name'] == 'bundle.jsonframe.gz' and
            validation['bundle']['bytes'] == st.st_size == outcome['bundle_bytes']
            and validation['bundle']['SHA256'] == outcome['bundle_SHA256'], 'durable/source bundle identity differs')
    require(validation['initial_bindings']['name'] == 'initial-bindings.json', 'durable binding name differs')
    raw = regular(ROOT/'initial-bindings.json', 65536, validation['initial_bindings']['bytes'],
                  validation['initial_bindings']['SHA256'], retain=True)
    require(outcome['durable_source_artifacts'] == validation, 'durability receipt differs')
    return dict(bundle=dict(bytes=st.st_size, SHA256=outcome['bundle_SHA256'], body_rehashed=False),
                initial_binding_SHA256=H(raw), initial_dependencies=initial_binding(raw),
                logical_source_read_bytes=32327445, source_files_read=67, output_member_denominator=36,
                physical_IO='UNKNOWN_NOT_MEASURED', original_system_cost=costs(outcome))


def receiver(intent, outcome):
    require(outcome['passed'] is True and outcome['completed_members'] == outcome['member_denominator'] == 36
            and outcome['initial_rank_denominator'] == 16 and outcome['metadata_only'] is True
            and outcome['GPU_GO'] is False and outcome['Model_GO'] is False and outcome['full_weight_identity'] is False,
            'receiver original failed or denominator/qualification differs')
    rows = outcome['files']
    names = set(PLAN['members']) | {'initial-bindings.json'}
    require(len(rows) == len({x['name'] for x in rows}) == 36 and {x['name'] for x in rows} == names,
            'receiver original member set differs')
    observed = {}
    for row in rows:
        name = row['name']
        if name == 'initial-bindings.json':
            raw = regular(ROOT/name, 65536, row['bytes'], row['SHA256'], retain=True)
            dependencies = initial_binding(raw)
            require(H(raw) == outcome['initial_binding_SHA256'], 'receiver binding SHA differs')
        else:
            pin = PLAN['members'][name]
            if pin['kind'] == 'metadata':
                require(row['bytes'] == pin['bytes'] and row['SHA256'] == pin['SHA256'], 'metadata original pin differs')
                regular(ROOT/name, PLAN['max_file_bytes'], row['bytes'], row['SHA256'])
            else:
                exported = regular(ROOT/name, PLAN['max_file_bytes'], row['bytes'], row['SHA256'], retain=True)
                require(len(exported) == 8+pin['header_bytes'] and struct.unpack('<Q', exported[:8])[0] == pin['header_bytes']
                        and H(exported[8:]) == pin['header_SHA256'], 'exported header differs')
        observed[name] = dict(bytes=row['bytes'], SHA256=row['SHA256'])
    return dict(bundle=dict(bytes=outcome['bundle_bytes'], SHA256=outcome['bundle_SHA256']),
                actual_written_members=36, members=observed, initial_binding_SHA256=outcome['initial_binding_SHA256'],
                initial_dependencies=dependencies, original_system_cost=costs(outcome),
                reservation=reservation(outcome['parent_reservations'], intent))


def main():
    global ROOT
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('independent reader60s')))
    signal.alarm(60)
    ROOT = Path(PLAN['source_namespace'] if MODE == 'source' else PLAN['target_namespace'])
    require(ROOT.is_dir() and not ROOT.is_symlink(), 'original endpoint namespace absent')
    intent, outcome = record('intent.json'), record('outcome.json')
    context(intent); context(outcome)
    require(CONTEXT['deadline'] == CONTEXT['epoch']+300
            and all(outcome[key] == intent[key] for key in ['pid', 'startticks', 'boot_id', 'uid']),
            'endpoint identity/epoch differs')
    process = post(intent)
    require(process['original_owned_process_live'] is False, 'original endpoint still live')
    details = source(intent, outcome) if MODE == 'source' else receiver(intent, outcome)
    require(outcome['weight_payload_bytes_read'] == outcome['initial_tensor_bytes_read'] == 0,
            'application tensor consumption differs')
    print(json.dumps(dict(passed=True, mode=MODE, context=CONTEXT, process=process, details=details,
                          original_record_pins=PINS, no_new_Slurm_GPU_Torch=True), allow_nan=False))


if __name__ == '__main__':
    main()
