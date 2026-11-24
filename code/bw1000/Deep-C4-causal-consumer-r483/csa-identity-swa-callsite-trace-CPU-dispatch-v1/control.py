"""Run one zero-GRES trace build alongside the exact live O01 standby."""

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'Deep-CSA-identity-SWA-callsite-trace-CPU-v1'
PARENT, NODE = '1194900', 'f12r3n17'
STANDBY_STEP = PARENT + '.696'
STANDBY_REQUEST = '801ad53122ae4310806ec81e7e66a6fe'
STANDBY_START, STANDBY_SECONDS = 1795493700.638613, 10719
STANDBY_LABEL = 'training80-dcu-standby-' + STANDBY_REQUEST[:12] + '-' + NODE
STANDBY = ROOT / 'results' / ('training80-dcu-standby-' + STANDBY_REQUEST) / NODE
STANDBY_TRES = 'cpu=8,gres/dcu=8,mem=16G,node=1'
WORKER_SHA = 'ff23cdf65815fa184fcc9b3db6bf718f062ed6ec0e18a77f7ee936d6503326da'
EXPECTED_SHA = '40401748d7887b563d2c91ff01fd1a08227deb4d2bbf9243ecb44506d85dab7d'
STAGE_SHA = '1fb25134d646af3c234bbad95d5526872ab453822194ad1be13ec83336089611'
REQUEST = sha256(f'{LABEL}:{PARENT}:{NODE}:{STAGE_SHA}'.encode()).hexdigest()[:32]
STAGE = ROOT / 'incoming' / LABEL
OUTPUT = ROOT / 'candidates' / LABEL
CONTROLLER = ROOT / 'results' / (LABEL + '-controller')
ACCOUNTING_COLUMNS = ['JobIDRaw', 'JobName%100', 'State', 'ExitCode', 'ElapsedRaw', 'NodeList', 'AllocTRES', 'Start', 'End']
EXPORTS = ['modern_deep_csa_commit_trace_arm_v1', 'modern_deep_csa_commit_trace_clear_v1',
           'modern_deep_csa_commit_trace_read_v1']
HEX64 = re.compile('[0-9a-f]{64}')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path, limit, pin=None):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit and (pin is None or digest(raw) == pin), 'wrong bytes: ' + path.name)
    return raw


def write(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def encode(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + '\n').encode()


def query(*argv, timeout=7):
    raw = subprocess.check_output(argv, stdin=subprocess.DEVNULL, timeout=timeout)
    require(len(raw) < 2 << 20, 'scheduler query exceeded its bound')
    return raw.decode().splitlines()


def fields(line):
    return dict(part.split('=', 1) for part in line.split() if '=' in part)


def accounting(query_rows=query):
    rows = [line.split('|') for line in query_rows('sacct', '-n', '-P', '-j', PARENT,
             '-S', '2026-11-24T00:00:00', '-o', ','.join(ACCOUNTING_COLUMNS)) if line]
    require(all(len(row) == len(ACCOUNTING_COLUMNS) for row in rows), 'invalid accounting layout')
    exact = [row for row in rows if row[1] == LABEL]
    require(len(exact) <= 1 and all(re.fullmatch(PARENT + r'\.[0-9]+', row[0]) and row[5] == NODE for row in exact),
            'compile label resolved to a different step or node')
    return exact[0] if exact else None


def admitted_standby(query_rows=query, wall=time.time, *, require_compile_window=True):
    [job] = list(map(fields, query_rows('scontrol', '-o', 'show', 'job', PARENT)))
    nodes = query_rows('scontrol', 'show', 'hostnames', job['NodeList'])
    require(job['JobId'] == PARENT and job['JobState'] == 'RUNNING' and job['NumNodes'] == '40'
            and job['UserId'].endswith(f'({os.getuid()})') and 'gres/dcu=320' in job['AllocTRES'].split(',')
            and len(set(nodes)) == len(nodes) == 40 and NODE in nodes, 'current parent or membership differs')
    active = []
    for row in map(fields, query_rows('scontrol', '-o', 'show', 'step', PARENT)):
        handle = row.get('StepId', '')
        if not re.fullmatch(PARENT + r'\.[0-9]+', handle) or row.get('State') not in ('RUNNING', 'PENDING', 'COMPLETING'):
            continue
        node_list = row['NodeList']
        members = query_rows('scontrol', 'show', 'hostnames', node_list) if re.search(r'[\[,]', node_list) else [node_list]
        if NODE in members:
            active.append(row)
    require(len(active) == 1, 'expected only the known standby on this node')
    [record] = active
    expected = {'StepId': STANDBY_STEP, 'Name': STANDBY_LABEL, 'State': 'RUNNING', 'NodeList': NODE, 'TRES': STANDBY_TRES}
    require(all(record.get(key) == value for key, value in expected.items()), 'exact running eight-card standby required')
    before = json.loads(read(STANDBY / 'expected.json', 8192, EXPECTED_SHA))
    worker = json.loads(read(STANDBY / 'worker.json', 8192, WORKER_SHA))
    require(before['seconds'] == STANDBY_SECONDS and before['request'] == STANDBY_REQUEST
            and worker['epoch'] == STANDBY_START and worker['step'] == STANDBY_STEP
            and worker['worker_pid'] == 1710860 and worker['synthetic'] is True
            and not os.path.lexists(STANDBY / 'terminal.json'), 'original standby receipt or worker differs')
    cards = [json.loads(read(STANDBY / f'card-{index}.json', 8192)) for index in range(8)]
    observed = wall()
    require(len(set(worker['pci'])) == 8 and set(worker['pci']) == {card['pci'] for card in cards}
            and all(card['active'] is True and card['calls'] > 0 and 0 <= observed - card['epoch'] < 30 for card in cards),
            'fresh eight-card worker receipts required')
    if require_compile_window:
        require(STANDBY_START + STANDBY_SECONDS - observed > 360, 'not enough time before standby handoff')
    return dict(expected, observed_epoch=observed, worker_pid=worker['worker_pid'], eight_card_receipts_fresh=True,
                latest_calls={card['pci']: card['calls'] for card in cards})


def stage_files(envelope):
    manifest = base64.b64decode(envelope['manifest'], validate=True)
    require(digest(manifest) == STAGE_SHA, 'stage manifest changed')
    pins = json.loads(manifest)['files_SHA256']
    require(set(envelope['files']) == set(pins) and len(pins) == 10, 'stage inventory changed')
    files = {}
    for name, pin in pins.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and len(path.parts) <= 2, 'invalid stage member path')
        raw = base64.b64decode(envelope['files'][name], validate=True)
        require(0 < len(raw) < 256 << 10 and digest(raw) == pin, 'stage member changed: ' + name)
        files[name] = raw
    return dict(files, **{'STAGE_FILES.json': manifest})


def command(epoch, claim_sha, outer):
    env = {'TUNING_CPU_PARENT': PARENT, 'TUNING_CPU_NODE': NODE, 'TUNING_CPU_EPOCH': str(epoch),
           'TUNING_CPU_DEADLINE': str(epoch + 300), 'TUNING_CPU_CLAIM_SHA': claim_sha,
           'TUNING_CPU_STAGE_SHA': STAGE_SHA, 'WAVE_INSIDE_R2323': LABEL + '/inside.sh'}
    return ['env', *(f'{k}={v}' for k, v in env.items()), 'srun', '--jobid=' + PARENT,
            '--exact', '--immediate=5', '--nodes=1', '--ntasks=1', '--nodelist=' + NODE,
            '--cpus-per-task=4', '--mem=8192M', '--gres=none', '--time=00:05:00',
            '--kill-on-bad-exit=1', '--job-name=' + LABEL, '/bin/bash', str(outer), 'deep']


def verify_result(row, epoch, claim_sha):
    require(row[2:4] == ['COMPLETED', '0:0'] and row[4].isdecimal() and int(row[4]) <= 300
            and row[6] in ('cpu=4,mem=8G,node=1', 'cpu=4,mem=8192M,node=1'), 'unexpected CPU terminal accounting')
    raw = read(OUTPUT / 'CPU-result.json', 128 << 10)
    doc = json.loads(raw)
    expected = dict(original_step=row[0], node=NODE, epoch=epoch, deadline=epoch + 300, GRES='none',
                    claim_SHA256=claim_sha, stage_manifest_SHA256=STAGE_SHA, completed=True, passed=True,
                    planned_HOST_TUs=1, planned_GPU_TUs=0, compile_invocations=1, link_invocations=1,
                    fresh_M_paths=998, actual_MD_paths=998, reused_objects_verified=49, loader_paths=68,
                    trace_CABI=EXPORTS, explicit_Native_calls=0, actual_GPU_steps=0, weight_reads=0, device_qualified=False)
    require(all(type(doc[key]) is type(value) and doc[key] == value for key, value in expected.items())
            and doc['strong_exports'] == doc['parent_strong_exports'] + 3, 'CPU result qualification differs')
    for name, pin in (('TU07.o','object_SHA256'), ('TU07.before.d','fresh_M_SHA256'), ('TU07.actual.d','actual_MD_SHA256'),
                      ('libDeep_csa_identity_swa_callsite_trace.so','library_SHA256')):
        require(HEX64.fullmatch(doc[pin]), 'missing output digests')
        read(OUTPUT / name, 32 << 20, doc[pin])
    return dict(receipt_SHA256=digest(raw), library_SHA256=doc['library_SHA256'], object_SHA256=doc['object_SHA256'],
                strong_exports=doc['strong_exports'], parent_strong_exports=doc['parent_strong_exports'],
                fresh_M_paths=998, actual_MD_paths=998, fresh_M_SHA256=doc['fresh_M_SHA256'],
                actual_MD_SHA256=doc['actual_MD_SHA256'], trace_CABI=EXPORTS)


def compile_original(envelope):
    claim_raw = base64.b64decode(envelope['claim'], validate=True)
    claim_sha = digest(claim_raw)
    claim = json.loads(claim_raw)
    require(HEX64.fullmatch(envelope['controller_SHA256']) and claim['controller_SHA256'] == envelope['controller_SHA256']
            and claim['request'] == REQUEST and claim['label'] == LABEL and claim['stage_manifest_SHA256'] == STAGE_SHA
            and claim['parent'] == PARENT and claim['node'] == NODE and claim['standby'] == STANDBY_STEP
            and claim['cpus'] == 4 and claim['memory_MiB'] == 8192 and claim['GRES'] == 0 and claim['wall_seconds'] == 300
            and claim['replay_allowed'] is False and claim['GPU_GO'] is False, 'compile claim identity changed')
    outer = base64.b64decode(envelope['outer'], validate=True)
    require(digest(outer) == claim['outer_SHA256'], 'preserved outer wrapper changed')
    files = stage_files(envelope)
    require(not any(os.path.lexists(path) for path in (STAGE, OUTPUT, CONTROLLER))
            and accounting() is None, 'compile namespace or original label already exists')
    before = admitted_standby()
    STAGE.mkdir(mode=0o700)
    (STAGE / 'native-src').mkdir(mode=0o700)
    for name, raw in files.items():
        write(STAGE / name, raw)
    CONTROLLER.mkdir(mode=0o700)
    outer_path = CONTROLLER / 'outer.sh'
    write(outer_path, outer)
    epoch, started = time.time(), time.monotonic()
    intent = dict(request=REQUEST, epoch=epoch, deadline=epoch + 300, standby=STANDBY_STEP,
                  claim_SHA256=claim_sha, stage_manifest_SHA256=STAGE_SHA, replay_allowed=False)
    write(CONTROLLER / 'intent.json', encode(intent))
    result = dict(intent, state='UNKNOWN', accounting_columns=ACCOUNTING_COLUMNS,
                  before=before, observations=[], actual_GPU_steps=0)

    def left():
        return min(epoch + 300 - time.time(), 300 - (time.monotonic() - started))

    try:
        with (CONTROLLER / 'srun.log').open('xb') as logfile:
            try:
                result['srun_rc'] = subprocess.run(command(epoch, claim_sha, outer_path), stdin=subprocess.DEVNULL,
                                                   stdout=logfile, stderr=subprocess.STDOUT, timeout=left()-25).returncode
            except subprocess.TimeoutExpired:
                result['srun_timeout'] = True
        row = None
        for _ in range(6):
            if left() <= 9:
                break
            row = accounting()
            result['observations'].append(row)
            if row and row[2].split()[0] in TERMINAL:
                result.update(state='TERMINAL_UNVERIFIED', accounting=row)
                result.update(verify_result(row, epoch, claim_sha), state='COMPILE_VERIFIED')
                break
            time.sleep(1)
        after = admitted_standby(require_compile_window=False)
        result['after'] = after
        result['standby_counter_growth_DCU'] = sum(after['latest_calls'][pci] > calls for pci, calls in before['latest_calls'].items())
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, subprocess.SubprocessError) as error:
        result['error'] = f'{type(error).__name__}: {str(error)[:180]}'
    finally:
        write(CONTROLLER / 'accounting-result.json', encode(result))
    return result


if __name__ == '__main__':
    os.umask(0o077)
    value = sys.stdin.buffer.read((1 << 20) + 1)
    require(0 < len(value) <= 1 << 20, 'wire exceeds its bound')
    outcome = compile_original(json.loads(value))
    print(encode(outcome).decode(), end='', flush=True)
    raise SystemExit(int(outcome['state'] != 'COMPILE_VERIFIED'))
