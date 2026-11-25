"""Issue the published T09 Conv3d scope once and preserve original Slurm evidence."""

import base64
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
REQUEST = '7e7233c2b6ab4db1b55d4dee6c23ad02'
LABEL = 't09-miopen-conv3d-7e7233c2b6ab'
MANIFEST = '1bc4ccb26fc2eef2c12b2453fa56f02d387db14ff179ba7e01f2504796e7b4d7'
ARCHIVE = 'e8e12ff40e6a960054df67a72612fc510fcaa3658b5e40052f4f955ddb9991b0'
SEALED = 'f9181a6542df478b1c1202951e47d7b6011efd60170135c4f5429da90cbadcab'
PARENT, NODE = '1103606', 'f13r3n13'
STAGE = ROOT / 'incoming' / LABEL
AUDIT = ROOT / 'results' / (LABEL + '-controller')
COLS = 'JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'
ENV = dict(os.environ, TZ='UTC', SLURM_TIME_FORMAT='%Y-%m-%dT%H:%M:%S')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL'}
MEMBERS = {'host.py', 'inside.py', 'scope.py', 'mount_observation.py', 'runtime_pci.py', 'smoke.py'}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def write(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(data)


def owned_file(path, cap=65536):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        meta = os.fstat(stream.fileno())
        need(stat.S_ISREG(meta.st_mode) and meta.st_uid == os.getuid()
             and meta.st_size <= cap, 'OWNED_BOUNDED_PRIVATE_FILE')
        raw = stream.read(cap + 1)
        need(len(raw) == meta.st_size, 'PRIVATE_FILE_CHANGED_OR_EXCEEDS_BOUND')
    return raw


def query(*args):
    data = subprocess.check_output(args, env=ENV, timeout=8)
    need(len(data) < 2 << 20, 'BOUNDED_SLURM_OUTPUT')
    return data.decode().splitlines()


def fields(line):
    return dict(item.split('=', 1) for item in line.split() if '=' in item)


def hosts(expression):
    if re.fullmatch(r'f\d{2}r\dn\d{2}', expression):
        return [expression]
    return query('scontrol', 'show', 'hostnames', expression)


def is_numeric(step):
    return re.fullmatch(PARENT + r'\.\d+', step) is not None


def node_clear():
    (job,) = [fields(line) for line in query('scontrol', 'show', 'job', PARENT, '--oneliner')]
    (parent_view,) = query('squeue', '-h', '-j', PARENT, '-o', '%i|%T|%N')
    parent, state, members = parent_view.split('|')
    controller_hosts, queue_hosts = hosts(job['NodeList']), hosts(members)
    need(job['JobId'] == parent == PARENT and job['JobState'] == state == 'RUNNING'
         and job['UserId'].endswith(f'({os.getuid()})')
         and len(set(controller_hosts)) == len(controller_hosts) == 40
         and set(controller_hosts) == set(queue_hosts) and NODE in controller_hosts
         and 'gres/dcu=320' in job['AllocTRES'].split(','), 'EXACT_OWNED_40_NODE_PARENT')
    for line in query('scontrol', 'show', 'step', PARENT, '--oneliner'):
        row = fields(line)
        if is_numeric(row.get('StepId', '')) and row.get('State') not in TERMINAL:
            need(NODE not in hosts(row['NodeList']), 'NODE_HAS_ACTIVE_CONTROLLER_STEP')
    for line in query('squeue', '--steps', '-h', '-j', PARENT, '-o', '%i|%N'):
        step, expression = line.split('|')
        if is_numeric(step):
            need(NODE not in hosts(expression), 'NODE_HAS_ACTIVE_QUEUE_STEP')


def rows(job=PARENT):
    result = [line.split('|') for line in query('sacct', '-nP', '-j', job, '--format', COLS)]
    need(all(len(row) == 9 for row in result), 'NINE_SLURM_ACCOUNTING_COLUMNS')
    return result


def terminal(phase, deadline, step=None):
    label = LABEL + '-' + phase
    limit = min(deadline - 3, time.time() + 12)
    result = []
    while time.time() < limit:
        result = [row for row in rows(step or PARENT) if row[1] == label]
        need(len(result) <= 1, 'ONE_EXACT_ORIGINAL_PHASE')
        if result and result[0][2] in TERMINAL:
            row = result[0]
            demand = {'cpu=4', 'node=1'}
            if phase == 'gpu':
                demand.add('gres/dcu=1')
            tres = row[6].split(',')
            need(is_numeric(row[0]) and (step is None or row[0] == step) and row[5] == NODE
                 and len(tres) == len(demand) + 1
                 and set(tres) in (demand | {'mem=8G'}, demand | {'mem=8192M'}),
                 'EXACT_ORIGINAL_PHASE_RESOURCE_TERMINAL')
            return row
        time.sleep(.35)
    return None


def bundle(payload):
    need(set(payload) == {'request', 'publication', 'sources'}
         and re.fullmatch('[0-9a-f]{32}', payload['request'])
         and re.fullmatch('[0-9a-f]{40}', payload['publication']), 'BOUND_TRANSPORT_AND_PUBLICATION')
    need(set(payload['sources']) == MEMBERS | {'SOURCE_FILES.json'}, 'EXACT_SOURCE_SET')
    sources = {name: base64.b64decode(data, validate=True) for name, data in payload['sources'].items()}
    manifest = sources['SOURCE_FILES.json']
    need(digest(manifest) == MANIFEST, 'EXACT_PUBLISHED_MANIFEST')
    record = json.loads(manifest)
    need(record['request'] == REQUEST and record['label'] == LABEL and set(record['files']) == MEMBERS,
         'EXACT_ORIGINAL_MODEL_FREE_SCOPE')
    for name, pin in record['files'].items():
        raw = sources[name]
        need(pin == dict(bytes=len(raw), SHA256=digest(raw)), 'EXACT_PUBLISHED_SOURCE')
    return sources


def phase_command(phase, authority_pin):
    return [
        'srun', '--jobid=' + PARENT, '--exact', '--nodes=1', '--ntasks=1', '--immediate=5',
        '--nodelist=' + NODE, '--cpus-per-task=4', '--mem=8192M',
        '--gres=' + ('dcu:1' if phase == 'gpu' else 'none'),
        '--time=' + ('00:01:30' if phase == 'gpu' else '00:00:45'),
        '--job-name=' + LABEL + '-' + phase, '/usr/local/python3.12/bin/python3', '-B',
        str(STAGE / 'host.py'), phase, '--manifest-sha', MANIFEST, '--authority-sha', authority_pin,
    ]


def run_phase(phase, common, **phase_authority):
    pin = write(STAGE.parent / (LABEL + '.' + phase + '.authority.json'),
                encoded(dict(common, phase=phase, single_DCU_GO=phase == 'gpu', **phase_authority)))
    argv = phase_command(phase, pin)
    remaining = common['deadline'] - time.time()
    need(remaining >= (110 if phase == 'gpu' else 8), 'INSUFFICIENT_SHARED_ABSOLUTE_BUDGET')
    write(AUDIT / (phase + '-dispatch-intent.json'),
          encoded(dict(phase=phase, argv_SHA256=digest(encoded(argv)), authority_SHA256=pin)))
    with (AUDIT / (phase + '.stdout')).open('xb') as out, (AUDIT / (phase + '.stderr')).open('xb') as err:
        call = subprocess.Popen(argv, env=ENV, stdout=out, stderr=err)
        write(AUDIT / (phase + '-srun.json'), encoded(dict(pid=call.pid, started_epoch=time.time())))
        # Leave the Slurm wall bound in force if observation of this original process expires.
        rc = call.wait(timeout=min(remaining - 3, 102 if phase == 'gpu' else 53))
    return dict(returncode=rc,
                stdout_SHA256=digest(owned_file(AUDIT / (phase + '.stdout'), 131072)),
                stderr_SHA256=digest(owned_file(AUDIT / (phase + '.stderr'), 131072)))


def execute(payload):
    source = bundle(payload)
    result = dict(request=payload['request'], scope=REQUEST, state='BEFORE_ORIGINAL_STAGE',
                  phase_results={}, replay_allowed=False)
    fd = os.open(ROOT / 'results/.training80-dcu-standby.lock',
                 os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    owns_audit = False
    try:
        need(stat.S_ISREG(os.fstat(fd).st_mode), 'EXACT_REGULAR_DISPATCH_LOCK')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        node_clear()
        namespace = [STAGE, AUDIT] + [STAGE.parent / (LABEL + '.' + phase + '.authority.json')
                                     for phase in ('pre', 'gpu', 'post')]
        labels = {LABEL + '-' + phase for phase in ('pre', 'gpu', 'post')}
        need(not any(os.path.lexists(path) for path in namespace)
             and not any(row[1] in labels for row in rows()), 'ORIGINAL_SCOPE_ALREADY_EXISTS')
        archived = [row for row in rows('1103606.3583') if row[0] == '1103606.3583']
        need(len(archived) == 1 and archived[0][2:4] == ['COMPLETED', '0:0']
             and archived[0][5] == NODE, 'EXACT_ORIGINAL_PRIVATE_HOME_TERMINAL')
        AUDIT.mkdir(mode=0o700)
        owns_audit = True
        STAGE.mkdir(mode=0o700)
        for name, raw in source.items():
            write(STAGE / name, raw)
            need(owned_file(STAGE / name) == raw, 'REMOTE_STAGE_READBACK_DIFFERS')
        epoch = time.time()
        deadline = epoch + 240
        common = dict(request=REQUEST, manifest_SHA256=MANIFEST, parent=PARENT, node=NODE,
                      root_approved=True, private_proof_archive_SHA256=ARCHIVE,
                      no_actual_model_gold_inputs=True, physical_and_scheduler_clearance=True,
                      epoch=epoch, deadline=deadline)
        result['intent_SHA256'] = write(AUDIT / 'intent.json',
                                        encoded(dict(common, transport=payload['request'],
                                                     publication=payload['publication'])))
        result['state'] = 'PRE_OBSERVATION_PENDING'
        pre_result = result['phase_results']['pre'] = run_phase('pre', common)
        pre_result['terminal'] = terminal('pre', deadline)
        pre_raw = owned_file(STAGE / 'pre.json')
        pre = json.loads(pre_raw)
        pre_physical = pre['physical']
        need(pre_result['returncode'] == 0 and pre_result['terminal']
             and pre_result['terminal'][0] == pre['step']
             and pre_result['terminal'][2:4] == ['COMPLETED', '0:0']
             and pre['request'] == REQUEST and pre['phase'] == 'pre' and pre['epoch'] == epoch
             and pre['passed'] is True and pre_physical['node'] == NODE
             and pre_physical['kfd_holder_count'] == 0 and len(set(pre_physical['pci'])) == 8,
             'EXACT_FRESH_PRE_AND_CLEAN_PHYSICAL_GATE')
        node_clear()
        pre_pin = digest(pre_raw)
        result['state'] = 'ORIGINAL_GPU_OBSERVATION_PENDING'
        gpu_result = result['phase_results']['gpu'] = run_phase(
            'gpu', common, pre_SHA256=pre_pin, sealed_3583_independent_SHA256=SEALED,
            current_route_SHA256=pre_pin)
        gpu_result['terminal'] = terminal('gpu', deadline)
        need(gpu_result['terminal'], 'ORIGINAL_GPU_TERMINAL_NOT_OBSERVED')
        intent_raw = owned_file(STAGE / 'gpu-intent.json')
        intent = json.loads(intent_raw)
        need(intent['request'] == REQUEST and intent['step'] == gpu_result['terminal'][0]
             and intent['pre_SHA256'] == pre_pin and intent['epoch'] == epoch,
             'EXACT_ORIGINAL_NUMBERED_GPU_INTENT')
        gpu_result['provider_reports'] = []
        for line in owned_file(AUDIT / 'gpu.stdout', 131072).splitlines():
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict) and parsed.get('request') == REQUEST:
                gpu_result['provider_reports'].append(parsed)
        result['state'] = 'GPU_TERMINAL_POST_OBSERVATION_PENDING'
        node_clear()
        post = result['phase_results']['post'] = run_phase(
            'post', common, pre_SHA256=pre_pin, gpu_intent_SHA256=digest(intent_raw),
            gpu_step=intent['step'])
        post['terminal'] = terminal('post', deadline)
        post_document = json.loads(owned_file(STAGE / 'post.json'))
        need(post['returncode'] == 0 and post['terminal'] and post['terminal'][0] == post_document['step']
             and post['terminal'][2:4] == ['COMPLETED', '0:0']
             and post_document.get('independent_physical_clearance') is True, 'EXACT_INDEPENDENT_POST')
        result['state'] = 'THREE_ORIGINAL_PHASES_OBSERVED'
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as err:
        result['error_class'] = type(err).__name__
        reason = str(err)
        result['qualification_error'] = reason if re.fullmatch('[A-Z0-9_]{1,100}', reason) else None
    finally:
        if owns_audit:
            result['outcome_SHA256'] = write(AUDIT / 'outcome.json', encoded(result))
        os.close(fd)
    return result


if __name__ == '__main__':
    os.umask(0o077)
    print(json.dumps(execute(json.load(sys.stdin)), sort_keys=True, allow_nan=False), flush=True)
