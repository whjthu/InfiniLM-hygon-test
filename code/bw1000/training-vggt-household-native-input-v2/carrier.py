"""One fresh household CPU step and an independent read-only terminal check."""

import base64
from datetime import datetime, timezone
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
PARENT, NODE = '1103606', 'f13r1n12'
OLD_REQUEST = '8e68f5d43e384bb1be937d38b9cff371'
OLD_LABEL = 'vggt-household-native-cpu-' + OLD_REQUEST[:12]
OLD = ROOT / 'incoming' / OLD_LABEL
REQUEST = '4c89023415bc467b94fc0ee30bd99c1c'
READ_REQUEST = '45bc5e4d4bfa4ee98a2d3a7554fb35a6'
CHECK_REQUEST = '555eca1f5cd74846bdb37b7ab461412e'
LABEL = 'vggt-household-native-cpu-' + REQUEST[:12]
RUN = ROOT / 'incoming' / LABEL
CAP = 300
PYTHON = '/usr/local/python3.12/bin/python3'
OLD_CARRIER = '225e8d1b10f1b0618f3e197458c47c7938d6ed8d717962553365a512195b01cd'
SOURCE_PIN = '6b345869c2b45516f7271a29543d2d3481dc0d1e6e77c560444fecbe65e71fac'
INPUT_PIN = '01ed8f1ae762a3b2736aeb76ebed1cca73f518072ef777613ad2bcb59f0305eb'
BACKEND_PIN = '762fc0d74a6ebb3ebbd052a8bdfbe302477a518a6868ec322bd89ef07e16c998'
INSTALL_PIN = '04ea9e6b847302d071f23a76785bc803af76b875528fe58bb4d072c2ff7196da'
ASSET_PIN = 'e72fa5f3a0d98a0872d147e3d29b699a67146112dc30b20099606419d4b9aea8'
OLD_OUTCOME = 'fd142d5e12e4483c3d491ad6cf52ab23e52c1e4ad7dd6e16369cfe5eca13afca'
OLD_INTENT = '21ec0e19f6f72f8e68912d70034f2092f02a68a1dfced7ec18f7f15b08358498'
ABORT_ARCHIVE = 'c7e4771fb51f440f783af1bff7a5733516779a2add7f6796937c2fa360e7cd2f'
OLD_PUBLICATION = 'cf37538ee36c5e8aeca660e683b81b4d83c54237'
OLD_ROW = ['1103606.3796', OLD_LABEL, 'CANCELLED by 0', '0:9', '0', NODE,
           'cpu=4,mem=48G,node=1', '2026-11-26T07:13:15', '2026-11-26T07:13:15']
PREFIX = b'VGGT_HOUSEHOLD_NATIVE_CPU '
SHA = re.compile(r'[0-9a-f]{64}')
STEP = re.compile(PARENT + r'\.[0-9]+')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def pin(raw):
    return {'SHA256': sha256(raw).hexdigest(), 'bytes': len(raw)}


def pack(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def read(path, cap, expected=None):
    require(path.resolve(strict=True) == path, 'NONCANONICAL_FILE')
    def identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(fd)
        raw = stream.read(cap + 1)
        after = os.fstat(fd)
    current = path.lstat()
    require(stat.S_ISREG(current.st_mode) and identity(before) == identity(after) == identity(current)
            and len(raw) == current.st_size <= cap, 'UNSTABLE_OR_OVERSIZE_FILE')
    require(expected is None or pin(raw)['SHA256'] == expected, 'PIN_MISMATCH')
    return raw


def source_api():
    original = read(OLD/'carrier.py', 32768, OLD_CARRIER)
    shared = {'__name__': '_original_household_carrier'}
    exec(compile(original, 'original_household_carrier.py', 'exec'), shared)
    return shared


def admissions(rehash):
    shared = source_api()
    installation = shared['installation'](INSTALL_PIN)
    require(installation['publication'] == OLD_PUBLICATION and installation['passed'] is True,
            'ORIGINAL_INSTALLATION_REQUIRED')
    source = shared['SRC']
    spec = json.loads(read(source/'SOURCE_FILES.json', 4096, SOURCE_PIN))['files']
    require(set(spec) == {'native_entry.py', 'tum_vggt_pair.py', 'INPUT_FILES.json'}
            and {x.name for x in source.iterdir()} == {*spec, 'SOURCE_FILES.json'}
            and all(pin(read(source/name, 65536)) == stamp for name, stamp in spec.items()),
            'ORIGINAL_SOURCE_CLOSURE_REQUIRED')
    asset = json.loads(read(OLD/'assets-receipt.json', 8192, ASSET_PIN))
    require(asset['passed'] is True and asset['request'] == shared['TRANSFER']
            and asset['verified_objects'] == 79 and asset['verified_manifest_bytes'] == 24200249
            and asset['input_SHA256'] == INPUT_PIN, 'ORIGINAL_79_ASSETS_REQUIRED')
    require(json.loads(read(OLD/'cpu-intent.json', 2048, OLD_INTENT))['request'] == OLD_REQUEST
            and json.loads(read(OLD/'cpu-receipt.json', 4096, OLD_OUTCOME))['accounting'] == [OLD_ROW],
            'ORIGINAL_3796_IMMUTABLE')
    shared['LABEL'] = LABEL
    scheduler = shared['baseline']()
    original_rows = [line.split('|') for line in scheduler['query'](
        ['sacct','-nP','-j',OLD_ROW[0],'-o',scheduler['COLUMNS']]) if line]
    require(original_rows == [OLD_ROW], 'ORIGINAL_3796_TERMINAL_REQUIRED')
    if rehash:
        files = shared['input_spec'](read(source/'INPUT_FILES.json', 65536, INPUT_PIN))
        require(shared['verify_tree'](shared['ASSETS'], files), 'ORIGINAL_79_ASSET_REHASH_FAILED')
    return scheduler


def command():
    return ['srun','--jobid='+PARENT,'--exact','--immediate=5','--cpu-bind=none','-N1','-n1',
            '-c4','--mem=49152M','--gres=none','-w',NODE,'-t','00:05:00','--job-name='+LABEL,
            PYTHON,'-I','-S','-B','-u',(RUN/'carrier.py').as_posix(),'worker']


def idle(a, publication):
    now = time.time()
    require(a['request'] == REQUEST and a['publication'] == publication and a['parent'] == PARENT
            and a['node'] == NODE and a['sole_zero_GRES_authority'] is True
            and a['ordinary_conflict_excluded'] is True and SHA.fullmatch(a['custody_evidence_SHA256'])
            and 0 <= now - a['issued_epoch'] <= 90 and a['expires_epoch'] >= now + 330,
            'FRESH_ROOT_AUTHORITY_REQUIRED')


def native_result(stdout, step):
    markers = [line[len(PREFIX):] for line in stdout.splitlines() if line.startswith(PREFIX)]
    require(len(markers) == 1, 'MISSING_OR_DUPLICATE_NATIVE_RESULT')
    result = json.loads(markers[0])
    require(result['numeric_step'] == step and result['input_manifest_SHA256'] == INPUT_PIN
            and result['backend_SHA256'] == BACKEND_PIN
            and result['planned_common_pairs'] == 20 and result['planned_scoring_slots'] == 40
            and result['frozen_unavailable'] == 1 and result['ready_for_scoring'] is False
            and result['actual_GPU'] == result['actual_predictions'] == result['model_weights_opened'] == 0,
            'NATIVE_RESULT_SCOPE')
    rows = result['rows']
    require([r['slot'] for r in rows] == list(range(1, 21))
            and [r['slot'] for r in rows if r['state'] == 'FROZEN_UNAVAILABLE'] == [15], 'FROZEN_DENOMINATOR')
    qualified = [r for r in rows if r['state'] == 'NATIVE_CPU_QUALIFIED']
    failed = [r for r in rows if r['state'] == 'NATIVE_CPU_FAILED']
    require(len(qualified) + len(failed) == 19
            and all(SHA.fullmatch(r['native_input_SHA256']) for r in qualified)
            and len(qualified) == result['qualified_input_pairs'] and len(failed) == result['failed_eligible']
            and result['passed'] is (len(qualified) == 19)
            and (SHA.fullmatch(result['ordered_native_batch_SHA256']) if len(qualified) == 19
                 else result['ordered_native_batch_SHA256'] is None), 'NATIVE_RESULT_COUNTS_OR_HASHES')
    return result


def starts_ends(row, intent):
    def stamp(v):
        return datetime.strptime(v, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    return intent['epoch'] - 1 <= stamp(row[7]) <= stamp(row[8]) <= intent['deadline']


def terminal(rows, intent, scheduler):
    require(len(rows) == 1 and len(rows[0]) == 9, 'ONE_NEW_ACCOUNTING_ROW_REQUIRED')
    row = rows[0]
    require(STEP.fullmatch(row[0]) and row[0] != OLD_ROW[0] and row[1] == LABEL and row[5] == NODE
            and scheduler['tres'](row[6]) and row[4].isdecimal() and int(row[4]) <= CAP
            and re.fullmatch(r'[0-9]+:[0-9]+', row[3]) and starts_ends(row, intent)
            and (row[2] in {'COMPLETED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','DEADLINE'}
                 or re.fullmatch(r'CANCELLED(?: by [0-9]+)?', row[2])), 'EXACT_NEW_TERMINAL_ROW_REQUIRED')
    return row


def launch(data):
    publication = data['publication']
    require(re.fullmatch(r'[0-9a-f]{40}', publication), 'PUBLICATION_REQUIRED')
    offline_program = base64.b64decode(data['program'], validate=True)
    require(pin(offline_program)['SHA256'] == data['program_SHA256'], 'EXACT_REVIEWED_PROGRAM')
    scheduler = admissions(True)
    argv = command()
    with scheduler['lock']():
        idle(data['authority'], publication)
        scheduler['fresh']()
        idle(data['authority'], publication)
        RUN.mkdir(mode=0o700)
        shared = source_api()
        shared['write'](RUN/'carrier.py', offline_program)
        epoch = time.time()
        intent = {'request': REQUEST, 'publication': publication, 'predecessor': OLD_ROW[0],
                  'authority': pin(pack(data['authority'])), 'argv': pin(pack(argv)),
                  'epoch': epoch, 'deadline': epoch + 325, 'program': pin(offline_program)}
        intent_pin = shared['write'](RUN/'intent.json',pack(intent))
        shared['sync'](RUN)
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Allow the shared lock to cover the immediate allocation handoff only.
        limit = time.monotonic() + 8
        handshake = 'SUBMISSION_UNKNOWN'
        while time.monotonic() < limit:
            if proc.poll() is not None:
                handshake = 'SRUN_EXITED'
                break
            try:
                rows = [scheduler['fields'](line) for line in scheduler['query'](
                    ['scontrol','show','step',PARENT,'--oneliner'],seconds=2)]
                if any(STEP.fullmatch(row.get('StepId','')) and row.get('NodeList') == NODE
                       and row.get('Name',row.get('JobName')) == LABEL
                       and row.get('State') == 'RUNNING' and scheduler['tres'](
                           row.get('TRES', row.get('AllocTRES',''))) for row in rows):
                    handshake = 'EXACT_STEP_VISIBLE'
                    break
            except (OSError, ValueError, subprocess.SubprocessError):
                break
            time.sleep(.15)
        shared['write'](RUN/'handoff.json', pack({'request': REQUEST, 'state': handshake}))
        shared['sync'](RUN)
    outcome = {'request': REQUEST, 'intent': intent_pin, 'passed': False, 'handshake': handshake,
               'planned_common_pairs': 20, 'planned_scoring_slots': 40, 'replay_allowed': False}
    try:
        try:
            stdout, stderr = proc.communicate(timeout=max(1, epoch+310-time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=5)
            outcome['client_timed_out'] = True
        require(len(stdout) <= 32768 and len(stderr) <= 65536, 'BOUNDED_NATIVE_OUTPUT')
        outcome.update(srun_rc=proc.returncode, stdout=shared['write'](RUN/'stdout.private',stdout),
                       stderr=shared['write'](RUN/'stderr.private',stderr))
        while time.time() <= intent['deadline']:
            try:
                row = terminal(scheduler['accounting'](),intent,scheduler)
                break
            except (OSError, ValueError, subprocess.SubprocessError):
                time.sleep(.3)
        else:
            raise ValueError('TERMINAL_ACCOUNTING_UNOBSERVED')
        outcome['accounting'] = row
        summary = native_result(stdout, row[0])
        outcome['qualified_input_pairs'] = summary['qualified_input_pairs']
        outcome['passed'] = proc.returncode == 0 and row[2:4] == ['COMPLETED','0:0'] and summary['passed']
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        outcome.update(error_type=type(error).__name__, error_SHA256=pin(str(error).encode())['SHA256'])
    receipt = shared['write'](RUN/'receipt.json',pack(outcome)); shared['sync'](RUN)
    return dict(outcome,receipt=receipt)


def observe(data):
    require(data['scope_request'] == REQUEST, 'INDEPENDENT_SCOPE_REQUIRED')
    scheduler = admissions(False)
    raw = read(RUN/'intent.json',4096,data.get('intent_SHA256'))
    intent = json.loads(raw)
    require(intent['request'] == REQUEST and intent['argv'] == pin(pack(command()))
            and intent['publication'] == data['publication'] and intent['predecessor'] == OLD_ROW[0]
            and intent['deadline'] - intent['epoch'] == 325
            and pin(read(RUN/'carrier.py',32768,data['program_SHA256'])) == intent['program'], 'EXACT_NEW_LAUNCH_INTENT')
    row = terminal(scheduler['accounting'](),intent,scheduler)
    numbered = [line.split('|') for line in scheduler['query'](
        ['sacct','-nP','-j',row[0],'-o',scheduler['COLUMNS']]) if line]
    require(numbered == [row], 'EXACT_NUMERIC_TERMINAL_INDEPENDENT_ACCOUNTING')
    result = {'request': READ_REQUEST, 'scope_request': REQUEST, 'terminal_verified': True, 'accounting': row,
              'intent': pin(raw),
              'planned_common_pairs': 20, 'planned_scoring_slots': 40, 'frozen_unavailable_ordinal': 15,
              'qualified_input_pairs': None, 'actual_GPU': 0, 'passed': False}
    try:
        receipt = json.loads(read(RUN/'receipt.json',8192,data.get('receipt_SHA256')))
        require(receipt['request'] == REQUEST and receipt['intent'] == pin(raw), 'ORIGINAL_NEW_RESULT_INTENT')
        stdout = read(RUN/'stdout.private',32768,receipt['stdout']['SHA256'])
        stderr = read(RUN/'stderr.private',65536,receipt['stderr']['SHA256'])
        require(receipt['stdout'] == pin(stdout) and receipt['stderr'] == pin(stderr), 'EXACT_OUTPUT_PINS')
        summary = native_result(stdout, row[0])
        result.update(qualified_input_pairs=summary['qualified_input_pairs'],
                      failed_eligible=summary['failed_eligible'], native=summary,
                      passed=summary['passed'] and receipt['srun_rc'] == 0 and row[2:4] == ['COMPLETED','0:0'])
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(native_evidence_error=type(error).__name__,
                      native_evidence_error_SHA256=pin(str(error).encode())['SHA256'])
    return result


def check():
    scheduler = admissions(True)
    scheduler['fresh']()
    require(not RUN.exists(), 'NEW_RUN_PATH_EXISTS')
    return {'request': CHECK_REQUEST, 'scope_request': REQUEST, 'passed': True, 'applied': False,
            'verified_assets': 79, 'planned_common_pairs': 20, 'frozen_unavailable_ordinal': 15,
            'old_terminal': OLD_ROW[0], 'actual_GPU': 0, 'argv': pin(pack(command()))}


def main():
    require(os.getuid() == 154459, 'EXACT_PROJECT_UID')
    os.umask(0o077)
    if sys.argv[1:] == ['worker']:
        shared = source_api()
        shared['LABEL'] = LABEL
        shared['worker'](shared['baseline']())
        return
    data = json.load(sys.stdin)
    request = data.get('request')
    try:
        mode = data['mode']
        require(request == {'launch': REQUEST, 'observe': READ_REQUEST, 'check': CHECK_REQUEST}[mode],
                'EXACT_TRANSPORT_REQUEST')
        response = {'launch': launch, 'observe': observe, 'check': lambda _: check()}[mode](data)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        response = {'request': request, 'passed': False, 'replay_allowed': False,
                    'planned_common_pairs': 20, 'planned_scoring_slots': 40, 'error_type': type(error).__name__,
                    'error_SHA256': pin(str(error).encode())['SHA256']}
    print(json.dumps(response,sort_keys=True),flush=True)


if __name__ == '__main__':
    main()
