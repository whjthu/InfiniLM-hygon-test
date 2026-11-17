"""Audit the original v2 HOST failure; never launch, load, or repair a run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

LABEL = 'training-qwen-treasury-explicit-dev3-gpu-v2'
RUN = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results') / LABEL
REQUEST = '581bb322b5bd4d989f7c29e0a4a15eb5'
COLUMNS = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
ARMS = {'base': ('1087656.340', 'a01r4n14'),
        'old_domain': ('1087656.339', 'a01r4n16'),
        'domain': ('1103606.168', 'f13r1n10')}
PINS = {'intent.json': '1923185dc9ddb5180ac8004e748ab51912ba82ce5e9841058d36daf743078e5b',
        'outcome.json': '434ddd3f6a9229afca6fe9951b13cef766857c30fe18a9c077fde3c95a6880b2',
        'source/arm_controller.py': 'f961cb742ca0ea8b4ece43cf89df08d7b2d884d2dde817deb4aee75537240102',
        'source/CLAIM.json': '65f7145c308064f9c7e10b53467e19ba057b5053cb367d71ca7fe75f0c2314ea',
        'source/SOURCE_FILES.json': 'f78e6501bbd523f08df5dfbf7d8a59f636a86011558aa9ced0c4fea10b3c0bab'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(output):
    evidence = {}

    def original(name, expected=None):
        p = RUN / name
        require(p.is_file() and not p.is_symlink(), 'missing or linked original ' + name)
        b = p.read_bytes()
        h = hashlib.sha256(b).hexdigest()
        require(expected is None or h == expected, 'original changed ' + name)
        evidence[name] = {'bytes': len(b), 'SHA256': h}
        return b

    pinned = {n: original(n, h) for n, h in PINS.items()}
    intent, outcome = (json.loads(pinned[n]) for n in ('intent.json', 'outcome.json'))
    require(intent['request'] == outcome['request'] == REQUEST and outcome['passed'] is False,
            'different original request or outcome')
    require(intent['deadline'] == intent['epoch'] + 1800 == outcome['deadline'], 'original epoch changed')
    rows = []
    for parent in ('1087656', '1103606'):
        b = subprocess.check_output(['sacct', '-n', '-P', '-j', parent, '-o', COLUMNS], timeout=12)
        rows += [x.split('|') for x in b.decode().splitlines() if LABEL in x]
    require(len(rows) == 3 and all(len(x) == 7 for x in rows), 'unexpected execution denominator')
    arms = {}
    for arm, (step, node) in ARMS.items():
        prefix = 'arms/' + arm + '/'
        row = [x for x in rows if x[0] == step]
        require(len(row) == 1 and row[0][1:4] == [LABEL + '-' + arm + '-pre', 'COMPLETED', '0:0']
                and row[0][5] == node and 'gres/dcu=' not in row[0][6], 'pre accounting changed')
        original_arm = outcome['arms'][arm]
        require(original_arm['rc'] == 1 and original_arm['original_outcome_present'] is False
                and original_arm['passed'] is False and original_arm['planned_rows'] == 108,
                'original HOST failure differs')
        accounting = json.loads(original(prefix + 'pre-accounting.json'))
        require(accounting['terminal_visible'] and accounting['row'] == row[0] and accounting['rc'] == 0,
                'original and independent accounting disagree')
        log = original(prefix + 'host-controller.log',
                       'd3bb2b5b6f4f0b7f3638e244517062e3f8f95b30c156e7c4489bf27b4662b154').decode()
        require('before = read(RUN / "pre.log")' in log and 'json.decoder.JSONDecodeError:' in log,
                'different failure boundary')
        pre = original(prefix + 'pre.log')
        try:
            json.loads(pre)
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError('original combined log unexpectedly valid JSON')
        # Diagnosis only: this deliberately is NOT a future admission parser.
        lines = pre.decode().splitlines()
        warnings = 1 if arm == 'base' else 2
        require(len(lines) == warnings + 1
                and lines[:-1] == ['srun: request too frequently, retrying ......'] * warnings,
                'different combined-stream contamination')
        snap = json.loads(lines[-1])
        require(snap['kind'] == 'training_qwen_device_snapshot' and snap['mode'] == 'host_complete'
                and snap['step'] == step and snap['node'] == node and len(set(snap['pci'])) == 8
                and snap['kfd_holder_count'] == 0 and snap['full_fd_inventory_performed'] is True,
                'different diagnostic snapshot')
        absent = ['before.json', 'work-intent.json', 'gpu-owned-srun.json', 'gpu-accounting.json',
                  'assigned.json', 'provider.json', 'input-entry-gate.json', 'worker-progress.json',
                  'controller-outcome.json', 'outputs']
        require(not any((RUN / prefix / x).exists() for x in absent), 'later stage evidence now exists')
        arms[arm] = {'pre_accounting': row[0], 'HOST_return_code': 1,
                     'planned_predictions': 108, 'missing_predictions': 108,
                     'failure': 'MERGED_SRUN_STDERR_BEFORE_JSON_PARSER',
                     'later_stage_paths_absent': absent}
    result = {'state': 'ORIGINAL_HOST_FAILURE_INDEPENDENTLY_VERIFIED', 'observed_epoch': time.time(),
              'request': REQUEST, 'accounting_columns': COLUMNS, 'arms': arms, 'originals': evidence,
              'new_GPU_steps': 0, 'generated_predictions': 0, 'planned_predictions': 324,
              'missing_predictions': 324, 'Model_constructed': False, 'postflight_required': False,
              'postflight_basis': 'NO_GPU_STEP_OR_WORKER_EVER_STARTED_IN_THIS_EXACT_NAMESPACE',
              'formal_effect': False, 'replay_allowed': False}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'state': result['state'], 'new_GPU_steps': 0, 'missing_predictions': 324,
                      'result_bytes': output.stat().st_size,
                      'result_SHA256': hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args().output)
