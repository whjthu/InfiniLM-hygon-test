"""One bounded actual prepared encoding + whole worker/client CPU closure."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

D = pathlib.Path(__file__).resolve().parent
R = pathlib.Path('/refactor')
step = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
O = R / 'candidates/Deep-paired-quality-r375' / ('CPU-' + step)
O.parent.mkdir(exist_ok=True)
O.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q = dict(completed=False, passed=False, CPU_only=True, GPU=0, Model=0, Native_dlopen=False,
         weights_opened=False, actual_CPU=step, phases={})
start = time.monotonic()
try:
    mfpath = D / 'CPU_SOURCE_MANIFEST.json'
    assert sha(mfpath) == os.environ['DEEP_R375_CPU_SOURCE_SHA']
    mf = json.loads(mfpath.read_bytes())
    for name, expected in mf['files_SHA256'].items():
        assert sha(D / name) == expected, name
    owner = D / 'owner/SOURCE_MANIFEST.json'
    q.update(CPU_source_manifest_SHA256=sha(mfpath), owner_manifest_SHA256=sha(owner))
    os.environ['DEEP_SOURCE_MANIFEST_SHA'] = sha(owner)
    os.environ['DEEP_R375_REPLICA'] = '0'
    encoded = O / 'encoded'
    env = dict(os.environ)
    for name, script, output in [('encoding', 'CPU_encode.py', encoded), ('whole_entry', 'CPU_entry.py', O / 'entry')]:
        if name == 'whole_entry':
            p = encoded / 'encoding.json'
            proof = json.loads(p.read_bytes())
            assert proof['completed'] and proof['passed'] and len(proof['cases']) == 129
            env['DEEP_R375_ENCODING_PATH'] = str(p)
            env['DEEP_R375_ENCODING_SHA'] = sha(p)
            q.update(encoding_receipt_path=str(p), encoding_receipt_SHA256=sha(p))
        log = O / (name + '.log')
        left = int(840 - (time.monotonic() - start))
        assert left > 60
        with log.open('x') as handle:
            result = subprocess.run([sys.executable, '-B', str(D / script), str(output)],
                                    env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=left)
        receipt = output / ('encoding.json' if name == 'encoding' else 'CPU-entry.json')
        q['phases'][name] = {'returncode': result.returncode, 'log_path': str(log), 'log_SHA256': sha(log),
                              'receipt_path': str(receipt), 'receipt_SHA256': sha(receipt) if receipt.exists() else None}
        assert result.returncode == 0
        value = json.loads(receipt.read_bytes())
        assert value['completed'] and value['passed']
        if name == 'whole_entry':
            seams = [x for x in value['cases'] if x['kind'] == 'actual_worker_report_to_actual_client_main66']
            assert {x['replica'] for x in seams} == set(range(4)) and all(x['completed_HTTP'] == 66 for x in seams)
            assert any(x['kind'] == 'actual_client_main_recorded4096cap_continues66' for x in value['cases'])
            q['whole_entry_cases'] = value['cases']
    q['passed'] = True
except BaseException as error:
    q.update(error=repr(error), traceback=traceback.format_exc())
finally:
    q.update(completed=True, seconds=time.monotonic() - start)
    (O / 'CPU-result.json').write_text(json.dumps(q, indent=2) + '\n')
print(json.dumps(q))
raise SystemExit(0 if q['passed'] else 1)
