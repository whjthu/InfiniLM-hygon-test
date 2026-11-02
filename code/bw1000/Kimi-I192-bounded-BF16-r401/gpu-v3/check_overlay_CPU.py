"""Exercise actual changed consumer with real report files; recording CAPI only."""
from pathlib import Path
import json
import tempfile
import time
from GPU_run import execute_calls


def case(kind):
    count = [0]; mode = [0]
    counters = dict(metadata=[0] * 7, decoder=[0] * 4)
    def set_mode(a): mode[0] = a; return 0
    def counter(): return {k: list(v) for k, v in counters.items()}
    def invoke(args):
        assert args[:7] == [2, 0, 512, 896, 3584, 192, 16]
        count[0] += 1
        if kind != 'bad_counter':
            m = counters['metadata']; m[0] += 1; m[1] += 512; m[2] += 3; m[3] += 9
        if mode[0]:
            v = counters['decoder']; v[0] += 3; v[1] += 33; v[2] += 3; v[3] += 33
        return 0
    def inspect(r):
        bad = kind in ['bad_numeric', 'nonfinite', 'primary_plus_save_error']
        return dict(finite=kind != 'nonfinite', relative_RMS=None if kind == 'nonfinite' else .1 if bad else 0,
                    max_abs=None if kind == 'nonfinite' else 1, maxabs_over_reference_max=None if kind == 'nonfinite' else .2 if bad else 0,
                    numeric_pass=not bad, readonly=True, output_tail_canary=True)
    with tempfile.TemporaryDirectory() as t:
        path = Path(t) / 'actual-report.json'
        def observe(rows):
            path.write_text(json.dumps(dict(completed_CAPI=count[0], qualification_passed=sum(v['passed'] for v in rows),
                                           trials=rows), allow_nan=False))
            if kind == 'primary_plus_save_error': raise OSError('recording later I/O error')
        error = None
        try: rows = execute_calls([[1] * 9, [2] * 9], time.time() + 30, set_mode, invoke, counter, inspect, observe)
        except RuntimeError as e: error = e
        saved = json.loads(path.read_bytes())
        if kind == 'success':
            assert error is None and len(rows) == saved['qualification_passed'] == saved['completed_CAPI'] == 68
        else:
            assert type(error) is RuntimeError and 'Completed call failed qualification' in str(error)
            assert count[0] == saved['completed_CAPI'] == len(saved['trials']) == 1
            assert saved['qualification_passed'] == 0 and not saved['trials'][0]['passed']
            assert saved['trials'][0]['CAPI_completed'] is True
            if kind == 'bad_counter': assert not saved['trials'][0]['counter_pass']
            if kind == 'nonfinite': assert saved['trials'][0]['relative_RMS'] is None and not saved['trials'][0]['finite']
            if kind == 'primary_plus_save_error': assert isinstance(error.__cause__, OSError)
    return dict(case=kind, passed=True)


if __name__ == '__main__':
    cases = [case(k) for k in ['success', 'bad_numeric', 'bad_counter', 'nonfinite', 'primary_plus_save_error']]
    print(json.dumps(dict(completed=True, passed=True, cases=cases, GPU=False, Runtime=False, Slurm=False,
                          scope='Actual Python overlay execute_calls, recording CAPI leaves, real tempfile/JSON. Original CPU801 C++/objects/MD unchanged.')))
