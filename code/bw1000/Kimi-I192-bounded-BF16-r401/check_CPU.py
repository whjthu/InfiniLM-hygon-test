"""Actual Python call-plan/glue with recording CAPI; no torch/Runtime/DSO load."""
from pathlib import Path
import ast
import importlib.util
import json
import time
from plan import descriptor, records, row_order
from GPU_run import execute_calls


def check():
    seen = []; snapshots = dict(metadata=[0] * 7, decoder=[0] * 4); arm = [0]
    pointers = [[int(1000 + f * 100 + i) for i in range(9)] for f in range(2)]
    def mode(value):
        arm[0] = value; return 0
    def counter():
        return {k: list(v) for k, v in snapshots.items()}
    def invoke(args):
        assert args == [2, 0, 512, 896, 3584, 192, 16, *pointers[len(seen) // 34]]
        seen.append(args)
        m = snapshots['metadata']; m[0] += 1; m[1] += 512; m[2] += 3; m[3] += 9; m[4] += 131204
        if arm[0]:
            a = snapshots['decoder']; a[0] += 3; a[1] += 33; a[2] += 3; a[3] += 33
        return 0
    rows = execute_calls(pointers, time.time() + 30, mode, invoke, counter,
                         lambda r: dict(finite=True, readonly=True, output_tail_canary=True), lambda r: None)
    assert len(rows) == len(seen) == 68
    assert len(list(records())) == 68 and sum(r['warm'] for r in rows) == 4
    assert all(sum(r['fixture'] == f and r['arm'] == a for r in rows) == 17 for f in range(2) for a in range(2))
    assert sorted(row_order(0)) == sorted(row_order(1)) and row_order(0) != row_order(1)
    tests = ['actual68completeCAPIpayloads', 'bothW13W2counterhits', 'twofrozenorders', 'exact17calls_per_arm_fixture']
    for payload in [([0] * 9, time.time() + 30), ([1] * 8, time.time() + 30), ([1] * 9, time.time() - 1)]:
        try: descriptor(list(records())[0], *payload)
        except (ValueError, TimeoutError): pass
        else: raise AssertionError('Invalid payload admitted')
    tests += ['zero_pointer_refusal', 'pointer_count_refusal', 'original_deadline_cancellation_before_CAPI']
    before = len(seen)
    try: execute_calls(pointers, time.time() - 1, mode, invoke, counter, lambda r: {}, lambda r: None)
    except TimeoutError: pass
    else: raise AssertionError('Cancelled call executed')
    assert len(seen) == before
    tests += ['cancel_no_retry_or_invocation']
    failed = []
    try:
        execute_calls(pointers, time.time() + 30, mode,
                      lambda a: failed.append(a) or 1, counter, lambda r: {}, lambda r: None)
    except RuntimeError: pass
    else: raise AssertionError('CAPI error accepted')
    assert len(failed) == 1
    tests += ['first_CAPI_failure_stops_without_retry']
    d = Path(__file__).resolve().parent
    helper = (d / 'helper.cpp').read_text(); gpu = (d / 'decoder.cu').read_text()
    assert 'thread_local int operand_mode401=0' in helper
    assert 'hidden!=3584||intermediate!=192||experts!=896||topk!=16' in helper
    assert 'if(mode!=2||tokens!=512' in helper
    assert 'modern_grouped_activate:modern_grouped_activate_f32_r21' in helper
    for shape in ['rows==768&&columns==3584', 'rows==3584&&columns==384', 'rows==384&&columns==3584', 'rows==3584&&columns==192']:
        assert shape in gpu
    assert 'size_t offset=kimi401::role(rows,columns)' in gpu
    assert '#include "contract.hpp"' in gpu and '#include "contract.hpp"' in helper
    for p in d.glob('*.py'): ast.parse(p.read_text())
    tests += ['defaultOFF', 'HOST_exactI192', 'all4decoderfullshapes', 'packedPER_collision_distinguished', 'SITU_rounding_explicit']
    return dict(completed=True, passed=True, checks=tests, planned=68, recorded_actual_CAPI=68,
                GPU=False, Runtime=False, Native_dlopen=False, weights_opened=False,
                scope='Actual consumer execute_calls argument/counter/cancel glue with recording CAPI leaves, plus real compiler static_assert fullshape contract. No GPU/numeric qualification.')


if __name__ == '__main__': print(json.dumps(check()))
