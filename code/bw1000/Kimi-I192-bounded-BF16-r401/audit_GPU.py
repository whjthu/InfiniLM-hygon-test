"""Read original watch/driver JSON only; no Runtime, submission, or device calls."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import sys


def audit(watch, driver):
    assert watch['completed'], 'Original controller still in progress'
    trials = driver.get('trials', [])
    q = dict(completed=True, actual_GPU=watch.get('actual_step'), controller_passed=watch.get('passed', False),
             driver_completed=driver.get('completed', False), driver_passed=driver.get('passed', False),
             attempted=driver.get('attempted'), CAPI_returned=driver.get('CAPI_returned'),
             completed_CAPI=driver.get('completed_calls'), qualification_passed_calls=driver.get('qualification_passed_calls'),
             saved_rows=len(trials), Model_gain=False, full_MoE_qualified=False, Runtime_world=1,
             scope='Single logicalTP16rank0 complete routedFFN only, historical16 expanded512 geometry, noCCL/Model.')
    if not watch.get('passed') or not driver.get('passed'):
        q.update(state='FAILED_OR_PARTIAL_STOP_NO_REPLAY', watch_error=watch.get('error'), driver_error=driver.get('error'),
                 last_saved_completed_call=trials[-1] if trials else None,
                 independent_post=watch.get('failure_post', watch.get('post')))
        return q
    assert watch['Slurm_COMPLETED0'] and watch['srun_exit0'] and watch['normal1']
    assert driver['normal_Runtime_close'] and driver['Torch_after_Runtime_close']
    assert driver['GPU_count'] == driver['Runtime_world'] == 1 and not driver['CCL'] and not driver['Model_loaded']
    assert driver['all_checkpoint_banks_readonly']
    assert len(trials) == driver['attempted'] == driver['CAPI_returned'] == driver['completed_calls'] == driver['qualification_passed_calls'] == 68
    for post in watch['post']:
        assert post['passed'] and not post['KFD_pids'] and not post['matching_GPU_driver_PIDs'] and not post['expected_PIDs_still_exist']
    q['fixtures'] = []
    for fixture in range(2):
        rows = [v for v in trials if v['fixture'] == fixture]
        assert len(rows) == 34 and [v['arm'] for v in rows[:2]] == [0, 1]
        blocks = []
        for block in range(8):
            group = [v for v in rows if not v['warm'] and v['block'] == block]
            assert [v['arm'] for v in group] == [0, 1, 1, 0]
            off = (group[0]['complete_call_wall_ms'] + group[3]['complete_call_wall_ms']) / 2
            on = (group[1]['complete_call_wall_ms'] + group[2]['complete_call_wall_ms']) / 2
            assert math.isfinite(off) and math.isfinite(on) and off > 0 and on > 0
            blocks.append(dict(block=block, OFF_ms=off, ON_ms=on, ON_over_OFF=on / off, rate_gain=off / on - 1))
        formal = [v for v in rows if not v['warm']]
        drift = formal[-1]['complete_call_wall_ms'] / formal[0]['complete_call_wall_ms'] - 1
        gain = statistics.median(v['rate_gain'] for v in blocks)
        for row in rows:
            assert row['CAPI_completed'] and row['CAPI_returncode'] == 0 and row['passed']
            assert row['counter_pass'] and row['finite'] and row['numeric_pass'] and row['readonly'] and row['output_tail_canary']
            assert math.isfinite(row['relative_RMS']) and row['relative_RMS'] <= 1 / 64
            assert math.isfinite(row['maxabs_over_reference_max']) and row['maxabs_over_reference_max'] <= 1 / 8
        q['fixtures'].append(dict(fixture=fixture, blocks=blocks, OFF_endpoint_drift=drift, median_paired_rate_gain=gain,
                                  positive_blocks=sum(v['rate_gain'] > 0 for v in blocks),
                                  performance_gate_pass=gain > 0 and gain > abs(drift),
                                  max_relative_RMS=max(v['relative_RMS'] for v in rows),
                                  max_abs=max(v['max_abs'] for v in rows),
                                  maxabs_over_reference_max=max(v['maxabs_over_reference_max'] for v in rows),
                                  warm_ms=[v['complete_call_wall_ms'] for v in rows[:2]],
                                  first_OFF_metadata=rows[0]['counters_delta'], first_ON_metadata=rows[1]['counters_delta']))
    q.update(numeric_state_pass=True, normal_Runtime_close=True, independent_post_clean=True,
             performance_gate_pass=all(v['performance_gate_pass'] for v in q['fixtures']))
    q['state'] = 'ROUTED_COMPONENT_PASS_NEXT_FULL_MOE_PROPOSAL_ONLY' if q['performance_gate_pass'] else 'NUMERIC_PASS_PERFORMANCE_GATE_FAIL_STOP'
    q['limits'] = ['Only per-call materialization recomputed, not an HBM-cold/cache-flushed guarantee.',
                   'No raw-output tensor archive; numeric statistics are the original consumer result, not an independent raw-tensor recomputation.',
                   'Logical rank0 ofTP16 is not actual16rank communication/allMoE or Model quality.',
                   'No scaling historical component gains by layers or wholeCAPI time.']
    return q


if __name__ == '__main__':
    watch_path, driver_path = map(Path, sys.argv[1:3])
    q = audit(json.loads(watch_path.read_bytes()), json.loads(driver_path.read_bytes()))
    q['raw_SHA256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [watch_path, driver_path]}
    print(json.dumps(q, indent=2, allow_nan=False))
