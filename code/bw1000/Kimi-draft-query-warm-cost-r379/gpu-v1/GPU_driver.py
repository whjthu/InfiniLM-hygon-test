"""Bounded four-process OFF/ON/ON/OFF original-query priming diagnostic."""
import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

from warm_plan import process_plan


def supervise(run_child, save=lambda value: None):
    result = dict(completed=False, passed=False, children=[],
                  four_fresh_processes_planned=True, planned_draft_loads=4,
                  planned_target_loads=0, no_retry_reload_after_failure=True,
                  scope='First-query latency vs explicit priming, not a combined cache/Model speedup')
    save(result)
    for item in process_plan():
        try:
            child = run_child(item)
        except BaseException as error:
            result.update(error=repr(error), stopped_child_index=item['index'])
            save(result)
            raise
        result['children'].append(child)
        save(result)
        assert child['completed'] and child['passed'], 'Stop first child failure; no replay or remaining child loads'
        assert child['child_index'] == item['index'] and child['warm_query'] == item['warm_query']
        assert child['normal_session_close'] and child['normal_Runtime_close']
        plan = child['query_plan']
        assert plan['protocol_commands'] == (137 if item['warm_query'] else 69)
        assert plan['priming_query_count'] == int(item['warm_query']) and plan['measured_query_count'] == 1
        assert [row['role'] for row in plan['phases']] == (['priming', 'measured'] if item['warm_query'] else ['measured'])
    children = result['children']
    assert len({child['PID'] for child in children}) == 4
    assert len({child['PCI'] for child in children}) == 1
    fixture = children[0]['feature_fixture']['measured']['SHA256']
    assert all(child['feature_fixture']['measured']['SHA256'] == fixture for child in children)
    measured = [next(phase for phase in child['query_plan']['phases'] if phase['role'] == 'measured')
                for child in children]
    off = sum(measured[i]['complete_dispatch_wall_s'] for i in [0, 3])/2
    on = sum(measured[i]['complete_dispatch_wall_s'] for i in [1, 2])/2
    assert off > 0 and on > 0
    result.update(completed=True, passed=True,
        measured_query=dict(OFF_s=off, ON_s=on, ON_over_OFF=on/off,
                            OFF_endpoint_ratio=measured[3]['complete_dispatch_wall_s']/measured[0]['complete_dispatch_wall_s'],
                            ON_endpoint_ratio=measured[2]['complete_dispatch_wall_s']/measured[1]['complete_dispatch_wall_s']),
        measured_proposals=[phase['proposals'] for phase in measured],
        first_query_cost_recoverable='Only this four-process diagnostic; target-overlap benefit UNKNOWN',
        priming_extra_setup_and_queries_fully_retained=True,
        all_child_wall_sum_s=sum(child['whole_child_wall_s'] for child in children),
        no_Model_or_acceptance_qualification=True,
        no_priming_cost_removed_from_startup=True,
        physical_GPU_cold='UNKNOWN', raw_activations_published=False)
    save(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--deadline-unix', required=True, type=float)
    args = parser.parse_args()
    directory = pathlib.Path(__file__).resolve().parent
    root = pathlib.Path('/refactor')
    sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    assert os.environ.get('KIMI_WARM_COST_R379') == '1'
    assert 30 < args.deadline_unix-time.time() <= 600
    assert sha(directory/'GPU_MANIFEST.json') == args.manifest_sha
    manifest = json.loads((directory/'GPU_MANIFEST.json').read_bytes())
    assert manifest['GPU_GO'] and manifest['planned_draft_loads'] == 4 and manifest['target_cards'] == 0
    assert all(sha(directory/name) == value for name, value in manifest['files_SHA256'].items())
    step = os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
    folder = root/'results'/('Kimi379-four-process-'+step)
    folder.mkdir(exist_ok=False)
    report_path = folder/'result.json'

    def save(result):
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(report_path)

    def run_child(item):
        remaining = int(args.deadline_unix-time.time())-15
        assert remaining > 20, 'No time for another planned child; no bound extension'
        output = folder/('child-'+str(item['index'])+'.json')
        argv = [sys.executable, '-B', str(directory/'child_driver.py'),
                '--manifest-sha', args.manifest_sha, '--deadline-unix', str(args.deadline_unix),
                '--index', str(item['index']), '--warm', str(int(item['warm_query'])), '--output', str(output)]
        with (folder/('child-'+str(item['index'])+'.log')).open('x') as log:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                raise TimeoutError('Original planned four-process deadline, no retry')
        if not output.exists():
            return dict(completed=True, passed=False, child_index=item['index'],
                        warm_query=item['warm_query'], PID=process.pid, child_returncode=code,
                        error='Child produced no report, no replay')
        child = json.loads(output.read_bytes())
        child.update(child_returncode=code, original_child_report_SHA256=sha(output))
        if code != 0: child['passed'] = False
        return child

    started = time.perf_counter()
    result = {'completed': False, 'passed': False}
    try:
        result = supervise(run_child, save)
        return 0
    finally:
        # A failed supervise saves partial children before propagating. Preserve it.
        if report_path.exists(): result = json.loads(report_path.read_bytes())
        result.update(completed=True, whole_GPU_driver_wall_s=time.perf_counter()-started,
                      actual_step=step, GPU_cards=1, target_cards=0,
                      source_manifest_SHA256=args.manifest_sha)
        save(result)


if __name__ == '__main__':
    raise SystemExit(main())
