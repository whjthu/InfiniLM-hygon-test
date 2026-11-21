"""Execute the frozen GLM fresh K/V Native recipe inside its single CPU step."""

import argparse
import ast
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

import prepare_glm474_fresh_kv_native_build as build


LABEL = 'GLM474-fresh-kv-pack-Native-CPU-v1'
PARENT = '1103606'
NODE = 'f12r2n13'
KERNEL_START = 'b9a7f0b78c566911ed40004fc906ce44'
PLACEMENT = dict(parent=PARENT, node=NODE, zone='O01', cpus=4, memory_MiB=16384, DCU=0)
CLAIM = build.REPO/'coordination/tasks/glm-r474-fresh-kv-pack-native-cpu-v1.json'
ORIGINAL_CLEANUP = build.CALLER.parent/'depth-stratified-observer-CPU-v1/build_once.py'
ORIGINAL_CLEANUP_SHA = '7e786da991f1c5f83d44645d12a8aaa7e8606b2b0e5bdabd6ae1b0b7ca944e46'
RECIPE_TOOL = build.REPO/'tools/prepare_glm474_fresh_kv_native_build.py'
PLANNED = dict(fresh_M=2, HOST_compile=2, GPU_compile=0, reused_objects=30, link_objects=33, links=1)


def phase_runner(recipe, command, verify_deps, verify_artifacts, verify_link, report, persist):
    build.need(recipe['planned'] == PLANNED and set(recipe['host']) == set(build.CHANGED)
               and len(recipe['reuse']) == 30, 'frozen two-unit recipe differs')
    for phase, key, counter, timeout in (('fresh', 'preprocess', 'fresh_M', 80),
                                           ('compile', 'compile', 'HOST_compile', 175)):
        report['phase'] = phase
        persist()
        for unit, row in recipe['host'].items():
            report['attempted'][counter] += 1
            command(f'{phase}-{unit}', row[key], timeout)
            report['actual'][counter] += 1
        report[phase+'_dependency_evidence'] = verify_deps('fresh' if phase == 'fresh' else 'actual')
    report['phase'] = 'recheck-reused-artifacts'
    persist()
    verify_artifacts(recipe)
    report['actual']['reused_objects'] = 30
    report['phase'] = 'link'
    report['attempted']['links'] += 1
    persist()
    command('link', recipe['link'], 120)
    report['actual'].update(links=1, link_objects=33)
    report['phase'] = 'verify-link'
    persist()
    report['linked'] = verify_link()
    build.need(report['actual'] == PLANNED, 'Native build denominators differ')
    report.update(passed=True, phase='complete')


def validate_claim(claim_sha):
    claim = json.loads(build.checked(CLAIM, claim_sha, 16384))
    build.need(claim['label'] == LABEL and claim['placement'] == PLACEMENT and claim['planned'] == PLANNED
               and claim['kernel_prerequisite_request'] == KERNEL_START and claim['original_CPU308'] == '1103606.308'
               and claim['whole_seconds'] == 1800 and claim['work_seconds'] == 1680
               and claim['CPU_GO'] is True and claim['GPU_GO'] is False and claim['Model_GO'] is False
               and claim['replay_allowed'] is False, 'frozen CPU-only Native claim differs')
    build.checked(RECIPE_TOOL, claim['recipe_tool_SHA256'])
    build.checked(Path(__file__), claim['runner_SHA256'])
    return claim


def validate_placement(environment, hostname, available_cpus):
    build.need(environment.get('SLURM_JOB_ID') == PARENT and re.fullmatch(r'\d+', environment.get('SLURM_STEP_ID', ''))
               and hostname == NODE and available_cpus == 4
               and all(environment.get(key) == '-1' for key in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'))
               and not any(environment.get(key) for key in ('SLURM_STEP_GPUS', 'LD_PRELOAD', 'LD_AUDIT')),
               'expected the frozen zero-GRES CPU step')


def owned_cleanup():
    raw = build.checked(ORIGINAL_CLEANUP, ORIGINAL_CLEANUP_SHA)
    [node] = [node for node in ast.parse(raw).body
              if isinstance(node, ast.FunctionDef) and node.name == 'cleanup_owned_command']
    scope = dict(os=os, signal=signal, subprocess=subprocess, Path=Path, need=build.need)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ORIGINAL_CLEANUP), 'exec'), scope)
    return scope['cleanup_owned_command']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim-sha256', required=True)
    parser.add_argument('--kernel-reconciliation', type=Path, required=True)
    parser.add_argument('--kernel-reconciliation-sha256', required=True)
    parser.add_argument('--epoch', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--work-deadline', type=float, required=True)
    args = parser.parse_args()
    now, started = time.time(), time.monotonic()
    build.need(all(math.isfinite(value) for value in (args.epoch, args.deadline, args.work_deadline))
               and args.deadline == args.epoch+1800 and args.work_deadline == args.epoch+1680
               and 0 < args.epoch <= now < args.work_deadline-25, 'expected the original single CPU epoch')
    claim = validate_claim(args.claim_sha256)
    validate_placement(os.environ, socket.gethostname().split('.')[0], len(os.sched_getaffinity(0)))
    cleanup = owned_cleanup()
    output = Path(build.NATIVE)
    build.need(not output.exists() and not output.is_symlink(), 'original Native namespace already exists; reconcile it')
    report = dict(label=LABEL, request=claim['request'], step=PARENT+'.'+os.environ['SLURM_STEP_ID'], node=NODE,
                  epoch=args.epoch, deadline=args.deadline, work_deadline=args.work_deadline,
                  claim_SHA256=args.claim_sha256, actual={key: 0 for key in PLANNED},
                  attempted=dict(fresh_M=0, HOST_compile=0, links=0), planned=PLANNED, commands=[],
                  GPU_GO=False, Model_GO=False, passed=False, phase='prepare', replay_allowed=False)
    initialized = False

    def left(reserve=0):
        remaining = min(args.work_deadline-time.time(), args.work_deadline-now-(time.monotonic()-started)) - reserve
        if remaining <= 0:
            raise TimeoutError('original CPU work deadline exhausted')
        return remaining

    def persist():
        tmp = output/'CPU-result.tmp'
        with tmp.open('w', encoding='utf8') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(output/'CPU-result.json')

    def command(label, argv, limit):
        entry = dict(phase=label, argv=argv, returncode=None, owned_child_reaped=False)
        report['commands'].append(entry)
        proc = None
        with (output/(label+'.out')).open('xb') as stdout, (output/(label+'.err')).open('xb') as stderr:
            try:
                budget = min(limit, left(8))
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
                entry.update(PID=proc.pid, owned_pgrp=proc.pid, startticks=None)
                try:
                    proc_row = Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')', 1)[1].split()
                    entry['startticks'] = int(proc_row[19])
                except FileNotFoundError:
                    pass
                entry['returncode'] = proc.wait(timeout=budget)
            finally:
                cleanup(proc, entry)
                persist()
        build.need(entry['returncode'] == 0 and entry['owned_child_reaped'], 'failed owned build phase: '+label)

    def verify_artifacts(recipe):
        objects = list(recipe['reuse'].values()) + [recipe['kernel']]
        for item in objects:
            left(8)
            build.checked(item['path'], item['SHA256'])

    def expired(*_):
        raise TimeoutError('original CPU work deadline exhausted')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, left(8))
    try:
        prepared = build.prepare(args)
        initialized = True
        report['recipe_SHA256'] = prepared['recipe_SHA256']
        persist()
        recipe = json.loads(build.checked(output/'BUILD_RECIPE.json', prepared['recipe_SHA256'], 2*1024**2))
        phase_runner(recipe, command, lambda phase: build.verify_dependencies(prepared['recipe_SHA256'], phase),
                     verify_artifacts, lambda: build.verify_link(prepared['recipe_SHA256']), report, persist)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        report.update(error_type=type(error).__name__, reason=str(error)[:240])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report['completed'] = True
        if initialized:
            persist()
    print(json.dumps({k: report[k] for k in ('request', 'step', 'phase', 'completed', 'passed', 'actual')}, allow_nan=False))
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
