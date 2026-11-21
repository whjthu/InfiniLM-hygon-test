"""Target HOST entry for the original zero-GRES Deep CSA TU07 build."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL = 'Deep-CSA-identity-SWA-commit-CPU-v1'
STAGE_SHA = '5e648f40a35042759d2d9beeaf018b3fd27e4107ba688876533141c34b963097'
PARENT, NODE, ZONE = '1087656', 'f12r3n17', 'O01'
PLACEMENT = dict(parent=PARENT, node=NODE, zone=ZONE, cpus=4, memory_MiB=8192, DCU=0)
REQUEST = sha256(f'{LABEL}:{PARENT}:{NODE}:{STAGE_SHA}'.encode()).hexdigest()[:32]
TARGET = ROOT / 'results' / (LABEL + '-controller')
CLAIM = 'coordination/tasks/deep-csa-identity-swa-compile-cpu-v1.json'
HOST = 'tools/control_deep_csa_identity_compile.py'
TEMPLATE = 'tools/control_deep_inverse_rope_compile.py'
MEMBERSHIP = 'tools/read_training_slurm_membership.py'
LEDGER = 'coordination/training/training80-active-node-task-map-v1.json'
ACCOUNTING = 'code/bw1000/training-qwen-usgs-exposure25-resume20-hip-v1/slurm_accounting.py'
BASE = 'code/bw1000/Deep-C4-causal-consumer-r483'
OUTER = 'code/bw1000/Deep-C4-constant-slot-metadata-r482/Native-CPU-v1/CPU_outer.sh'
BUILDER = BASE + '/csa-identity-swa-commit-CPU-v1/prepare_stage.py'
WORKER = BASE + '/csa-identity-swa-commit-CPU-v1/compile_once.py'
PLAN = BASE + '/csa-identity-swa-commit-build-plan-v1/tu07_plan.py'
RENDER = BASE + '/csa-identity-swa-commit-source-v1/render.py'
HOST_FILES = {HOST, TEMPLATE, MEMBERSHIP, LEDGER, ACCOUNTING, OUTER, BUILDER, WORKER, PLAN, RENDER}
ARTIFACTS = {'TU07.o': 'object_SHA256', 'TU07.before.d': 'fresh_M_SHA256', 'TU07.actual.d': 'actual_MD_SHA256',
             'libDeep_csa_identity_swa_commit.so': 'library_SHA256'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, bound, expected=None):
    with path.open('rb') as stream:
        raw = stream.read(bound + 1)
    need(0 < len(raw) <= bound and (expected is None or sha256(raw).hexdigest() == expected),
         'frozen bytes differ: ' + path.name)
    return raw


def scope(repo, claim_sha):
    claim = json.loads(read(repo / CLAIM, 8192, claim_sha))
    expected = dict(label=LABEL, request=REQUEST, placement=PLACEMENT, stage_manifest_SHA256=STAGE_SHA,
                    wall_seconds=300, planned_HOST_TUs=1, planned_GPU_TUs=0,
                    CPU_GO=True, GPU_GO=False, Model_GO=False, replay_allowed=False)
    need(all(type(claim.get(key)) is type(value) and claim[key] == value for key, value in expected.items()),
         'original zero-GRES scope differs')
    need(claim['host_files'].keys() == HOST_FILES, 'unexpected HOST file set')
    for name, digest in claim['host_files'].items():
        read(repo / name, 64 * 1024, digest)
    owner = json.loads((repo / LEDGER).read_bytes())['groups'][ZONE]['allocations'][PARENT]
    need(NODE in owner, 'selected node is outside the frozen zone')
    return claim


def check_stage(path):
    index = path / 'STAGE_FILES.json'
    need(index.resolve(strict=True) == index, 'stage manifest path is not canonical')
    manifest = json.loads(read(index, 8192, STAGE_SHA))['files_SHA256']
    need(len(manifest) == 9, 'staged member count differs')
    for name, pin in manifest.items():
        current = path / name
        need(current.resolve(strict=True) == current and current.is_relative_to(path),
             'staged path is not canonical')
        read(current, 256 * 1024, pin)


def command(repo, epoch, claim_sha):
    overrides = {
        'TUNING_CPU_PARENT': PARENT, 'TUNING_CPU_NODE': NODE, 'TUNING_CPU_EPOCH': str(epoch),
        'TUNING_CPU_DEADLINE': str(epoch + 300), 'TUNING_CPU_CLAIM_SHA': claim_sha,
        'TUNING_CPU_STAGE_SHA': STAGE_SHA, 'WAVE_INSIDE_R2323': LABEL + '/inside.sh',
    }
    return ['env', *(f'{name}={value}' for name, value in overrides.items()),
            'srun', '--jobid=' + PARENT, '--exact', '--immediate=5', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=4', '--mem=8192M', '--gres=none', '--nodelist=' + NODE, '--time=00:05:00',
            '--job-name=' + LABEL, '/bin/bash', str(repo / OUTER), 'deep']


def check_result(output, row, epoch, claim_sha, allocated_dcu):
    if not row or row[2:4] != ['COMPLETED', '0:0']:
        return None
    resources = dict(field.split('=', 1) for field in row[6].split(','))
    need(re.fullmatch(PARENT + r'\.[0-9]+', row[0]) and row[1] == LABEL and row[5] == NODE
         and row[4].isdecimal() and int(row[4]) <= 300 and resources.get('cpu') == '4'
         and resources.get('node') == '1' and resources.get('mem') in ('8G', '8192M')
         and allocated_dcu(row[6]) == 0, 'original zero-GRES accounting differs')
    raw = read(output / 'CPU-result.json', 128 * 1024)
    result = json.loads(raw)
    expected = dict(original_step=row[0], node=NODE, epoch=epoch, deadline=epoch+300, GRES='none',
                    claim_SHA256=claim_sha, stage_manifest_SHA256=STAGE_SHA, completed=True, passed=True,
                    planned_HOST_TUs=1, planned_GPU_TUs=0, compile_invocations=1, link_invocations=1,
                    fresh_M_paths=997, actual_MD_paths=997, reused_objects_verified=49, loader_paths=68,
                    explicit_Native_calls=0, actual_GPU_steps=0, weight_reads=0, device_qualified=False)
    need(all(type(result.get(key)) is type(value) and result[key] == value for key, value in expected.items()),
         'original build receipt is incomplete or has a different scope')
    for name, key in ARTIFACTS.items():
        need(re.fullmatch(r'[0-9a-f]{64}', result[key]), 'original artifact digest missing')
        read(output / name, 32 * 1024**2, result[key])
    return dict(receipt_SHA256=sha256(raw).hexdigest(), object_SHA256=result['object_SHA256'],
                library_SHA256=result['library_SHA256'])


def module(repo, name, pin):
    raw = read(repo / name, 64 * 1024, pin)
    namespace = {'__name__': 'frozen_deep_csa_dependency', '__file__': str(repo / name)}
    exec(compile(raw, str(repo / name), 'exec'), namespace)
    return namespace


def reconcile(template, terminal, claim_sha, verify):
    raw = read(TARGET / 'intent.json', 8192)
    intent = json.loads(raw)
    epoch = intent['epoch']
    need(type(epoch) is float and math.isfinite(epoch) and epoch > 0
         and intent['request'] == REQUEST and intent['parent'] == PARENT and intent['node'] == NODE
         and intent['source_manifest_SHA256'] == STAGE_SHA and intent['claim_SHA256'] == claim_sha
         and intent['deadline'] == epoch + 300 and intent['replay_allowed'] is False,
         'original controller intent differs')
    row = template['accounting'](8)
    result = dict(request=REQUEST, state='UNKNOWN', row=row, intent_SHA256=sha256(raw).hexdigest(),
                  independent_observation=True, replay_allowed=False)
    if terminal(row):
        result['state'] = 'TERMINAL_UNVERIFIED'
        try:
            checked = verify(ROOT / 'candidates' / LABEL, row, epoch)
            if checked is not None:
                result.update(checked, state='COMPILE_VERIFIED')
        except (OSError, ValueError, TypeError, KeyError) as error:
            result['error'] = type(error).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--claim-sha256', required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-stage', action='store_true')
    modes.add_argument('--dispatch-original', action='store_true')
    modes.add_argument('--reconcile-existing', action='store_true')
    args = parser.parse_args()
    claim = scope(REPO, args.claim_sha256)
    pins = claim['host_files']
    stage = ROOT / 'incoming' / LABEL
    if args.prepare_stage:
        if not stage.exists() and not stage.is_symlink():
            subprocess.run([sys.executable, '-I', '-S', '-B', str(REPO / BUILDER), str(stage)],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True, timeout=35)
        check_stage(stage)
        print(json.dumps(dict(request=REQUEST, stage_manifest_SHA256=STAGE_SHA, members=9, actual_new_Slurm_steps=0)),
              flush=True)
        return 0
    template = module(REPO, TEMPLATE, pins[TEMPLATE])
    helper = module(REPO, ACCOUNTING, pins[ACCOUNTING])

    def verify(output, row, epoch):
        return check_result(output, row, epoch, args.claim_sha256, template['allocated_dcu'])

    template.update(PARENT=PARENT, NODE=NODE, ZONE=ZONE, LABEL=LABEL, REQUEST=REQUEST, MANIFEST=STAGE_SHA,
                    command=lambda epoch: command(REPO, epoch, args.claim_sha256), check_result=verify)
    if args.reconcile_existing:
        result = reconcile(template, helper['is_terminal'], args.claim_sha256, verify)
    else:
        check_stage(stage)
        result = template['controller'](TARGET, ROOT / 'candidates' / LABEL, args.claim_sha256, helper['wait_terminal'])
    print(json.dumps({key: result.get(key) for key in ('request', 'state', 'row', 'receipt_SHA256',
          'library_SHA256', 'independent_observation', 'error')}, allow_nan=False), flush=True)
    return int(result['state'] != 'COMPILE_VERIFIED')


if __name__ == '__main__':
    raise SystemExit(main())
