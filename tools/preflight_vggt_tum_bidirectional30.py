"""Read the frozen VGGT training assets and Slurm namespace before dispatch."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time

from read_training_slurm_membership import snapshot


REPO = Path(__file__).resolve().parents[1]
RESULTS = Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results')
LABEL = 'training-vggt-tum-bidirectional30-v1'
REQUEST = 'ab8a312813524d949608defa7124c3bf'
PARENT, NODE, ZONE = '1103606', 'f13r4n09', 'T12'
PACKAGE = Path('code/bw1000') / LABEL
LEDGER = Path('coordination/training/training80-active-node-task-map-v1.json')
FRAMEWORK_BASE = Path('training-qwen-a68952c-swift-initial-CPU-v1')
INPUT_BASE = Path('training-vggt-tum-continuous25-recovery-v1')
UPSTREAM = Path('training-vggt-native-provider-HIP-CPU-v1/upstream')
WEIGHT_PATH = Path('training-vggt-original-public-model-v1/model/model.pt')
WEIGHT_STAT = (5026874952, 20701942394, 1794489337000000000)
WEIGHT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
FRAMEWORK_SHA = 'e54fd2031287277d1823a4a703d6dc0540067809d3d0f505c6e081cf4c5cec41'
SOURCE_MANIFEST = '33eca3f033c5e36675d529fbe15ffa1204bbe62c769f1e61803b1c4de12b7456'
ORIGINAL_DATA_INVENTORY = '3c4e3ffb19e73628c359a1c8c3d073e4b6ea8adb619a06692b84813e232b8c4f'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def exact(root, name, pin, size=None):
    path = root / name
    need(path.is_relative_to(root) and path.resolve(strict=True) == path and path.is_file(),
         'original asset path differs: ' + str(name))
    with path.open('rb') as stream:
        raw = stream.read((size if size is not None else 8*1024**2) + 1)
    need((len(raw) == size if size is not None else 0 < len(raw) <= 8*1024**2)
         and sha256(raw).hexdigest() == pin, 'original asset bytes differ: ' + str(name))
    return raw


def check_source(repo):
    package = repo / PACKAGE
    manifest = json.loads(exact(package, 'SOURCE_FILES.json', SOURCE_MANIFEST))['files']
    need(len(manifest) == 15 and 'CLAIM.json' in manifest, 'candidate source inventory differs')
    original_claim = None
    for name, entry in manifest.items():
        raw = exact(package, name, entry['SHA256'], entry['bytes'])
        if name == 'CLAIM.json':
            original_claim = json.loads(raw)
    need(original_claim['request'] == REQUEST and original_claim['node'] == NODE and original_claim['parent'] == PARENT
         and original_claim['planned_updates'] == 30 and original_claim['framework_manifest_SHA256'] == FRAMEWORK_SHA
         and all(original_claim[flag] is False for flag in ('GPU_GO','training_GO','root_GO','formal17_GO','Model_GO')),
         'expected the original source-only training claim')
    sources = original_claim['sources']
    all_raw = {name: exact(repo, entry['path'], entry['sha']) for name, entry in sources.items()}
    need(sources['original_claim.json']['sha'] == ORIGINAL_DATA_INVENTORY, 'original data claim differs')
    prior = json.loads(all_raw['prior_directed_terminal.json'])
    need(prior['actual_updates'] == 45 and prior['recovery_passed'] is True and prior['gpu_step'] == '1103606.479'
         and prior['independent_post'] == '1103606.485', 'prior directed training is outside the frozen dependency')
    return all_raw, len(manifest)


def verify_assets(base, data_claim, upstream_claim, *, model_stat=WEIGHT_STAT):
    model = base / WEIGHT_PATH
    meta = model.lstat()
    need(model.resolve(strict=True) == model and stat.S_ISREG(meta.st_mode)
         and (meta.st_size, meta.st_ino, meta.st_mtime_ns) == model_stat, 'original model file identity differs')
    framework = base / FRAMEWORK_BASE
    members = json.loads(exact(framework / 'source', 'FRAMEWORK_FILES.json', FRAMEWORK_SHA))['files']
    need(len(members) == 393, 'original framework member count differs')
    for name, entry in members.items():
        exact(framework / 'framework', name, entry['SHA256'], entry.get('bytes'))
    data = json.loads(data_claim)['data_files']
    need(len(data) == 17 and sum(entry['bytes'] for entry in data.values()) <= 4*1024**2,
         'original TUM media inventory differs')
    for name, entry in data.items():
        exact(base / INPUT_BASE / 'data', name, entry['sha'], entry['bytes'])
    upstream = json.loads(upstream_claim)['files']
    need(len({entry['name'] for entry in upstream}) == len(upstream) > 0, 'original provider member list differs')
    for entry in upstream:
        exact(base / UPSTREAM, entry['name'], entry['SHA256'], entry['bytes'])
    need((base / 'training-t06-sdxl-runtime-v1/site').is_dir(), 'original target training runtime path is absent')
    return dict(model_stat_verified=True, original_model_pin=WEIGHT_SHA, model_content_bytes_read=0,
                verified_framework_files=len(members), verified_training_media_files=len(data),
                verified_upstream_files=len(upstream), runtime_path_present=True, runtime_imported=False)


def check_namespace(raw, base):
    expected = {LABEL+'-'+phase for phase in ('pre','gpu','post')}
    rows = [line.split('|') for line in raw.decode().splitlines() if line.strip()]
    need(rows and all(len(row) == 7 for row in rows), 'missing or malformed original Slurm accounting')
    need(any(row[0] == PARENT for row in rows), 'accounting did not include the original parent')
    found = [row for row in rows if row[1] in expected]
    namespace = base / LABEL
    absent = not namespace.exists() and not namespace.is_symlink()
    return dict(namespace_absent=absent, accounting_label_matches=len(found),
                discovered_rows=found, original_read_only_guard_passed=absent and not found,
                raw_accounting_SHA256=sha256(raw).hexdigest())


def check_placement(repo):
    ledger = json.loads((repo / LEDGER).read_bytes())
    parents = {job['JobId'] for job in ledger['jobs']}
    groups = {parent: {zone: group['allocations'][parent] for zone, group in ledger['groups'].items()}
              for parent in parents}
    report = snapshot(groups, owner_uid=os.getuid())
    item = report['parents'][PARENT]['coverage_by_group_and_node'][ZONE][NODE]
    need(report['membership_qualified'] and item['state'] == 'NO_NUMBERED_STEP'
         and item['numbered_step_assigned_DCU_bounds'] == [0,0], 'original T12 node has a step or unverified placement')
    return report['observed_epoch']


def live_accounting():
    columns = 'JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList,AllocTRES'
    raw = subprocess.check_output(['sacct','-n','-P','-j',PARENT,'-S','2026-11-20T00:00:00','-o',columns], timeout=10)
    need(len(raw) <= 2*1024**2, 'original accounting exceeds bound')
    return raw


def preflight(repo, base, *, get_accounting=live_accounting, placement=check_placement):
    original = check_namespace(get_accounting(), base)
    if not original['original_read_only_guard_passed']:
        return dict(state='ORIGINAL_REQUIRES_RECONCILIATION', request=REQUEST, original=original)
    source, source_count = check_source(repo)
    assets = verify_assets(base, source['original_claim.json'], source['UPSTREAM_FILES.json'])
    observed = placement(repo)
    final = check_namespace(get_accounting(), base)
    state = 'TARGET_ASSETS_VERIFIED_DEVICE_PENDING' if final['original_read_only_guard_passed'] else 'ORIGINAL_REQUIRES_RECONCILIATION'
    return dict(state=state, request=REQUEST, left_unmodified=True, original=final,
                first_namespace_accounting_SHA256=original['raw_accounting_SHA256'], assets=assets,
                verified_source_members=source_count, parent=PARENT, node=NODE, membership_observed_epoch=observed,
                returned_epoch=time.time(), proposed_updates=30, PCI_KFD_PID_verified=False, training_READY=False)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    def deadline(*_):
        raise TimeoutError('original VGGT read-only preflight budget expired')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 90)
    report = preflight(REPO, RESULTS)
    print(json.dumps(report, allow_nan=False), flush=True)
    return int(report['state'] != 'TARGET_ASSETS_VERIFIED_DEVICE_PENDING')


if __name__ == '__main__':
    raise SystemExit(main())
