"""Prepare the exact Qwen USGS training admission after its live read-only check."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

from preflight_qwen_usgs_exposure25 import CLAIM_SHA, DEV_NODES, LABEL, MANIFEST_SHA, NODE, PARENT, PORT, REQUEST, SOURCE
from prepare_training80_vggt_admission import as_json, exact_value, published, write_once
from record_qwen_usgs_dev80_absence import bounded, read_record, record_once, validate as validate_dev80
from report_training80_backlog import BACKLOG, REPO, has_original_intent, hex_value, invocation, need, verify_launch


BASE_PUBLICATION = '1b01e9faed7134f3198cb4b6270b0ed8b3a8bd9d'
CANDIDATE = 'qwen-usgs-previous-data-exposure25-resume20'
CASE = 'T01.domain-usgs-exposure25-resume20'
LOCAL = f'local-only/{LABEL}-original-dispatch'
MARKERS = [f'{LOCAL}/intent.json', f'{LOCAL}/attempt.json']
ADOPTION_STATE = 'ROOT_ADOPTED_ONE_TRAINING_REQUEST_DEVICE_PENDING'


def original_blob(repo, name):
    return subprocess.check_output(['git', '--no-optional-locks', '-c', 'safe.directory='+repo.as_posix(),
                                    'cat-file', 'blob', f'{BASE_PUBLICATION}:{SOURCE.as_posix()}/{name}'],
                                   cwd=repo, timeout=10, stdin=subprocess.DEVNULL)


def adoption(repo, *, read_original=original_blob):
    manifest_raw = read_original(repo, 'SOURCE_FILES.json')
    claim_raw = read_original(repo, 'CLAIM.json')
    need(sha256(manifest_raw).hexdigest() == MANIFEST_SHA and sha256(claim_raw).hexdigest() == CLAIM_SHA,
         'original source bytes differ from read-only preflight')
    manifest, claim = json.loads(manifest_raw), json.loads(claim_raw)
    exact_value(claim, dict(request=REQUEST, parent=PARENT, node=NODE, master_port=PORT, device_count=8,
                            optimizer_updates=30, updates_by_arm={'continuous': 25, 'resumed': 5},
                            root_GO=False, GPU_GO=False, training_GO=False, formal_training_GO=False,
                            effect_GO=False, performance_GO=False, Model_GO=False, replay_allowed=False),
                'original claim is not the source-only 25+5 training scope')
    need(len(manifest['files']) == 21 and manifest['files']['CLAIM.json'] ==
         dict(bytes=len(claim_raw), SHA256=CLAIM_SHA), 'original package inventory differs')
    changed_claim = as_json(dict(claim, state=ADOPTION_STATE, root_GO=True, GPU_GO=True, training_GO=True))
    manifest['files']['CLAIM.json'] = dict(bytes=len(changed_claim), SHA256=sha256(changed_claim).hexdigest())
    return changed_claim, as_json(manifest), claim, manifest


def target_checks(repo, raw):
    proof = validate_dev80(repo, raw)
    value = json.loads(raw)
    exact_value(value, dict(request=REQUEST, node=NODE, parent=PARENT, required_port=PORT,
                            state='TRAINING_ASSETS_VERIFIED_DEVICE_PENDING', read_only=True,
                            source_manifest_SHA256=MANIFEST_SHA, claim_SHA256=CLAIM_SHA,
                            actual_slurm_submissions=0, device_admission=False, training_READY=False),
                'original read-only Qwen training preflight is absent or out of scope')
    for name in ('first', 'last'):
        item = value[name]
        exact_value(item, dict(current_namespace_absent=True, current_rows=[], predecessor_namespace_absent=True,
                               predecessor_rows={}, clear=True), 'Qwen training or Dev80 original request requires reconciliation')
        need(hex_value(item.get('SHA256'), 64), 'missing original Slurm accounting digest')
    exact_value(value['assets'], dict(framework_files=393, train_rows=88, train_media_files=22,
                                     model_shard_metadata=4, small_model_files=10, model_weight_body_bytes_read=0,
                                     initial_state_metadata=True, initial_state_body_bytes_read=0),
                'read-only Qwen training or framework assets differ')
    seen, returned = value['membership']['observed_epoch'], value['returned_epoch']
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (seen, returned, proof['observed_epoch'])) and
         0 < proof['observed_epoch'] <= seen <= returned <= proof['observed_epoch']+90 and
         value['membership']['node_membership'] == dict.fromkeys(DEV_NODES.values(), 'NO_NUMBERED_STEP') and
         type(value['assets']['CPU_input_log_bytes']) is int and value['assets']['CPU_input_log_bytes'] > 0,
         'original prerequisites or bounded observed window differs')


def new_receipts(repo, output):
    output = output.resolve()
    need(output.is_relative_to(repo/'local-only') and output != repo/'local-only' and
         not has_original_intent(repo, MARKERS), 'scope outside private receipts or original launch already attempted')
    return output


def prepare_adoption(repo, raw, output, *, read_original=original_blob):
    output = new_receipts(repo, output)
    target_checks(repo, raw)
    claim, manifest, _, _ = adoption(repo, read_original=read_original)
    output.mkdir()
    return dict(original_preflight=write_once(repo, output/'target-preflight.json', raw),
                claim=write_once(repo, output/'CLAIM.json', claim),
                manifest=write_once(repo, output/'SOURCE_FILES.json', manifest),
                actual_slurm_submissions=0, actual_training_ready=0)


def prepare(repo, backlog, raw, output, publication, *, check_publication=published, read_original=original_blob):
    output = new_receipts(repo, output)
    target_checks(repo, raw)
    need(hex_value(publication, 40), 'invalid adopted source publication identity')
    [candidate] = [row for row in backlog['items'] if row['id'] == CANDIDATE]
    exact_value(candidate, dict(case=CASE, kind='training', state='blocked', placement={NODE: 8}, min_updates=30),
                'base backlog no longer refers to the original Qwen candidate')
    claim, manifest, original, inventory = adoption(repo, read_original=read_original)
    paths = dict(claim=(SOURCE/'CLAIM.json').as_posix(), manifest=(SOURCE/'SOURCE_FILES.json').as_posix(),
                 launcher=(SOURCE/'launch.py').as_posix())
    need((repo/paths['claim']).read_bytes() == claim and (repo/paths['manifest']).read_bytes() == manifest,
         'only the exact adopted claim and manifest may enter the queue')
    expected = {**{(SOURCE/name).as_posix(): pin for name, pin in inventory['files'].items()},
                **{pin['path']: pin for pin in original['public_prerequisites']}}
    for path,pin in expected.items():
        data = (repo/path).read_bytes()
        need(len(data) == pin['bytes'] and sha256(data).hexdigest() == pin['SHA256'], 'adopted input differs: '+path)
    check_publication(repo, publication, sorted({paths['manifest'], *expected}))
    need(not has_original_intent(repo, MARKERS), 'original intent appeared while checking publication')
    if read_record(repo) is None:
        record_once(repo, raw)
    output.mkdir()
    capture = write_once(repo, output/'target-preflight.json', raw)
    pins = dict(claim=dict(path=paths['claim'], SHA256=sha256(claim).hexdigest()),
                manifest=dict(path=paths['manifest'], SHA256=sha256(manifest).hexdigest()),
                launcher=dict(path=paths['launcher'], SHA256=expected[paths['launcher']]['SHA256']))
    workload = sha256(as_json(original)).hexdigest()
    launch = dict(pins, request=REQUEST, publication_commit=publication, workload_SHA256=workload,
                  required_ports=[PORT], authorization_key='training_GO', attempt_receipts=MARKERS)
    receipts = {
        'publication_receipt': dict(state='PUBLISHED_MATCHED', publication_commit=publication,
                                    **{name+'_SHA256': pin['SHA256'] for name,pin in pins.items()}),
        'prerequisite_receipt': dict(state='TARGET_PREREQUISITES_VERIFIED', workload_SHA256=workload,
                                     placement={NODE: 8}, min_updates=30, required_ports=[PORT]),
        'original_request_receipt': dict(state='REMOTE_UNSTARTED_VERIFIED', original_namespace_absent=True,
                                         accounting_label_matches=0, prior_invocation=False),
    }
    for name, receipt in receipts.items():
        launch[name] = write_once(repo, output/(name+'.json'), as_json(dict(receipt, request=REQUEST,
                                                                          target_preflight=capture)))
    ready = dict(candidate, state='ready', launch=launch,
                 next_action='取得同一节点的有效八卡和端口收据，登记原请求后调用一次冻结启动器。')
    need(verify_launch(repo, ready) == 'ready' and invocation(repo, ready) is not None,
         'original training request changed or launcher could not be bound')
    overlay = dict(backlog, items=[ready if row is candidate else row for row in backlog['items']])
    return dict(backlog=write_once(repo, output/'backlog.json', as_json(overlay)), request=REQUEST,
                verified_waiting_for_current_device_receipt=True, actual_slurm_submissions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    for mode in ('adoption-bytes', 'overlay'):
        cmd = commands.add_parser(mode)
        cmd.add_argument('target_preflight', type=Path)
        cmd.add_argument('output', type=Path)
        if mode == 'overlay':
            cmd.add_argument('publication')
            cmd.add_argument('--backlog', type=Path, default=BACKLOG,
                             help='base backlog or a previously verified local overlay')
    args = parser.parse_args()
    raw = bounded(args.target_preflight)
    if args.action == 'adoption-bytes':
        result = prepare_adoption(REPO, raw, args.output)
    else:
        result = prepare(REPO, json.loads(bounded(args.backlog)), raw, args.output, args.publication)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
