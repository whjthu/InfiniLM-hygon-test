"""Turn the original VGGT target preflight into a local training80 backlog overlay."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

from preflight_vggt_tum_bidirectional30 import (
    CLAIM_SHA, LABEL, NODE, PACKAGE, PARENT, REQUEST, SOURCE_MANIFEST, WEIGHT_SHA, check_source,
)
from report_training80_backlog import BACKLOG, REPO, has_original_intent, hex_value, need, verify_launch


CANDIDATE = 'vggt-tum-bidirectional-native30'
INTENT = f'local-only/{LABEL}-original-dispatch/intent.json'
ATTEMPT = f'local-only/{LABEL}-original-dispatch/attempt.json'
BRANCH = 'refs/heads/bw1000-experiments'
ORIGIN = 'git@github.com:whjthu/InfiniLM-hygon-test.git'


def exact_value(document, values, detail):
    need(all(key in document and type(document[key]) is type(value) and document[key] == value
             for key, value in values.items()), detail)


def target_checks(result, upstream_count):
    exact_value(result, dict(state='TARGET_ASSETS_VERIFIED_DEVICE_PENDING', request=REQUEST,
                            parent=PARENT, node=NODE, source_manifest_SHA256=SOURCE_MANIFEST,
                            claim_SHA256=CLAIM_SHA, left_unmodified=True, proposed_updates=30,
                            verified_source_members=15, PCI_KFD_PID_verified=False, training_READY=False),
                'target receipt does not bind the adopted read-only scope')
    original = result['original']
    exact_value(original, dict(namespace_absent=True, accounting_label_matches=0, discovered_rows=[],
                              original_read_only_guard_passed=True), 'original request requires reconciliation')
    need(all(hex_value(value, 64) for value in (original['raw_accounting_SHA256'],
                                                result['first_namespace_accounting_SHA256'])),
         'both original accounting observations are required')
    exact_value(result['assets'], dict(model_stat_verified=True, original_model_pin=WEIGHT_SHA,
                                     model_content_bytes_read=0, verified_framework_files=393,
                                     verified_training_media_files=17, verified_upstream_files=upstream_count,
                                     runtime_path_present=True, runtime_imported=False), 'target asset inventory differs')
    start, end = result['membership_observed_epoch'], result['returned_epoch']
    need(all(type(value) in (int, float) and math.isfinite(value) for value in (start, end))
         and 0 < start <= end <= start + 90, 'target observations are outside the same bounded preflight')


def published(repo, publication, members):
    def git(*args):
        return subprocess.check_output(['git', '-c', 'safe.directory='+repo.as_posix(), *args],
                                       cwd=repo, timeout=12, stdin=subprocess.DEVNULL)
    need(git('remote', 'get-url', 'origin').decode().strip() == ORIGIN, 'unexpected publication origin')
    need(git('ls-remote', '--exit-code', 'origin', BRANCH).decode().split() == [publication, BRANCH],
         'published tip changed; verify the current source before taking a device receipt')
    for path in members:
        need(git('cat-file', 'blob', f'{publication}:{path}') == (repo / path).read_bytes(),
             'published source bytes differ: '+path)


def write_once(repo, path, raw):
    with path.open('xb') as stream:
        stream.write(raw)
    return dict(path=path.relative_to(repo).as_posix(), SHA256=sha256(raw).hexdigest())


def as_json(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def prepare(repo, backlog, raw, output, publication, *, check_publication=published):
    repo = repo.resolve()
    output = output.resolve()
    need(output.is_relative_to(repo/'local-only') and output != repo/'local-only',
         'write only a new private receipt directory')
    need(hex_value(publication, 40) and not has_original_intent(repo, [INTENT, ATTEMPT]),
         'bad publication identity or a prior original intent; reconcile without replay')
    [candidate] = [row for row in backlog['items'] if row['id'] == CANDIDATE]
    exact_value(candidate, dict(case='T12.domain-tum-bidirectional30', kind='training', state='blocked',
                                placement={NODE: 1}, min_updates=30, attempt_receipts=[INTENT, ATTEMPT]),
                'base backlog must preserve the original candidate')
    sources, _ = check_source(repo)
    target_checks(json.loads(raw), len(json.loads(sources['UPSTREAM_FILES.json'])['files']))

    manifest_path = PACKAGE/'SOURCE_FILES.json'
    claim_path = PACKAGE/'CLAIM.json'
    launcher = PACKAGE/'launch.py'
    manifest = json.loads((repo/manifest_path).read_bytes())['files']
    claim = json.loads((repo/claim_path).read_bytes())
    members = {manifest_path.as_posix(), *((PACKAGE/name).as_posix() for name in manifest),
               *(pin['path'] for pin in claim['sources'].values())}
    check_publication(repo, publication, sorted(members))
    need(not has_original_intent(repo, [INTENT, ATTEMPT]), 'original intent appeared during publication check')
    output.mkdir()
    original = write_once(repo, output/'target-preflight.json', raw)
    source = dict(claim=dict(path=claim_path.as_posix(), SHA256=CLAIM_SHA),
                  manifest=dict(path=manifest_path.as_posix(), SHA256=SOURCE_MANIFEST),
                  launcher=dict(path=launcher.as_posix(), SHA256=manifest['launch.py']['SHA256']))
    scope = sha256(as_json(claim['semantic_key'])).hexdigest()
    launch = dict(source, request=REQUEST, publication_commit=publication, workload_SHA256=scope,
                  required_ports=[], authorization_key='training_GO', attempt_receipts=[INTENT, ATTEMPT])
    receipts = {
        'publication_receipt': dict(state='PUBLISHED_MATCHED', publication_commit=publication,
                                    **{name+'_SHA256': pin['SHA256'] for name, pin in source.items()}),
        'prerequisite_receipt': dict(state='TARGET_PREREQUISITES_VERIFIED', workload_SHA256=scope,
                                     placement={NODE: 1}, min_updates=30, required_ports=[]),
        'original_request_receipt': dict(state='REMOTE_UNSTARTED_VERIFIED', original_namespace_absent=True,
                                         accounting_label_matches=0, prior_invocation=False),
    }
    for name, receipt in receipts.items():
        launch[name] = write_once(repo, output/(name+'.json'), as_json(dict(receipt, request=REQUEST,
                                                                          target_preflight=original)))
    ready = dict(candidate, state='ready', launch=launch,
                 next_action='用新鲜设备回执和原子派发申领提交原训练，原步骤终态及独立清理后交还节点。')
    need(verify_launch(repo, ready) == 'ready', 'original request now requires reconciliation')
    overlay = dict(backlog, items=[ready if row is candidate else row for row in backlog['items']])
    return write_once(repo, output/'backlog.json', as_json(overlay))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_preflight', type=Path)
    parser.add_argument('publication')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.target_preflight.open('rb') as stream:
        raw = stream.read(128 * 1024 + 1)
    need(0 < len(raw) <= 128 * 1024, 'invalid authenticated preflight capture size')
    overlay = prepare(REPO, json.loads(BACKLOG.read_bytes()), raw, args.output, args.publication)
    print(json.dumps(dict(backlog=overlay, request=REQUEST, actual_slurm_submissions=0)))


if __name__ == '__main__':
    main()
