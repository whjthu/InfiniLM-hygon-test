"""Record the authenticated no-submission result for the original Qwen Dev80 request."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from preflight_qwen_usgs_exposure25 import REQUEST as TRAINING_REQUEST
from reconcile_qwen_usgs_dev80_original import (
    AUTH_SHA, NODES, PARENT, PUBLICATION, REPO, REQUEST, need, pre_exec_receipt,
)


JOB = 'qwen-usgs-fixed-dev80-base-adapter'
RECORD = 'local-only/training80-reconciliation/qwen-usgs-dev80-original-absence.json'
MAX_BYTES = 128 * 1024


def validate(repo, raw):
    need(0 < len(raw) <= MAX_BYTES, 'invalid original Dev80 reconciliation receipt size')
    proof = json.loads(raw)
    if type(proof) is dict and proof.get('request') == TRAINING_REQUEST:
        need(proof['state'] == 'TRAINING_ASSETS_VERIFIED_DEVICE_PENDING' and proof['read_only'] is True,
             'the Qwen training preflight did not complete')
        for observation in ('first', 'last'):
            observed = proof[observation]
            need(observed['predecessor_namespace_absent'] is True and observed['predecessor_rows'] == {},
                 'the Qwen training preflight found a later Dev80 request')
        proof = proof['predecessor']
    expected = dict(state='ABSENCE_VERIFIED_AFTER_PRE_EXEC_AUTH_FAILURE', request=REQUEST,
                    publication=PUBLICATION, parent=PARENT, node_placement=NODES,
                    original_replay_allowed=False, original_not_submitted_verified=True,
                    original_hold_eligible_for_removal=True, current_pci_KFD_port_admission=False,
                    training_READY=False, pre_exec_receipt_SHA256=AUTH_SHA)
    need(type(proof) is dict and all(type(proof.get(k)) is type(v) and proof[k] == v for k,v in expected.items()),
         'original Dev80 absence is not verified')
    need(pre_exec_receipt(repo) == AUTH_SHA, 'original Dev80 pre-execution evidence differs')
    for observation in ('first', 'last'):
        item = proof[observation]
        accounting = item['accounting']
        digest = accounting['raw_SHA256']
        namespace = item['namespace']
        need(type(namespace) is dict and namespace.get('present') is False and len(namespace) == 1 and accounting['rows'] == {}
             and type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
             'original Dev80 namespace or Slurm accounting includes a possible submission')
    placement = proof['membership']
    epoch = placement['observed_epoch']
    need(placement['node_membership'] == dict.fromkeys(NODES.values(), 'NO_NUMBERED_STEP')
         and type(epoch) in (int,float) and math.isfinite(epoch) and epoch > 0,
         'original Dev80 nodes were not both clear during reconciliation')
    return dict(request=REQUEST, publication=PUBLICATION, receipt_SHA256=sha256(raw).hexdigest(),
                nodes=sorted(NODES.values()), observed_epoch=epoch, replay_allowed=False,
                status='verified_not_submitted')


def bounded(path):
    with path.open('rb') as stream:
        return stream.read(MAX_BYTES + 1)


def read_record(repo):
    record = repo / RECORD
    need(record.resolve() == record and not record.is_symlink(),
         'original Dev80 receipt escaped the private receipt directory')
    try:
        raw = bounded(record)
    except FileNotFoundError:
        return None
    return validate(repo, raw)


def record_once(repo, raw):
    result = validate(repo, raw)
    path = repo / RECORD
    need(path.resolve() == path and not path.is_symlink(),
         'original Dev80 receipt escaped the private receipt directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(raw)
    except FileExistsError:
        need(bounded(path) == raw, 'existing original Dev80 receipt differs; preserve its original bytes')
    return dict(result, path=RECORD, actual_slurm_submissions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('authenticated_receipt', type=Path, help='retained stdout from original read-only reconciliation or Qwen preflight')
    args = parser.parse_args()
    print(json.dumps(record_once(REPO, bounded(args.authenticated_receipt)), allow_nan=False))


if __name__ == '__main__':
    main()
