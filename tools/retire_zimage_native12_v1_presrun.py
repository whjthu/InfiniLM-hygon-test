"""Close only the rejected, pre-srun Z-Image V1 validation attempt."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from report_training80_backlog import QUEUE_DB, REPO, TRAINING80_LEDGER, fleet_view
from tuning_queue import Queue


REQUEST = '4ce20eb249de478a9f2c849b28f0db93'
RUN = '3970c88638857b890371389b68e130bc67b3186b497e2de3fb80e2d61d2b68e4'
PARENT, NODE = '1087656', 'f11r4n13'
DIRECTORY = Path('local-only/training-zimage-style18-native12-CPU-v1-original')
INTENT = (DIRECTORY / 'intent.json').as_posix()
PINS = {
    'original-plan.json': 'f3381714e339c248991ef05b86a8425f922a8a1d006817f4975e48e99edec547',
    'stderr.private': 'cc7f4b0f6aeaca5d88efe0a709ec17a2ec2bcfd6d3f913e909e023bd291f399f',
    'transport.json': 'e9ae0b41dacb390f4063c543b346e62a9e43cc260596a856e8397a5d6785b6c5',
    'original-remote-presrun-absence-proof.json': 'e164a2c51e9bafcdd606a8284481c287f7a34c32f26baa575f8f3c48cf062cac',
}


def need(check, detail):
    if not check:
        raise ValueError(detail)


def checked(directory, name):
    path = directory / name
    need(path.resolve(strict=True) == path and path.is_file(), 'original scope has noncanonical evidence')
    with path.open('rb') as source:
        raw = source.read(8193)
    need(0 < len(raw) <= 8192 and sha256(raw).hexdigest() == PINS[name], 'original evidence differs: '+name)
    return raw


def close(snapshot, repo=REPO):
    """Consume one fresh zero-card device check after the original read-only reconciliation."""
    directory = (repo / DIRECTORY).resolve(strict=True)
    need(directory == repo / DIRECTORY, 'original private scope is not canonical')
    plan = json.loads(checked(directory, 'original-plan.json'))
    transport = json.loads(checked(directory, 'transport.json'))
    remote = json.loads(checked(directory, 'original-remote-presrun-absence-proof.json'))
    logs = checked(directory, 'stderr.private').decode(errors='replace')
    need("ModuleNotFoundError: No module named 'qualified_sampler'" in logs
         and 'scheduler = load_scheduler(HERE)' in logs,
         'the original target import failure differs')
    need(plan['request'] == transport['request'] == remote['z_request'] == REQUEST
         and transport['queue_run'] == RUN and transport['transport_attempted'] is True
         and transport['transport_returncode'] == 1 and remote['z_plan_SHA256'] == PINS['original-plan.json']
         and plan['epoch'] == remote['z_epoch'] and remote['observed_UTC_epoch'] > plan['epoch'] + 300,
         'the original attempt or its deadline differs')
    expected_remote_flags = ('z_missing_scheduler_leaf', 'z_no_result_namespace', 'z_no_intent',
                             'z_no_submitted', 'z_no_first_accounting')
    need(all(remote[k] is True for k in expected_remote_flags) and remote['z_acc'] == [],
         'an original result or step exists; preserve the claim')
    now = time.monotonic()
    _, nodes, fresh = fleet_view(snapshot, now, time.time()-now, repo / TRAINING80_LEDGER.relative_to(REPO))
    state, item = nodes[NODE], snapshot['nodes'][NODE]
    need(fresh and state['admission'] == 'clear' and (state['parent'],state['zone']) == (PARENT,'T07')
         and item['membership_valid'] is item['PID_clean'] is item['KFD_clean'] is True
         and not item['active_steps'] and len(set(item['PCIs'])) == 8
         and re.fullmatch(PARENT+r'\.[0-9]+',item['device_probe_step'])
         and item['device_probe_step'] != plan['before_CPU_step'],
         'independent device cleanup absent or stale')
    receipt = dict(request_id=REQUEST, intent=INTENT, failure_phase='PRE_SLURM_IMPORT',
                   transport_returncode=1, launch_attempted=False, matching_accounting_rows=0,
                   post_clean=True, node=NODE)
    record = dict(receipt, original_pins=PINS, new_probe_step=item['device_probe_step'],
                  fresh_device_SHA256=snapshot['local_receipt']['raw_response_SHA256'])
    raw = (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode()
    receipt['evidence_SHA256'] = sha256(raw).hexdigest()
    with (directory/'independent-presrun-clean-post.json').open('xb') as stream:
        stream.write(raw)
    queue = Queue(repo / QUEUE_DB.relative_to(REPO))
    with closing(queue.db):
        queue.retire_remote_prelaunch_direct(RUN,receipt)
        need(queue.dispatch_state(RUN)['state'] == 'RETIRED_PRELAUNCH','original queue transition failed')
    return dict(record, evidence_SHA256=receipt['evidence_SHA256'], queue_state='RETIRED_PRELAUNCH')
