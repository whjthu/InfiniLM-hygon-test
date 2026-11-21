"""Bind one SDXL stage receive to the existing authenticated binary-stdin client."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import PurePosixPath

from report_training80_backlog import REPO, need, pin_file


REQUEST = '274f1a27b7ea44f198eaa08589b845b49'
PACKET_SHA = 'd860465dc1dde228a1fabbc716f581f2adbe67fe72f2f31841305023a9a0c302'
PACKET_BYTES = 290733
BINDING = 'bc268682bdfad89dbd3a05a93214fdeffa7415ca3b3d9dc3c8804c9ec9f96c58'
TARGET = 'results/training-sdxl-USGS-style18-domain25-data-v1'
PACKET = 'local-only/training-sdxl-USGS-style18-domain25-data-v1-first/train-only.zip'
WORK = 'local-only/training-sdxl-style18-train-staging-v1-original-dispatch'
ROOT = PurePosixPath('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
RECEIVER = dict(path='tools/stage_sdxl_style18_train.py',
                SHA256='b578b5b76a52a1c7a4256ad73f27fac1664c2decbf555aa46b1f742419b1de10')
TIMEOUT = 75
MAX_RECEIPT = 2048


def argv(remote_checkout, action):
    checkout = PurePosixPath(remote_checkout)
    need(checkout.is_relative_to(ROOT) and checkout.as_posix() == remote_checkout and '..' not in checkout.parts
         and all(part.isascii() and all(char.isalnum() or char in '-_.' for char in part)
                 for part in checkout.parts[1:]), 'expected the canonical checkout under the original project')
    need(action in ('inspect', 'receive'), 'unknown frozen stage action')
    return ('/usr/local/python3.12/bin/python3', '-B', str(checkout / RECEIVER['path']),
            action, '--packet-sha256', PACKET_SHA)


def checked_packet(repo):
    pin_file(repo, RECEIVER, ('tools/',))
    path = repo / PACKET
    need(path.resolve(strict=True) == path and path.is_file(), 'expected the original private packet file')
    with path.open('rb') as stream:
        raw = stream.read(PACKET_BYTES + 1)
    need(len(raw) == PACKET_BYTES and sha256(raw).hexdigest() == PACKET_SHA, 'original private packet bytes differ')
    return raw


def parse_receipt(raw, action):
    need(isinstance(raw, bytes) and 0 < len(raw) <= MAX_RECEIPT, 'stage response is not one bounded binary receipt')
    doc = json.loads(raw)
    state = doc['state']
    core = dict(request=REQUEST, training_READY=False, state=state)
    if action == 'inspect' and state in ('ORIGINAL_DATA_NAMESPACE_ABSENT', 'ORIGINAL_DATA_NAMESPACE_INCOMPLETE'):
        need(doc == core and doc['training_READY'] is False, 'original namespace receipt scope differs')
        return doc
    expected = dict(core, binding_SHA256=BINDING, packet_SHA256=PACKET_SHA, files=13, train_bytes=289527)
    if action == 'receive':
        wall = doc['wall_seconds']
        need(state == 'SDXL_TRAIN_DATA_STAGED' and type(wall) in (int, float) and math.isfinite(wall) and 0 <= wall <= 60,
             'receiver receipt does not describe a bounded stage')
        expected['wall_seconds'] = wall
    else:
        need(state == 'SDXL_TRAIN_DATA_STAGED_AND_RECHECKED', 'unexpected original inspection state')
    need(doc == expected and doc['training_READY'] is False and type(doc['files']) is int and
         type(doc['train_bytes']) is int, 'original completed stage receipt differs')
    return doc


def invoke(transport, command, action, payload):
    info = dict(request=REQUEST, transport_returncode=None, state='UNKNOWN', training_ready_added=0,
                target_installed_verified=False, replay_allowed=False)
    try:
        reply = transport(command, input=payload, timeout=TIMEOUT)
        need(type(reply.returncode) is int, 'missing transport exit status')
        info['transport_returncode'] = reply.returncode
        if reply.returncode == 0:
            doc = parse_receipt(reply.stdout, action)
            info.update(state=doc['state'], receipt=doc, receipt_SHA256=sha256(reply.stdout).hexdigest(),
                        target_installed_verified=action == 'inspect' and
                        doc['state'] == 'SDXL_TRAIN_DATA_STAGED_AND_RECHECKED')
    except Exception as error:
        info['transport_or_receipt_error'] = type(error).__name__
    return info


def inspect(remote_checkout, transport):
    """Read the same original namespace with empty binary stdin, without resending."""
    return invoke(transport, argv(remote_checkout, 'inspect'), 'inspect', b'')


def write_once(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def receive_once(remote_checkout, transport, *, repo=REPO):
    """Observe first, persist the sole local intent, then pass the exact ZIP bytes."""
    repo = repo.resolve(strict=True)
    work = repo / WORK
    need(work.parent.resolve(strict=True) == work.parent, 'expected canonical local-only directory')
    need(not work.exists() and not work.is_symlink(), 'original local stage attempt exists; use inspect only')
    command = argv(remote_checkout, 'receive')
    packet = checked_packet(repo)
    before = inspect(remote_checkout, transport)
    if before['state'] != 'ORIGINAL_DATA_NAMESPACE_ABSENT':
        return dict(before, receive_attempted=False)
    work.mkdir()
    write_once(work / 'intent.json', dict(request=REQUEST, packet_bytes=PACKET_BYTES, packet_SHA256=PACKET_SHA,
                                        target_scope=TARGET, argv=command, prior_inspection=before,
                                        local_epoch_UTC=datetime.now(timezone.utc).isoformat(),
                                        remote_outcome='UNKNOWN', replay_allowed=False))
    outcome = dict(invoke(transport, command, 'receive', packet), receive_attempted=True)
    write_once(work / 'receive-result.json', outcome)
    return outcome
