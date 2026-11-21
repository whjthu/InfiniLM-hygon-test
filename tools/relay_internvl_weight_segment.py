"""Pipe one source segment into its receiver using pre-admitted launch callbacks."""

import json
import math
import os
import re
from subprocess import TimeoutExpired
from tempfile import TemporaryFile
from time import monotonic

from stage_internvl_weight_segment import need, segment_length


def read_receipt(stream):
    stream.seek(0)
    raw = stream.read(16385)
    need(0 < len(raw) <= 16384, 'missing or oversized endpoint receipt')
    value = json.loads(raw)
    need(isinstance(value, dict), 'endpoint receipt must be an object')
    return value


def relay(spec, offset, deadline, start_sender, start_receiver):
    """Use a caller's monotonic deadline and Popen callbacks with binary stdio.

    The receiver starts first. Neither callback should block waiting for its
    child. Both commands must already be bound to the same original segment,
    authorized hosts and remote deadline. A failure requires reconciliation of
    the target checkpoint before any further receiver or sender invocation.
    """
    count = segment_length(spec, offset)
    need(math.isfinite(deadline) and monotonic() < deadline <= monotonic()+600,
         'one bounded relay deadline required')
    children = []
    with TemporaryFile() as source_receipt, TemporaryFile() as target_receipt, TemporaryFile() as target_error:
        try:
            read_fd, write_fd = os.pipe()
            with os.fdopen(read_fd, 'rb') as reader, os.fdopen(write_fd, 'wb') as writer:
                receiver = start_receiver(stdin=reader, stdout=target_receipt, stderr=target_error)
                children.append(receiver)
                reader.close()
                need(monotonic() < deadline, 'relay deadline expired before source start')
                sender = start_sender(stdout=writer, stderr=source_receipt)
                children.append(sender)
            source_rc = sender.wait(timeout=max(0, deadline-monotonic()))
            target_rc = receiver.wait(timeout=max(0, deadline-monotonic()))
            need((source_rc, target_rc) == (0, 0), f'endpoint exit codes {source_rc}/{target_rc}')
            source = read_receipt(source_receipt)
            target = read_receipt(target_receipt)
            identity = (spec['file_id'], offset, count)
            keys = ('file_id', 'offset', 'raw_bytes')
            need(tuple(source[k] for k in keys) == tuple(target[k] for k in keys) == identity
                 and source['source_stamp_verified'] is True
                 and target['whole_file_qualified'] is False
                 and target['confirmed_bytes'] == offset+count
                 and re.fullmatch('[a-f0-9]{64}', source['segment_SHA256'])
                 and source['segment_SHA256'] == target['segment_SHA256'],
                 'source and target segment receipts disagree')
            return target
        except (OSError, ValueError, TimeoutExpired, KeyError, TypeError) as error:
            raise RuntimeError('inspect the original target segment checkpoint before further transfer') from error
        finally:
            cleanup_deadline = monotonic()+5
            for child in children:
                if child.poll() is None:
                    child.kill()
            for child in children:
                child.wait(timeout=max(0, cleanup_deadline-monotonic()))
