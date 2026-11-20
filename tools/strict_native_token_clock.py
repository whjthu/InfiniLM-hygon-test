"""Require a complete SSE transport before accepting the native token clock."""

from hashlib import sha256
import json
from pathlib import Path
import re
import time

import client_clock as legacy
import token_clock as producer


PINS = (
    (legacy, 'c1fd4ff49ab222472b1ec6be054f939865f131094ec7e6afa7b10590f9b30ffa'),
    (producer, '086d9ba47efe965a856d4e570c8a7f3f965f58fe203141aa8821383d45be8be6'),
)
EVENT_END = re.compile(rb'\r?\n\r?\n')
PREFIX = producer.PREFIX.encode('ascii')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


class ClientClock(legacy.ClientClock):
    def __init__(self, start_ns):
        super().__init__(start_ns)
        self.pending = bytearray()
        self.done_received = False
        self.eof_received = False
        self.http_body_truncated = False
        self.last_receive_ns = start_ns
        self.request_id = None

    def begin_response(self, status):
        require(self.http_status is None, 'HTTP response status was already set')
        super().begin_response(status)

    def feed_event(self, event, received_ns):
        lines = event.splitlines()
        if not any(line.strip() for line in lines):
            return
        require(not self.done_received, 'unexpected SSE event after [DONE]')
        data = b'\n'.join(line[5:].lstrip() for line in lines if line.startswith(b'data:'))
        notices = [line for line in lines if line.startswith(PREFIX)]
        require(len(notices) <= 1 and not (notices and data), 'mixed native clock and SSE data event')
        prior_terminal = self.terminal
        if prior_terminal is not None:
            require(data == b'[DONE]' and not notices, 'the native terminal must be followed by [DONE]')
        if notices:
            record = json.loads(notices[0][len(PREFIX):])
            request_id = record.get('request_id')
            require(isinstance(request_id, str) and request_id and self.request_id in (None, request_id),
                    'native clock request identity differs')
        super().feed(event, received_ns)
        if notices:
            if self.terminal is not prior_terminal:
                emitted, sent = record.get('emitted'), record.get('notices_sent')
                overflow = record.get('observer_overflow')
                require(type(emitted) is int and type(sent) is int and sent == len(self.tokens) and
                        type(record.get('owned_finished')) is bool and type(overflow) is bool and
                        (emitted >= sent if overflow else emitted == sent),
                        'native terminal counts or flags differ from the receipt')
            self.request_id = request_id
        if data == b'[DONE]':
            require(self.terminal is not None, '[DONE] arrived without the native terminal')
            self.done_received = True

    def feed(self, raw, received_ns, final=False):
        require(self.http_status is not None and not self.eof_received, 'unexpected response body boundary')
        if self.http_status != 200:
            return
        require(received_ns >= self.last_receive_ns, 'nonmonotonic client receive clock')
        self.last_receive_ns = received_ns
        self.pending.extend(raw)
        start = 0
        for boundary in EVENT_END.finditer(self.pending):
            end = boundary.end()
            self.feed_event(bytes(self.pending[start:end]), received_ns)
            start = end
        del self.pending[:start]
        if final:
            require(not self.pending.strip(), 'truncated SSE event at EOF')
            self.eof_received = True

    def summary(self):
        result = super().summary()
        wire_complete = self.done_received and self.eof_received and not self.http_body_truncated
        result.update(standard_done_received=self.done_received, http_eof_received=self.eof_received,
                      http_body_truncated=self.http_body_truncated,
                      observer_clock_coverage_complete=result['clock_coverage_complete'],
                      clock_coverage_complete=wire_complete and result['clock_coverage_complete'],
                      complete=wire_complete and result['complete'],
                      missing_or_rejected=not wire_complete or result['missing_or_rejected'])
        return result


def collect_response(response, start_ns, save_read=None):
    for module, digest in PINS:
        require(sha256(Path(module.__file__).read_bytes()).hexdigest() == digest, 'unexpected native clock dependency')
    clock = ClientClock(start_ns)
    clock.begin_response(response.status)
    if response.status != 200:
        return clock
    while True:
        raw = response.read1(65536)
        received_ns = time.monotonic_ns()
        if save_read is not None:
            save_read(raw, received_ns)
        clock.feed(raw, received_ns, final=not raw)
        if not raw:
            remaining = getattr(response, 'length', None)
            clock.http_body_truncated = type(remaining) is int and remaining > 0
            return clock
