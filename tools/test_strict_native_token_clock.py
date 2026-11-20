from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
from threading import Thread
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/bw1000/public-token-clock-r480'))
try:
    import strict_native_token_clock as strict
finally:
    del sys.path[0]


def notice(**row):
    return strict.PREFIX + json.dumps(dict(request_id='test-request', **row)).encode() + b'\n\n'


TOKEN = notice(ordinal=0, token_id=4, eos=False, special=False, visible=True)
TEXT = 'data: {"choices":[{"delta":{"content":"雪"}}]}\n\n'.encode()
TERMINAL = notice(terminal=True, emitted=1, notices_sent=1, observer_overflow=False, owned_finished=True,
                  finish_reason='eos_token')
DONE = b'data: [DONE]\n\n'


def loopback(body, length):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(length))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()
        client = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        reads = []
        try:
            start = time.monotonic_ns()
            client.request('GET', '/')
            response = client.getresponse()
            result = strict.collect_response(response, start, lambda raw, stamp: reads.append((raw, stamp)))
        finally:
            client.close()
            thread.join(5)
        return result.summary(), reads


class StrictTokenClockTest(unittest.TestCase):
    def clock(self):
        clock = strict.ClientClock(10)
        clock.begin_response(200)
        return clock

    def test_real_http_eof_after_internal_terminal_keeps_request_incomplete(self):
        body = TOKEN + TEXT + TERMINAL
        observed, reads = loopback(body, len(body + DONE))
        legacy = strict.legacy.ClientClock(reads[0][1] - 1)
        legacy.begin_response(200)
        for raw, stamp in reads:
            legacy.feed(raw, stamp, final=not raw)
        self.assertTrue(legacy.summary()['complete'])
        self.assertTrue(observed['observer_clock_coverage_complete'])
        self.assertFalse(observed['standard_done_received'])
        self.assertTrue(observed['http_body_truncated'])
        self.assertFalse(observed['complete'])
        self.assertTrue(observed['missing_or_rejected'])

    def test_real_http_complete_body_retains_both_ttft_clocks(self):
        body = TOKEN + TEXT + TERMINAL + DONE
        observed, reads = loopback(body, len(body))
        self.assertEqual(b''.join(raw for raw, _ in reads), body)
        self.assertTrue(observed['standard_done_received'] and observed['http_eof_received'])
        self.assertTrue(observed['complete'])
        self.assertIsNotNone(observed['semantic_TTFT_s'])
        self.assertIsNotNone(observed['token_ID_receipt_TTFT_s'])
        self.assertIsNone(observed['decode_tok_s'])

    def test_real_http_receipt_with_done_still_requires_complete_content_length(self):
        body = TOKEN + TEXT + TERMINAL + DONE
        observed, _ = loopback(body, len(body) + 1)
        self.assertTrue(observed['standard_done_received'])
        self.assertTrue(observed['http_body_truncated'])
        self.assertFalse(observed['complete'])

    def test_split_crlf_unicode_and_done_preserve_receipt_boundary(self):
        clock = self.clock()
        stream = (TOKEN + TEXT + TERMINAL + DONE).replace(b'\n', b'\r\n')
        for tick, octet in enumerate(stream, 20):
            clock.feed(bytes((octet,)), tick)
        self.assertFalse(clock.summary()['complete'])
        clock.feed(b'', 20 + len(stream), final=True)
        self.assertTrue(clock.summary()['complete'])
        self.assertGreater(clock.summary()['semantic_TTFT_s'], clock.summary()['token_ID_receipt_TTFT_s'])

    def test_done_without_terminal_and_extra_event_after_done_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'without the native terminal'):
            self.clock().feed(TOKEN + DONE, 20)
        with self.assertRaisesRegex(ValueError, r'after \[DONE\]'):
            self.clock().feed(TOKEN + TERMINAL + DONE + TEXT, 20)

    def test_terminal_mismatch_and_request_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'counts or flags differ'):
            self.clock().feed(TOKEN + TERMINAL.replace(b'"notices_sent": 1', b'"notices_sent": 2'), 20)
        with self.assertRaisesRegex(ValueError, 'request identity differs'):
            self.clock().feed(TOKEN + TERMINAL.replace(b'test-request', b'wrong-request'), 20)

    def test_overflow_keeps_a_complete_wire_outside_qualified_results(self):
        clock = self.clock()
        overflow = TERMINAL.replace(b'"observer_overflow": false', b'"observer_overflow": true')
        overflow = overflow.replace(b'"emitted": 1', b'"emitted": 2')
        clock.feed(TOKEN + overflow + DONE, 20, final=True)
        self.assertTrue(clock.summary()['standard_done_received'])
        self.assertFalse(clock.summary()['complete'])

    def test_coalesced_token_reads_do_not_create_distinct_timestamps(self):
        clock = self.clock()
        second = TOKEN.replace(b'"ordinal": 0', b'"ordinal": 1')
        terminal = TERMINAL.replace(b'"emitted": 1', b'"emitted": 2').replace(b'"notices_sent": 1', b'"notices_sent": 2')
        clock.feed(TOKEN + second + terminal + DONE, 20, final=True)
        summary = clock.summary()
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['visible_nonEOS'], 2)
        self.assertEqual(summary['ITL_p99_s'], 0)
        self.assertIsNone(summary['decode_tok_s'])


if __name__ == '__main__':
    unittest.main()
