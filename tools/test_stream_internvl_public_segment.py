"""Exercise range admission before the target can be mutated."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import stream_internvl_public_segment as relay


class Response(BytesIO):
    status = 206
    headers = {'Content-Range': 'bytes 1048576-1048580/1048581', 'Content-Length': '5'}

    def geturl(self):
        return 'https://us.aws.cdn.hf.co/opaque'


class PublicSegmentTest(unittest.TestCase):
    spec = {'file_id': 4, 'filename': 'model-00004-of-00004.safetensors',
            'bytes': 1048581, 'segment_bytes': 1048576}

    def fetch(self, response):
        def opener(request, *, timeout):
            self.assertIn(relay.REVISION, request.full_url)
            self.assertEqual(request.get_header('Range'), 'bytes=1048576-1048580')
            self.assertGreater(timeout, 0)
            return response

        return relay.public_range(self.spec, 1048576, time.monotonic() + 3, open_url=opener)

    def test_exact_short_final_range(self):
        self.assertEqual(self.fetch(Response(b'abcde')), b'abcde')

    def test_unexpected_headers_rejected(self):
        for name, value in [('Content-Range', 'bytes 0-4/5'),
                            ('Content-Length', '6'), ('Content-Encoding', 'gzip')]:
            with self.subTest(name=name):
                response = Response(b'abcde')
                response.headers = {**response.headers, name: value}
                with self.assertRaises(ValueError):
                    self.fetch(response)

    def test_truncated_or_extra_body_rejected(self):
        for body in (b'abcd', b'abcdef'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.fetch(Response(body))

    def test_timeout_resumes_at_last_returned_byte(self):
        class Interrupted(Response):
            def read1(self, size):
                if self.tell():
                    raise TimeoutError('interrupted after returning data')
                return super().read1(min(size, 2))

        suffix = Response(b'cde')
        suffix.headers = {'Content-Range': 'bytes 1048578-1048580/1048581', 'Content-Length': '3'}
        responses = iter((Interrupted(b'ab'), suffix))
        seen = []

        def opener(request, *, timeout):
            seen.append(request.get_header('Range'))
            return next(responses)

        raw = relay.public_range(self.spec, 1048576, time.monotonic() + 3, open_url=opener)
        self.assertEqual(raw, b'abcde')
        self.assertEqual(seen, ['bytes=1048576-1048580', 'bytes=1048578-1048580'])

    def test_reconnects_are_bounded_by_original_budget_and_count(self):
        calls = []

        def timeout(*args, **kwargs):
            calls.append(kwargs['timeout'])
            raise TimeoutError('publisher interrupted')

        with self.assertRaises(TimeoutError):
            relay.public_range(self.spec, 1048576, time.monotonic() + 3, open_url=timeout)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(0 < value <= 3 for value in calls))
        with self.assertRaises(ValueError):
            relay.public_range(self.spec, 1048576, time.monotonic() - 1, open_url=timeout)
        self.assertEqual(len(calls), 4)

    def test_frozen_receiver_bootstrap(self):
        packet = relay.make_program(self.spec, 'inspect', None, None, 5)
        self.assertEqual(int.from_bytes(packet[:8], 'little'), len(packet) - 8)
        compile(packet[8:], '<receiver>', 'exec')

    def test_window_resumes_and_seals_after_last_range(self):
        spec = {**self.spec, 'bytes': 10, 'segment_bytes': 5}
        entries = [
            {'action': 'segment_committed', 'target': {'offset': 5, 'confirmed_bytes': 10}},
            {'action': 'sealed', 'target': {'whole_file_qualified': True}},
        ]
        with TemporaryDirectory() as tmp, patch.object(relay.stage, 'shard_spec', return_value=spec):
            directory = Path(tmp) / 'window'
            with patch.object(relay, 'run_segment', side_effect=entries) as run:
                relay.continue_shard(4, 'published', directory, None, start_offset=5,
                                     max_range_count=1, window_seconds=420, report=lambda entry: None)
            self.assertEqual([call.kwargs['expected_offset'] for call in run.call_args_list], [5, 10])
            self.assertEqual(json.loads((directory / 'committed-0001.json').read_bytes()), entries[0])
            self.assertEqual(json.loads((directory / 'window-terminal.json').read_bytes())['state'],
                             'WHOLE_SHARD_VERIFIED')

    def test_window_stops_at_unverified_range(self):
        spec = {**self.spec, 'bytes': 15, 'segment_bytes': 5}
        first = {'action': 'segment_committed', 'target': {'offset': 5, 'confirmed_bytes': 10}}
        with TemporaryDirectory() as tmp, patch.object(relay.stage, 'shard_spec', return_value=spec):
            directory = Path(tmp) / 'window'
            with patch.object(relay, 'run_segment', side_effect=[first, TimeoutError()]) as run:
                with self.assertRaises(TimeoutError):
                    relay.continue_shard(4, 'published', directory, None, start_offset=5,
                                         max_range_count=2, window_seconds=420, report=lambda entry: None)
            self.assertEqual(run.call_count, 2)
            status = json.loads((directory / 'window-terminal.json').read_bytes())
            self.assertEqual(status['last_acknowledged_offset'], 10)
            self.assertEqual(status['state'], 'STOPPED_REQUIRES_FRESH_TARGET_RECONCILIATION')

if __name__ == '__main__':
    unittest.main()
