"""Exercise range admission before the target can be mutated."""

from io import BytesIO
import time
import unittest

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

    def test_frozen_receiver_bootstrap(self):
        packet = relay.make_program(self.spec, 'inspect', None, None, 5)
        self.assertEqual(int.from_bytes(packet[:8], 'little'), len(packet) - 8)
        compile(packet[8:], '<receiver>', 'exec')


if __name__ == '__main__':
    unittest.main()
