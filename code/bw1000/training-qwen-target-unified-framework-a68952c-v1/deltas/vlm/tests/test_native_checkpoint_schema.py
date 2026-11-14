"""Header boundaries including the requirement that no payload byte is read."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from vlm.runtime.native_checkpoint_schema import read_header


class HeaderTests(unittest.TestCase):
    def write(self, directory, header, payload=b'\x12\x34\x56\x78'):
        raw = header if isinstance(header, bytes) else json.dumps(header).encode()
        p = Path(directory) / 'test.safetensors'
        p.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
        return p

    def test_bounded_header_and_no_full_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            p = self.write(directory, {'p': dict(dtype='BF16', shape=[2], data_offsets=[0, 4])})
            value = read_header(p, p.stat().st_size, check=lambda: None)
            self.assertEqual(value['payload_bytes_read'], 0)
            self.assertEqual(value['bytes_read'], p.stat().st_size-4)
            self.assertIsNone(value['full_file_SHA256'])
            self.assertEqual(value['records']['p']['shape'], [2])

    def test_corrupt_tensor_metadata_is_rejected(self):
        bad = [dict(dtype='BF16', shape=[3], data_offsets=[0, 4]),
               dict(dtype='BF16', shape=[True], data_offsets=[0, 2]),
               dict(dtype='unknown', shape=[2], data_offsets=[0, 4]),
               dict(dtype='BF16', shape=[-1], data_offsets=[0, 4]),
               dict(dtype='BF16', shape=[2], data_offsets=[1, 5]),
               dict(dtype='BF16', shape=[2], data_offsets=[True, 4])]
        for record in bad:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as directory:
                p = self.write(directory, {'p': record})
                with self.assertRaises(ValueError):
                    read_header(p, p.stat().st_size, check=lambda: None)

    def test_duplicate_overlap_hole_and_wrong_size(self):
        records = [b'{"p":{},"p":{}}',
                   {'p': dict(dtype='BF16', shape=[1], data_offsets=[0, 2])},
                   {'p': dict(dtype='BF16', shape=[1], data_offsets=[2, 4])},
                   {'p': dict(dtype='BF16', shape=[2], data_offsets=[0, 4]),
                    'q': dict(dtype='BF16', shape=[1], data_offsets=[0, 2])}]
        for header in records:
            with self.subTest(header=header), tempfile.TemporaryDirectory() as directory:
                p = self.write(directory, header)
                with self.assertRaises(ValueError):
                    read_header(p, p.stat().st_size, check=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            p = self.write(directory, {'p': dict(dtype='BF16', shape=[2], data_offsets=[0, 4])})
            with self.assertRaises(ValueError):
                read_header(p, p.stat().st_size+1, check=lambda: None)


if __name__ == '__main__':
    unittest.main()
