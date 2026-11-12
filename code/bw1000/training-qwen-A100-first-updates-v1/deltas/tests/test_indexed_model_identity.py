"""CPU fixtures reject malformed indexed weights before any model import."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.hygon_indexed_model_identity import header_layout, inspect_indexed_model, stat_identity


class IndexedIdentityTest(unittest.TestCase):
    def fixture(self, root):
        files = {}
        for name, tensor in [('one.safetensors', 'a'), ('two.safetensors', 'b')]:
            header = json.dumps({tensor: dict(dtype='BF16', shape=[2, 3], data_offsets=[0, 12])}).encode()
            raw = len(header).to_bytes(8, 'little')+header+bytes(range(12))
            (root/name).write_bytes(raw)
            files[name] = raw
        raw = json.dumps(dict(metadata={'total_size': 24}, weight_map={'a': 'one.safetensors', 'b': 'two.safetensors'})).encode()
        (root/'index.json').write_bytes(raw)
        return files, hashlib.sha256(raw).hexdigest()

    def test_complete_read_and_header_index_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);files, pin = self.fixture(root);events = []
            result = inspect_indexed_model(root, 'index.json', pin, {n: stat_identity(root/n) for n in files}, check_deadline=lambda: None, progress=events.append)
            self.assertEqual(result['tensors'], 2)
            self.assertEqual(result['full_file_bytes_read'], sum(map(len, files.values())))
            self.assertEqual(result['tensor_payload_bytes'], 24)
            for name, raw in files.items():
                self.assertEqual(result['shards'][name]['SHA256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(events[-1]['completed_shards'], ['one.safetensors', 'two.safetensors'])

    def test_shape_dtype_holes_overlap_and_trailing_bytes_rejected(self):
        good = {'x': dict(dtype='F32', shape=[2], data_offsets=[0, 8])}
        for changed in (
                {'x': dict(dtype='BF16', shape=[2], data_offsets=[0, 8])},
                {'x': dict(dtype='UNKNOWN', shape=[2], data_offsets=[0, 8])},
                {'x': dict(dtype='F32', shape=[-2], data_offsets=[0, 8])},
                {'x': dict(dtype='F32', shape=[1], data_offsets=[4, 8])},
                dict(good, y=dict(dtype='F32', shape=[1], data_offsets=[4, 8]))):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                header_layout(json.dumps(changed).encode(), 8)
        with self.assertRaises(ValueError):
            header_layout(json.dumps(good).encode(), 12)

    def test_duplicate_header_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            header_layout(b'{"x":{},"x":{}}', 8)

    def test_wrong_hash_shard_set_stat_and_index_tensor_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);files, pin = self.fixture(root)
            original = {n: stat_identity(root/n) for n in files}
            for kind in ('hash', 'set', 'stat', 'mapping'):
                stats = {n: dict(v) for n, v in original.items()};digest = pin
                if kind == 'hash':
                    digest = '0'*64
                elif kind == 'set':
                    stats.pop('two.safetensors')
                elif kind == 'stat':
                    stats['one.safetensors']['bytes'] += 1
                else:
                    raw = json.dumps(dict(weight_map={'wrong': 'one.safetensors', 'b': 'two.safetensors'})).encode()
                    (root/'index.json').write_bytes(raw);digest = hashlib.sha256(raw).hexdigest()
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    inspect_indexed_model(root, 'index.json', digest, stats, check_deadline=lambda: None, progress=lambda _: None)

    def test_deadline_stops_with_partial_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);files, pin = self.fixture(root);events = []
            def deadline():
                if events and events[-1].get('completed_shards') == ['one.safetensors']:
                    raise TimeoutError('fixture deadline')
            with self.assertRaises(TimeoutError):
                inspect_indexed_model(root, 'index.json', pin, {n: stat_identity(root/n) for n in files}, check_deadline=deadline, progress=events.append)
            self.assertEqual(events[-1]['completed_shards'], ['one.safetensors'])

    def test_flat_paths_and_required_progress_deadline_rejected(self):
        for name in ('../index.json', '/index.json', 'a\\index.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                inspect_indexed_model('.', name, 'a'*64, {'x': {}}, check_deadline=lambda: None, progress=lambda _: None)
        with self.assertRaises(ValueError):
            inspect_indexed_model('.', 'index.json', 'a'*64, {}, check_deadline=lambda: None, progress=None)


if __name__ == '__main__':
    unittest.main()
