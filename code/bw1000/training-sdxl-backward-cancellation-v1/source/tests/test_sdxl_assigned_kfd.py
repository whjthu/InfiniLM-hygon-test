"""Single-card topology permission boundaries, with local filesystem fixtures."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from hygon_sdxl_assigned_kfd import assigned_kfd


class AssignedKFDTest(unittest.TestCase):
    def fixture(self, root):
        for index, minor, location in ((0, 0, 0), (1, 128, 2304), (2, 129, 13824)):
            node = root / str(index)
            node.mkdir()
            (node / 'properties').write_text(
                f'drm_render_minor {minor}\ndomain 0\nlocation_id {location}\n')
            (node / 'gpu_id').write_text(str(7000 + index if index else 0))
        return dict(nodes=root, pci_lookup=lambda render: {'renderD128': '0000:09:00.0',
                                                         'renderD129': '0000:36:00.0'}[render])

    def test_does_not_read_unassigned_gpu_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            options = self.fixture(root)
            original = Path.read_text
            identities = []
            def read(path, *a, **kw):
                if path.name == 'gpu_id':
                    identities.append(path.parent.name)
                    if path.parent.name != '1':
                        raise PermissionError('unassigned device')
                return original(path, *a, **kw)
            with patch.object(Path, 'read_text', read):
                proof = assigned_kfd('0000:09:00.0', **options)
            self.assertEqual(identities, ['1'])
            self.assertEqual(proof['gpu_id'], 7001)
            self.assertFalse(proof['other_devices_gpu_id_read'])

    def test_unassigned_properties_may_be_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            options = self.fixture(root)
            original = Path.read_text
            def read(path, *a, **kw):
                if path.parent.name == '2':
                    raise PermissionError('unassigned topology')
                return original(path, *a, **kw)
            with patch.object(Path, 'read_text', read):
                proof = assigned_kfd('0000:09:00.0', **options)
            self.assertEqual(proof['denied_unassigned_properties'], ['2'])

    def test_assigned_permission_error_is_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            options = self.fixture(root)
            original = Path.read_text
            def read(path, *a, **kw):
                if path.parent.name == '1' and path.name == 'gpu_id':
                    raise PermissionError('assigned identity unavailable')
                return original(path, *a, **kw)
            with patch.object(Path, 'read_text', read), self.assertRaises(PermissionError):
                assigned_kfd('0000:09:00.0', **options)

    def test_invalid_or_foreign_pci_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            options = self.fixture(Path(d))
            for pci in ('not-a-pci', '0000:55:00.0', None):
                with self.subTest(pci=pci), self.assertRaises(ValueError):
                    assigned_kfd(pci, **options)

    def test_zero_gpu_and_wrong_mapping_are_rejected(self):
        for field, value in (('gpu_id', '0'), ('properties',
                             'drm_render_minor 128\ndomain 0\nlocation_id 0\n')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                options = self.fixture(root)
                (root / '1' / field).write_text(value)
                with self.assertRaises(ValueError):
                    assigned_kfd('0000:09:00.0', **options)

    def test_duplicate_matching_topology_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            options = self.fixture(root)
            (root / '2/properties').write_text((root / '1/properties').read_text())
            with self.assertRaises(ValueError):
                assigned_kfd('0000:09:00.0', **options)


if __name__ == '__main__':
    unittest.main()
