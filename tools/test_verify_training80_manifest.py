from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from verify_training80_manifest import verify_package_members


class ManifestTest(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.repo = Path(root.name)
        self.directory = self.repo / 'code/bw1000/fixture'
        self.directory.mkdir(parents=True)
        self.launcher = self.pinned('launch.py', b'print("fixture")\n')
        self.claim = self.pinned('claim.json', b'{}\n')

    def pinned(self, name, raw):
        file = self.directory / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(raw)
        return dict(path=file.relative_to(self.repo).as_posix(), SHA256=sha256(raw).hexdigest(), bytes=len(raw))

    def manifest(self, value):
        raw = json.dumps(value).encode() if not isinstance(value, bytes) else value
        return self.pinned('SOURCE_FILES.json', raw)

    def test_size_and_sha_manifest_verifies_required_member_pins(self):
        nested = self.pinned('src/module.py', b'NAME = "fixture"\n')
        members = {'launch.py': self.launcher, 'claim.json': self.claim, 'src/module.py': nested}
        pins = {name: {key: value for key, value in row.items() if key != 'path'} for name, row in members.items()}
        manifest = self.manifest(dict(files=pins))
        report = verify_package_members(self.repo, manifest, required_files=[self.launcher, self.claim])
        self.assertEqual(report['member_count'], 3)
        self.assertEqual(report['member_bytes'], sum(pin['bytes'] for pin in members.values()))

    def test_compiler_hash_only_manifest(self):
        manifest = self.manifest(dict(files_SHA256={'launch.py': self.launcher['SHA256']}))
        report = verify_package_members(self.repo, manifest, required_files=[self.launcher])
        self.assertEqual(report['member_count'], 1)

    def test_member_change_with_unchanged_manifest_is_rejected(self):
        manifest = self.manifest(dict(files={'launch.py': dict(SHA256=self.launcher['SHA256'])}))
        self.pinned('launch.py', b'print("changed")\n')
        with self.assertRaisesRegex(ValueError, 'source member bytes differ'):
            verify_package_members(self.repo, manifest)

    def test_required_launcher_missing_from_manifest_is_rejected(self):
        manifest = self.manifest(dict(files_SHA256={'claim.json': self.claim['SHA256']}))
        with self.assertRaisesRegex(ValueError, 'required source reference'):
            verify_package_members(self.repo, manifest, required_files=[self.launcher])

    def test_duplicate_json_key_is_rejected(self):
        digest = self.launcher['SHA256']
        manifest = self.manifest(('{' + f'"files_SHA256":{{"launch.py":"{digest}","launch.py":"{digest}"}}' + '}').encode())
        with self.assertRaisesRegex(ValueError, 'duplicate source manifest key'):
            verify_package_members(self.repo, manifest)

    def test_traversal_and_alternate_path_syntax_are_rejected(self):
        for unsafe in ('../other', 'src/../claim.json', 'C:/other', 'launch.py:extra', './launch.py', 'src\\mod.py'):
            with self.subTest(unsafe=unsafe):
                manifest = self.manifest(dict(files_SHA256={unsafe: self.launcher['SHA256']}))
                with self.assertRaisesRegex(ValueError, 'normalized relative path'):
                    verify_package_members(self.repo, manifest)

    def test_manifest_digest_and_size_are_bound(self):
        manifest = self.manifest(dict(files_SHA256={'launch.py': self.launcher['SHA256']}))
        with self.assertRaisesRegex(ValueError, 'source manifest SHA256 differs'):
            verify_package_members(self.repo, dict(manifest, SHA256='a' * 64))
        with self.assertRaisesRegex(ValueError, 'source manifest byte count differs'):
            verify_package_members(self.repo, dict(manifest, bytes=manifest['bytes']+1))


if __name__ == '__main__':
    unittest.main()
