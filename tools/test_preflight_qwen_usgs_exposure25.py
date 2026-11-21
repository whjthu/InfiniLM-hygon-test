import json
from hashlib import sha256
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import preflight_qwen_usgs_exposure25 as qwen
import reconcile_qwen_usgs_dev80_original as dev


PARENT = b'1103606|original-parent|RUNNING|0:0|10|parent-nodes|cpu=2560,node=40,gres/dcu=320'
CPU = f'1103606.490|{qwen.CPU_LABEL}|COMPLETED|0:0|9|{qwen.NODE}|cpu=8,node=1'.encode()
LIVE = PARENT+b'\n'+CPU


class QwenPreflightTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.verified_assets = []

    def membership(self, repo):
        self.assertEqual(repo, qwen.REPO)
        return dict(observed_epoch=123.5, node_membership={node: 'NO_NUMBERED_STEP' for node in dev.NODES.values()})

    def check_assets(self, base, models, tables):
        self.assertEqual(base, self.base)
        self.assertEqual(len(models['model']), 14)
        self.assertEqual(len(tables), 22)
        self.verified_assets.append(True)
        return dict(framework_files=393, train_rows=88, train_media_files=22, model_weight_body_bytes_read=0)

    def check(self, rows):
        remaining = iter(rows)
        result = qwen.preflight(qwen.REPO, self.base, account=lambda: next(remaining), membership=self.membership,
                                verify=self.check_assets)
        json.dumps(result, allow_nan=False)
        return result

    def test_original_package_and_two_namespaces_are_bound(self):
        result = self.check([LIVE] * 4)
        self.assertEqual(result['state'], 'TRAINING_ASSETS_VERIFIED_DEVICE_PENDING')
        self.assertEqual(result['request'], qwen.REQUEST)
        self.assertEqual(result['predecessor']['request'], dev.REQUEST)
        self.assertEqual(result['required_port'], 29918)
        self.assertEqual(result['assets']['train_media_files'], 22)
        self.assertEqual(self.verified_assets, [True])
        self.assertFalse(result['device_admission'])
        self.assertFalse(result['training_READY'])
        self.assertEqual(list(self.base.iterdir()), [])

    def test_predecessor_marker_prevents_asset_verification(self):
        (self.base/dev.LABEL).mkdir()
        result = self.check([LIVE, LIVE])
        self.assertEqual(result['state'], 'DEV80_RECONCILIATION_REQUIRED')
        self.assertFalse(self.verified_assets)
        self.assertFalse((self.base/qwen.LABEL).exists())

    def test_late_original_phase_for_either_scope_stops_admission(self):
        for label in (qwen.LABEL+'-gpu', dev.LABEL+'-base-gpu'):
            with self.subTest(label=label):
                row = f'1103606.512|{label}|RUNNING|0:0|8|{qwen.NODE}|cpu=64,node=1,gres/dcu=8'.encode()
                result = self.check([LIVE, LIVE, LIVE, LIVE+b'\n'+row])
                self.assertEqual(result['state'], 'ORIGINAL_REQUEST_RECONCILIATION_REQUIRED')
                self.assertFalse(result['last']['clear'])
                self.assertFalse(result['training_READY'])

    def test_missing_bad_and_duplicate_original_cpu_steps_are_rejected(self):
        for last in (PARENT, LIVE.replace(b'COMPLETED', b'FAILED'), LIVE+b'\n'+CPU,
                     LIVE.replace(b'cpu=8,node=1', b'cpu=8,node=1,gres/dcu=8')):
            with self.subTest(last=last):
                with self.assertRaisesRegex(ValueError, 'original CPU input step'):
                    self.check([LIVE, LIVE, LIVE, last])

    def test_model_check_reads_metadata_only_and_binds_size(self):
        shard = self.base/'shard'
        shard.write_bytes(b'example')
        with patch.object(Path, 'open', side_effect=AssertionError('body must remain unread')):
            qwen.model_metadata(shard, 7)
            with self.assertRaisesRegex(ValueError, 'metadata differs'):
                qwen.model_metadata(shard, 8)

    def test_asset_contract_reads_training_only_and_requires_exact_media(self):
        def put(name, body):
            path = self.base/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            return dict(SHA256=sha256(body).hexdigest(), bytes=len(body), mode=stat.S_IMODE(path.stat().st_mode))

        cpu = put(Path(qwen.CPU_LABEL)/'step.log', b'CPU input')
        files = {f'src/{i}.py': put(qwen.BASE/'framework'/f'src/{i}.py', str(i).encode()) for i in range(393)}
        manifest = put(qwen.BASE/'source/FRAMEWORK_FILES.json', json.dumps({'files': files}).encode())
        initial_path = qwen.INITIAL/'initial.safetensors'
        initial = put(initial_path, b'never read initial body')
        binding = put(qwen.BASE/'source/QWEN_TARGET_NATIVE_BINDING.json', json.dumps({
            'initial_state': dict(initial, path=str(self.base/initial_path))}).encode())
        original = put(qwen.BASE/'source/original-launch.json', b'{}')
        train = put(qwen.DATA/'data/train/swift.jsonl', b'{}\n'*88)
        stage = {'request': '72e8a7066f73425cbf5a7e0921baf09c', 'staged': True, 'blind_and_dev_gold_uploaded': False,
                 'output_identity_SHA256': 'd7c5cfc9b2b00982ebd25fb03c341807fcc272e7f989886c33118316bd8edd8b'}
        put(qwen.DATA/'stage-receipt.json', json.dumps(stage).encode())
        media_names = [f'table-{i:03d}.png' for i in range(22)]
        tables = {name: put(qwen.DATA/'data/train/media'/name, f'train-media-{i}'.encode())
                  for i, name in enumerate(media_names)}
        model = {f'model-{i}.safetensors': put(qwen.MODEL/f'model-{i}.safetensors', b'never read shard body')
                 for i in range(1, 5)}
        model['tokenizer.json'] = put(qwen.MODEL/'tokenizer.json', b'{}')
        models = {'model': model, 'staged_source': {'original-launch.json': original['SHA256'],
                                                   'QWEN_TARGET_NATIVE_BINDING.json': binding['SHA256']}}
        original_open = Path.open

        def protected_open(path, *args, **kwargs):
            self.assertFalse(path.name.endswith('.safetensors'))
            self.assertNotIn('dev', path.parts)
            self.assertNotIn('blind', path.parts)
            return original_open(path, *args, **kwargs)

        with patch.multiple(qwen, CPU_LOG_SHA=cpu['SHA256'], FRAMEWORK_SHA=manifest['SHA256'], TRAIN_SHA=train['SHA256']), \
                patch.object(Path, 'open', protected_open):
            facts = qwen.assets(self.base, models, tables)
            self.assertEqual((facts['framework_files'], facts['train_rows'], facts['train_media_files']), (393, 88, 22))
            self.assertEqual(facts['model_shard_metadata'], 4)
            put(qwen.DATA/'data/train/media'/media_names[-1], b'wrong')
            with self.assertRaisesRegex(ValueError, 'SHA-256 differs'):
                qwen.assets(self.base, models, tables)


if __name__ == '__main__':
    unittest.main()
