from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import preflight_sdxl_style18_target as preflight


class SDXLTargetPreflightTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve(strict=True)
        self.train = self.root / 'train'
        self.train.mkdir()
        self.backend = self.root / 'backend.py'
        self.backend.write_bytes(b'# fixture common backend')
        self.rules = SimpleNamespace(TRAINER_RELATIVE='examples/fixture.py', BINDING_SHA256='a' * 64,
                                    BACKEND_SHA256={sha256(self.backend.read_bytes()).hexdigest()},
                                    bind_config=self.config)
        self.source = self.root / 'target-source'
        self.trainer = self.source / self.rules.TRAINER_RELATIVE
        self.trainer.parent.mkdir(parents=True)

    @staticmethod
    def config(path):
        return dict(train_data_dir=str(path), resolution=256, lora_rank=4, max_train_steps=25,
                    train_batch_size=1, warmup_steps=5, extra_checkpoint_steps='20,25')

    def write_trainer(self, missing=()):
        flags = preflight.REQUIRED_FLAGS.difference(missing)
        source = '\n'.join(f'parser.add_argument({flag!r})' for flag in sorted(flags))
        data = ('raise RuntimeError("the deployed script must not execute")\n' + source).encode()
        self.trainer.write_bytes(data)
        return sha256(data).hexdigest()

    def run_probe(self, expected=None):
        return preflight.inspect(self.rules, self.train, self.source, self.backend, expected)

    def test_missing_train_or_trainer_keeps_distinct_states(self):
        self.assertEqual(self.run_probe()['state'], 'DEPLOYED_TRAINER_MISSING')
        self.train.rmdir()
        self.assertEqual(self.run_probe()['state'], 'TARGET_TRAIN_DIRECTORY_MISSING')

    def test_missing_train_retains_deployed_source_observation(self):
        self.train.rmdir()
        pin = self.write_trainer()
        observation = self.run_probe()
        self.assertEqual(observation['state'], 'TARGET_TRAIN_DIRECTORY_MISSING')
        self.assertEqual(observation['deployed_source_state'], 'DEPLOYED_TRAINER_PIN_PENDING')
        self.assertEqual(observation['observed_deployed_trainer_SHA256'], pin)
        self.assertEqual(observation['observed_common_backend_SHA256'], sha256(self.backend.read_bytes()).hexdigest())
        result = self.run_probe(pin)
        self.assertEqual(result['state'], 'TARGET_TRAIN_DIRECTORY_MISSING')
        self.assertEqual(result['deployed_source_state'], 'DEPLOYED_SOURCE_VERIFIED_NATIVE_IO_PENDING')
        self.assertFalse(result['train_inputs_verified'])
        self.assertEqual(result['training_ready_added'], 0)

    def test_observation_cannot_self_admit_or_accept_the_A100_pin(self):
        sha = self.write_trainer()
        observation = self.run_probe()
        self.assertEqual(observation['state'], 'DEPLOYED_TRAINER_PIN_PENDING')
        self.assertEqual(observation['observed_deployed_trainer_SHA256'], sha)
        mismatch = self.run_probe(preflight.ORIGINAL_A100_SHA)
        self.assertEqual(mismatch['state'], 'DEPLOYED_TRAINER_PIN_MISMATCH')
        self.assertEqual(mismatch['training_ready_added'], 0)

    def test_verified_target_identity_and_static_flags_do_not_run_the_trainer(self):
        result = self.run_probe(self.write_trainer())
        self.assertEqual(result['state'], 'TRAIN_ONLY_DATA_AND_DEPLOYED_SOURCE_VERIFIED_NATIVE_IO_PENDING')
        self.assertEqual(result['declared_common_flags_checked'], len(preflight.REQUIRED_FLAGS))
        self.assertEqual(result['native_training_subprocesses'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(result['actual_GPU_steps'], 0)
        self.assertFalse(result['native_tokenizer_VAE_verified'])

    def test_target_trainer_cannot_omit_the_common_checkpoint_contract(self):
        flag = '--gencuda_checkpoint_steps'
        result = self.run_probe(self.write_trainer(missing=[flag]))
        self.assertEqual(result['state'], 'DEPLOYED_TRAINER_CLI_INCOMPLETE')
        self.assertEqual(result['missing_frozen_flags'], [flag])

    def test_common_backend_also_requires_original_bytes(self):
        observed = self.write_trainer()
        self.backend.write_bytes(b'reformatted fixture')
        self.assertEqual(self.run_probe(observed)['state'], 'DEPLOYED_COMMON_BACKEND_PIN_MISMATCH')


if __name__ == '__main__':
    unittest.main()
