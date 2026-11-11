"""Check that diagnostic observers preserve the frozen VLM recipe and native callback."""
import copy
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import hygon_vlm_recipe as recipe


class RecipeDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.original = {
            "command": ["/source/swift", "sft", "--model", "/source/MiniCPM-V-4.6",
                        "--dataset", "/source/coco64.jsonl", "--output_dir", "/source/run",
                        "--max_steps", "25", "--gradient_accumulation_steps", "4",
                        "--warmup_steps", "5", "--external_plugins", "/source/native.py",
                        "--callbacks", "native", "--attn_impl", "sdpa"],
            "environment": {"PATH": "/source/bin", "NPROC_PER_NODE": "8",
                            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
        }
        self.raw = json.dumps(self.original).encode()
        self.patch = patch.dict(recipe.A100_LAUNCH_SHA256,
                                {"T05.sft": hashlib.sha256(self.raw).hexdigest()})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def build(self, **kwargs):
        return recipe.build_launch("T05.sft", self.raw,
                                   *[PurePosixPath(x) for x in
                                     ("/runtime/swift", "/models", "/input/coco.jsonl",
                                      "/plugins", "/output/train")], 29873, **kwargs)

    def test_normal_recipe_remains_without_an_observer(self):
        row = self.build()
        self.assertIsNone(row["diagnostic_observer"])
        self.assertEqual(row["required_assets"]["plugins"], ["/plugins/native.py"])
        self.assertEqual(row["command"][-4:], ["--callbacks", "native", "--attn_impl", "sdpa"])

    def test_diagnostic_adds_only_its_plugin_and_callback(self):
        baseline = self.build()
        row = self.build(diagnostic_plugin=PurePosixPath("/diagnostics/observe.py"),
                         diagnostic_callback="capture_updates")
        reconstructed = copy.deepcopy(row["command"])
        reconstructed.remove("/diagnostics/observe.py")
        reconstructed.remove("capture_updates")
        self.assertEqual(reconstructed, baseline["command"])
        self.assertEqual(row["environment_overrides"], baseline["environment_overrides"])
        self.assertFalse(row["diagnostic_observer"]["formal_performance"])
        self.assertEqual(row["required_assets"]["plugins"],
                         ["/plugins/native.py", "/diagnostics/observe.py"])
        self.assertEqual(json.loads(self.raw), self.original)

    def test_incomplete_or_duplicate_diagnostic_is_rejected(self):
        invalid = [dict(diagnostic_callback="capture"),
                   dict(diagnostic_plugin=PurePosixPath("/observe.py")),
                   dict(diagnostic_plugin=PurePosixPath("/observe.py"), diagnostic_callback="native"),
                   dict(diagnostic_plugin=PurePosixPath("/observe.py"), diagnostic_callback="invalid-callback")]
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.build(**case)

    def test_ordered_initialization_and_first_update_observers(self):
        baseline = self.build()
        row = self.build(diagnostic_plugin=[PurePosixPath('/observe/initial.py'), PurePosixPath('/observe/updates.py')],
                         diagnostic_callback=['native_lora_dropout_alignment', 'original_first_updates'])
        argv = row['command']
        i = argv.index('--callbacks')
        self.assertEqual(argv[i + 1:i + 4], ['native', 'native_lora_dropout_alignment', 'original_first_updates'])
        for token in ['/observe/initial.py', '/observe/updates.py', 'native_lora_dropout_alignment', 'original_first_updates']:
            argv.remove(token)
        self.assertEqual(argv, baseline['command'])
        self.assertEqual(row['environment_overrides'], baseline['environment_overrides'])
        self.assertEqual([x['callback'] for x in row['diagnostic_observer']['observers']],
                         ['native_lora_dropout_alignment', 'original_first_updates'])
        for plugins, callbacks in [([], []), ([PurePosixPath('/a.py')], ['a', 'b']),
                ([PurePosixPath('/a.py'), PurePosixPath('/a.py')], ['a', 'b']),
                ([PurePosixPath('/a.py'), PurePosixPath('/b.py')], ['a', 'a'])]:
            with self.subTest(plugins=plugins), self.assertRaises(ValueError):
                self.build(diagnostic_plugin=plugins, diagnostic_callback=callbacks)

    def test_repeated_cli_options_keep_the_requested_observer_order(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'launch.json';path.write_bytes(self.raw)
            argv=['T05.sft','--a100-launch',str(path),'--swift','/runtime/swift',
                  '--models','/models','--dataset','/data/input.jsonl','--plugins','/plugins',
                  '--output','/new/train','--master-port','29873',
                  '--diagnostic-plugin','/observe/initial.py','--diagnostic-callback','initialize',
                  '--diagnostic-plugin','/observe/update.py','--diagnostic-callback','record_updates']
            output=io.StringIO()
            with contextlib.redirect_stdout(output):self.assertEqual(recipe.main(argv),0)
            plan=json.loads(output.getvalue());i=plan['command'].index('--callbacks')
            self.assertEqual(plan['command'][i+1:i+4],['native','initialize','record_updates'])
            self.assertEqual([x['callback'] for x in plan['diagnostic_observer']['observers']],
                             ['initialize','record_updates'])

    def test_initialization_before_the_baseline_digest_through_actual_cli(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'launch.json';path.write_bytes(self.raw)
            argv=['T05.sft','--a100-launch',str(path),'--swift','/runtime/swift',
                  '--models','/models','--dataset','/input/coco.jsonl','--plugins','/plugins',
                  '--output','/output/train','--master-port','29873',
                  '--diagnostic-plugin','/observe/initial.py',
                  '--diagnostic-callback','native_lora_dropout_alignment',
                  '--diagnostic-plugin','/observe/updates.py',
                  '--diagnostic-callback','original_first_updates',
                  '--diagnostic-callback-position','before']
            output=io.StringIO()
            with contextlib.redirect_stdout(output):self.assertEqual(recipe.main(argv),0)
            plan=json.loads(output.getvalue());i=plan['command'].index('--callbacks')
            self.assertEqual(plan['command'][i+1:i+4],
                ['native_lora_dropout_alignment','original_first_updates','native'])
            self.assertEqual(plan['diagnostic_callback_position'],'before')
            reconstructed=plan['command'].copy()
            for token in ('/observe/initial.py','/observe/updates.py',
                          'native_lora_dropout_alignment','original_first_updates'):
                reconstructed.remove(token)
            self.assertEqual(reconstructed,self.build()['command'])
            self.assertEqual(plan['environment_overrides'],self.build()['environment_overrides'])
        for kwargs in ({'diagnostic_callback_position':'before'},
                       {'diagnostic_callback_position':'invalid'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.build(**kwargs)

    def test_original_launch_identity_is_still_required(self):
        changed = self.raw + b" "
        with self.assertRaisesRegex(ValueError, "SHA256"):
            recipe.build_launch("T05.sft", changed,
                                *[PurePosixPath("/x") for _ in range(5)], 29873,
                                diagnostic_plugin=PurePosixPath("/observe.py"),
                                diagnostic_callback="capture")

    def test_resume_replaces_fresh_callback_and_preserves_full_schedule(self):
        self.original['command'][self.original['command'].index('--callbacks') + 1] = 'nvidia_baseline'
        self.raw = json.dumps(self.original).encode()
        recipe.A100_LAUNCH_SHA256['T05.sft'] = hashlib.sha256(self.raw).hexdigest()
        base = self.build()
        row = self.build(resume_from_checkpoint=PurePosixPath('/original/checkpoint-20'),
                         diagnostic_plugin=PurePosixPath('/observe/resume.py'),
                         diagnostic_callback='native_resume_witness')
        argv = row['command'].copy()
        self.assertNotIn('nvidia_baseline', argv[argv.index('--callbacks') + 1:])
        self.assertEqual(argv[-8:], ['--resume_from_checkpoint', '/original/checkpoint-20',
                                    '--resume_only_model', 'false', '--ignore_data_skip', 'false',
                                    '--load_args', 'false'])
        argv = argv[:-8]
        argv.remove('/observe/resume.py')
        argv[argv.index('native_resume_witness')] = 'nvidia_baseline'
        self.assertEqual(argv, base['command'])
        self.assertEqual(row['environment_overrides'], base['environment_overrides'])
        for checkpoint in ('/output', '/output/train', '/output/train/checkpoint-20'):
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                self.build(resume_from_checkpoint=PurePosixPath(checkpoint),
                           diagnostic_plugin=PurePosixPath('/observe/resume.py'),
                           diagnostic_callback='native_resume_witness')
        with self.assertRaises(ValueError):
            self.build(resume_from_checkpoint=PurePosixPath('/original/checkpoint-20'))

    def test_resume_manifest_checked_before_runtime_or_model_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint-20'
            checkpoint.mkdir()
            names = ['trainer_state.json', 'args.json', 'adapter_model.safetensors',
                     'adapter_config.json', 'optimizer.pt', 'scheduler.pt']
            names += ['rng_state_%d.pth' % i for i in range(8)]
            pins = {}
            for name in names:
                raw = ('fixture-' + name).encode()
                (checkpoint / name).write_bytes(raw)
                pins[name] = dict(bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
            config = root / 'config.json'
            raw = json.dumps(dict(checkpoint=str(checkpoint), checkpoint_files=pins)).encode()
            config.write_bytes(raw)
            plan = dict(environment_overrides={'OUTPUT_DIR': str(root / 'new' / 'train')},
                        resume_from_checkpoint=str(checkpoint))
            env = dict(NATIVE_RESUME_CONFIG=str(config),
                       NATIVE_RESUME_CONFIG_SHA256=hashlib.sha256(raw).hexdigest())
            recipe.validate_diagnostic_outputs(plan, env)
            (checkpoint / 'optimizer.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                recipe.validate_diagnostic_outputs(plan, env)



if __name__ == "__main__":
    unittest.main()
