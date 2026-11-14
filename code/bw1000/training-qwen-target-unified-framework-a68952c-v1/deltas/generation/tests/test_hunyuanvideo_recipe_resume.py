"""Test the unified HunyuanVideo recovery path without loading a model."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[2] / 'tools'
sys.path.insert(0, str(TOOLS))
import hygon_training_recipe as recipe


class HunyuanRecipeResumeTest(unittest.TestCase):
    def test_canonical_sft_resume_preserves_schedule(self):
        frozen = {'config': {'warmup_steps': 5, 'max_steps': 20, 'checkpoint_steps': [10, 20]},
                  'weights': '/original/weights', 'source': '/original/hunyuanvideo-1.5'}
        raw = json.dumps(frozen).encode()
        registration = (hashlib.sha256(raw).hexdigest(), 'hunyuanvideo-1.5', 'train', 1)
        with tempfile.TemporaryDirectory() as root:
            launch = Path(root) / 'launch.json'
            launch.write_bytes(raw)
            assets = Path(root) / 'assets.json'
            assets.write_text(json.dumps({frozen['weights']: '/target/weights', frozen['source']: '/target/hunyuanvideo-1.5'}))
            args = ['T09.sft', '--a100-launch', str(launch), '--asset-map', str(assets), '--run']
            with patch.dict(recipe.GENERATION, {'T09.sft': registration}):
                with patch.object(recipe, 'run_generation', return_value=0) as run:
                    self.assertEqual(recipe.main(args + ['--resume-from-checkpoint', '/checkpoints/state.pt']), 0)
                    plan, _ = run.call_args.args
                    self.assertEqual(plan['config'], dict(frozen['config'], resume_from_checkpoint='/checkpoints/state.pt'))
                    self.assertEqual(recipe.main(args), 0)
                    self.assertEqual(run.call_args.args[0]['config'], frozen['config'])
                with patch.object(recipe, 'run_generation') as run, contextlib.redirect_stderr(io.StringIO()):
                    missing_assets = ['T09.sft', '--a100-launch', str(launch), '--run',
                                      '--resume-from-checkpoint', '/checkpoints/state.pt']
                    self.assertEqual(recipe.main(missing_assets), 2)
                    run.assert_not_called()
                launch.write_bytes(raw + b' ')
                with patch.object(recipe, 'run_generation') as run, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(recipe.main(args + ['--resume-from-checkpoint', '/checkpoints/state.pt']), 2)
                    run.assert_not_called()

    def test_dp2_recovery_reaches_distributed_backend_with_frozen_topology(self):
        config = dict(parallel_mode="data_parallel", world_size=2, sp=1, warmup_steps=5,
                      max_steps=20, checkpoint_steps=[10, 20])
        frozen = dict(config=config, weights="/original/weights", source="/original/hunyuanvideo-1.5")
        raw = json.dumps(frozen).encode()
        registration = (hashlib.sha256(raw).hexdigest(), "hunyuanvideo-1.5", "train", 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launch = root / "launch.json"
            launch.write_bytes(raw)
            assets = root / "assets.json"
            assets.write_text(json.dumps({frozen["weights"]: "/target/weights",
                                          frozen["source"]: "/target/hunyuanvideo-1.5"}))
            args = ["T09.dp2", "--a100-launch", str(launch), "--asset-map", str(assets), "--run",
                    "--source-root", "/target", "--output-root", "/experiments", "--run-name", "dp2-resumed",
                    "--lock-root", "/locks", "--lock-host", "assigned-host", "--cards", "0,1",
                    "--resume-from-checkpoint", "/previous/checkpoint-20"]
            sys.path.insert(0, str(TOOLS.parent / "generation" / "src"))
            from gencuda import cli

            def capture(options):
                self.assertEqual(options.accelerator, "hip")
                self.assertEqual(options.cards, "0,1")
                self.assertEqual(json.loads(Path(options.config).read_bytes()),
                                 dict(config, resume_from_checkpoint="/previous/checkpoint-20"))
                return 0

            with patch.dict(recipe.GENERATION, {"T09.dp2": registration}):
                with patch.object(cli, "cmd_run", side_effect=capture) as launch_backend:
                    self.assertEqual(recipe.main(args), 0)
                    launch_backend.assert_called_once()
                    args[args.index("0,1")] = "0"
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(recipe.main(args), 2)
                    launch_backend.assert_called_once()

    def test_only_hunyuan_sft_or_dp2_execution_accepts_generation_resume_flag(self):
        for args in (['T09.sft'], ['T09.dp2'], ['T09.sage-grpo', '--run'],
                     ['T09.sft', '--run', '--resume-from-checkpoint', 'relative/state.pt']):
            supplied = args if '--resume-from-checkpoint' in args else args + ['--resume-from-checkpoint', '/checkpoints/state.pt']
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    recipe.main(supplied)
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
