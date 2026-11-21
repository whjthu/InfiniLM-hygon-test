"""Exercise the generated package, immutable staging, and terminal cleanup guard."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch_once
from render_target import BASE, LABEL, REQUEST, REUSED, digest, render


class TargetPackageTests(unittest.TestCase):
    def test_reused_files_actual_call_and_native_cpu_before_gpu(self):
        bundle = render()
        self.assertEqual(len(bundle), 18)
        for name in REUSED:
            self.assertEqual(bundle[name], (BASE/name).read_bytes())
        campaign = bundle['campaign.py'].decode()
        self.assertLess(campaign.index("check=run('-CPU-contract'"), campaign.index("before=snap('pre')"))
        self.assertLess(campaign.index("assert check['passed']"), campaign.index("gpu=run('',True"))
        self.assertLess(campaign.index("assert gpu['terminal']['terminal_visible']"), campaign.index("after=snap('post')"))
        worker = bundle['run_platform.py'].decode()
        self.assertIn('evidence=replay_once', worker)
        self.assertIn("evidence['actual_adjoint_calls']==140", worker)
        conditional = [node for node in ast.walk(ast.parse(campaign)) if isinstance(node, ast.If)
                       and ast.unparse(node.test) == 'is_terminal(one_row(LABEL))']
        self.assertEqual(len(conditional), 1)
        self.assertIn("after = snap('failure-post')", ast.unparse(conditional[0]))

    def test_remote_bootstrap_stages_one_namespace_and_passes_bound_epoch(self):
        bundle = render()
        with TemporaryDirectory() as directory:
            target = str(Path(directory)/'new-original-run')
            with patch.object(launch_once, 'TARGET', target):
                script, pin = launch_once.bootstrap('b'*40, bundle)
            output = StringIO()
            with patch('subprocess.Popen', return_value=SimpleNamespace(pid=0)) as run, redirect_stdout(output):
                exec(compile(script, '<target-bootstrap>', 'exec'), {})
                with self.assertRaises(FileExistsError):
                    exec(compile(script, '<target-bootstrap>', 'exec'), {})
            stored = json.loads(output.getvalue())
            self.assertEqual(stored['manifest_SHA256'], pin)
            self.assertEqual(stored['request'], REQUEST)
            self.assertEqual(run.call_count, 1)
            argv = run.call_args.args[0]
            self.assertEqual(argv[-3:], [str(Path(target)/'campaign.py'), str(stored['epoch']), REQUEST])
            self.assertEqual((Path(target)/'hygon_sdxl_first_step.py').read_bytes(), bundle['hygon_sdxl_first_step.py'])

    def test_existing_local_attempt_blocks_second_transport_and_keeps_unknown(self):
        pin = digest(render()['SOURCE_FILES.json'])
        calls = []

        def transport(script, out, stderr):
            calls.append(digest(script))
            return SimpleNamespace(returncode=255)

        with TemporaryDirectory() as directory:
            path = Path(directory)/LABEL
            kwargs = dict(local=path, check_publication=lambda _: pin)
            result = launch_once.launch('b'*40, REQUEST, transport, **kwargs)
            with self.assertRaises(FileExistsError):
                launch_once.launch('b'*40, REQUEST, transport, **kwargs)
            intent = json.loads((path/'intent.json').read_bytes())
            self.assertEqual([intent['body_SHA256']], calls)
            self.assertEqual(result['remote_outcome'], 'UNKNOWN')
            self.assertFalse(result['replay_allowed'])

    def test_wrong_published_target_digest_fails_before_intent(self):
        with TemporaryDirectory() as directory, patch('subprocess.Popen') as transport:
            path = Path(directory)/LABEL
            with self.assertRaisesRegex(AssertionError, 'published generated package differs'):
                launch_once.launch('b'*40, REQUEST, transport, local=path, check_publication=lambda _: '0'*64)
            self.assertFalse(path.exists())
            transport.assert_not_called()


if __name__ == '__main__':
    unittest.main()
