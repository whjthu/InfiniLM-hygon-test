"""New v2 receiver tests only; no old RSS suite, remote, Slurm or model."""
import ast
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import launch
from contract import D,ROOT,LABEL,REQUEST,DISPATCH_REQUEST,CORE,parse_step_membership
from controller import container_command


class Repair(unittest.TestCase):
    def test_actual_Windows_built_body_and_bootstrap_order(self):
        body=launch.make_body('a'*40,100.,dict(bytes=1,SHA256='b'*64))
        tree=ast.parse(body)
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='remote_root' for t in n.targets))
        remote=ast.literal_eval(node.value)
        self.assertEqual(remote,ROOT.as_posix())
        self.assertTrue(PurePosixPath(remote).is_absolute())
        self.assertNotIn('\\',remote)
        self.assertEqual(PurePosixPath(remote).name,LABEL)
        NativePath=type(D)
        with tempfile.TemporaryDirectory() as tmp:
            target=NativePath(tmp)/'target'
            calls=[]
            def target_path(value):
                if value==remote:
                    calls.append(value)
                    return target
                self.assertFalse(PurePosixPath(str(value)).is_absolute())
                return NativePath(value)
            def final_exec(executable,args):
                self.assertEqual(args[:5],[sys.executable,'-I','-S','-B','-c'])
                data=json.loads((target/'intent.json').read_bytes())
                self.assertEqual(data['request_id'],REQUEST)
                self.assertEqual(data['dispatch_request_id'],DISPATCH_REQUEST)
                self.assertNotEqual(REQUEST,DISPATCH_REQUEST)
                self.assertTrue((target/'src/controller.py').exists())
                self.assertEqual((target/'src/contract.py').read_bytes(),(D/'contract.py').read_bytes())
                entry=args[5]
                # Execute the actual generated -I-S entry in a fresh interpreter,
                # intercepting only __main__ dispatch so no controller/Slurm runs.
                code=('import runpy;original=runpy.run_path;'
                    'runpy.run_path=lambda p,run_name:original(p,run_name="source_only");'
                    'exec('+repr(entry)+');import sys;assert "torch" not in sys.modules;print("TARGET_ENTRY_IMPORT_PASS")')
                proc=subprocess.run([sys.executable,'-I','-S','-B','-c',code],capture_output=True,timeout=10)
                self.assertEqual(proc.returncode,0,proc.stderr.decode())
                self.assertEqual(proc.stdout.strip(),b'TARGET_ENTRY_IMPORT_PASS')
            with patch('pathlib.Path',target_path),patch('os.execv',final_exec):
                exec(compile(body,'Windows-generated-target-receiver','exec'),{'__name__':'receiver_fixture'})
            self.assertEqual(calls,[remote])
            self.assertEqual(sorted(p.name for p in NativePath(tmp).iterdir()),['target'])

    def test_noncanonical_root_rejected_before_any_payload(self):
        for root in (Path('relative'),Path('/public/home/../evil')):
            with self.subTest(root=str(root)),patch.object(launch,'ROOT',root),self.assertRaises(ValueError):
                launch.make_body('a'*40,100.,{})

    def test_bash_and_step_columns(self):
        command=container_command()
        index=command.index('-c')
        self.assertEqual(command[index+2],'gemma-continuous25')
        self.assertEqual(parse_step_membership('1103606.batch|f13r1n12\n1103606.7|f13r1n[10-12]'),
                         [('1103606.7','f13r1n[10-12]')])
        for text in ('1103606.7|n|extra','1103606.7','1.2|n','1103606.7|(null)'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_step_membership(text)
        self.assertTrue(CORE.as_posix().endswith('/incoming/training-gemma-usgs72-fresh-initial-domain25-v1-source'))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Repair))
    print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        actual_Windows=os.name=='nt',Torch_imported='torch' in sys.modules,Slurm_calls=0,GPU_calls=0,actual_updates=0,
        old_v1_modified=False,stage_restarted=False,source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
