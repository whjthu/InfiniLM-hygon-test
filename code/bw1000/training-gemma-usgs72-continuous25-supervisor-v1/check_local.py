"""Narrow stdlib integration checks; no Slurm, Torch, media or model reads."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import socket
from unittest.mock import patch

import contract
from contract import D,REQUEST,Clock,need,parse_accounting,step_argv,allocated
from controller import container_command
from launch import launch,make_body
import rss_monitor as rss
from supervisor import BOOTSTRAP,native_command


class Wiring(unittest.TestCase):
    def test_helper_exact_source_reuse(self):
        repo=D.parents[2]
        source=json.loads((D/'SOURCE_REUSE.json').read_bytes())
        for name in ('held_child.py','owned_processes.py','slurm_accounting.py','snapshot.py'):
            self.assertEqual((D/name).read_bytes(),(repo/source[name]['path']).read_bytes())
        old=ast.parse((repo/source['rss_functions']['path']).read_bytes())
        new=ast.parse((D/'rss_monitor.py').read_bytes())
        for name in source['rss_functions']['functions']:
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)

    def test_accounting_does_not_hide_bad_rows(self):
        valid='1103606.900|label|COMPLETED|0:0|20|f13r1n12|cpu=2,mem=4096M'
        self.assertEqual(len(parse_accounting(valid)),1)
        for bad in (valid+'\nmalformed',valid+'\n1103606|only2',valid+'|extra'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):parse_accounting(bad)
        self.assertEqual(allocated('cpu=64,gres/dcu=8,gres/dcu:K100=8'),8)
        self.assertEqual(allocated('cpu=2,mem=4G'),0)
        with self.assertRaises(ValueError):allocated('cpu=64,gres/dcu=8,gres/dcu:K100=7')

    def test_actual_slurm_and_original_clock(self):
        self.assertEqual(step_argv('gpu',['python'],2100)[-4:],['-t','35','--job-name='+contract.LABEL+'-gpu','python'])
        self.assertIn('35',step_argv('gpu',['python'],2100))
        for seconds in (float('nan'),59,2101):
            with self.assertRaises(ValueError):step_argv('gpu',['python'],seconds)
        with patch('contract.time.time',return_value=100.0),patch('contract.time.monotonic',return_value=10.):
            clock=Clock(90.)
            self.assertEqual(clock.left(),2390.)
        with patch('contract.time.time',return_value=2490.),patch('contract.time.monotonic',return_value=2400.):
            with self.assertRaises(ValueError):clock.left()

    def test_explicit_transport_required_before_dispatch(self):
        with self.assertRaises(ValueError):
            launch(transport_start=None,publication='a'*40,local_output=D/'MUST_NOT_BE_CREATED')
        self.assertFalse((D/'MUST_NOT_BE_CREATED').exists())

    def test_receiver_and_clean_I_S_only_delivered_imports(self):
        body=make_body('a'*40,100.,dict(bytes=1,SHA256='b'*64))
        ast.parse(body)
        self.assertIn(b'"-I","-S","-B","-c",entry',body)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for src in D.glob('*.py'):shutil.copyfile(src,p/src.name)
            code='import sys,runpy;sys.path.insert(0,'+repr(str(p))+');'
            code+=';'.join('runpy.run_path('+repr(str(p/name))+',run_name="source_only")' for name in
                          ('controller.py','supervisor.py','devices.py','read_terminal.py','launch.py'))
            code+=';assert "torch" not in sys.modules;print("CLEAN_IMPORT_PASS")'
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',code],capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            self.assertEqual(result.stdout.strip(),b'CLEAN_IMPORT_PASS')

    def test_real_held_bootstrap_GO_and_bad_token(self):
        code=BOOTSTRAP
        if os.name=='nt':
            # Windows CRT execv does not preserve this Linux target pipe
            # observation. Intercept only the fixture's final syscall; execute
            # the real token-read code and check exact exec argv.
            code=('import os,sys\n'
                'def capture(executable,args):\n'
                ' assert executable==sys.executable and args==[sys.executable,"-c",\'print("HELD_FIXTURE_EXECUTED")\']\n'
                ' print("HELD_FIXTURE_EXECUTED",flush=True)\n'
                ' raise SystemExit(0)\n'
                'os.execv=capture\nexec('+repr(BOOTSTRAP)+')')
        cmd=[sys.executable,'-B','-c',code,'-c','print("HELD_FIXTURE_EXECUTED")']
        good=subprocess.run(cmd,input=b'GO\n',capture_output=True,timeout=5,start_new_session=True)
        self.assertEqual(good.returncode,0,good.stderr.decode())
        self.assertEqual(good.stdout.strip(),b'HELD_FIXTURE_EXECUTED')
        bad=subprocess.run(cmd,input=b'NO\n',capture_output=True,timeout=5,start_new_session=True)
        self.assertNotEqual(bad.returncode,0)
        self.assertEqual(bad.stdout,b'')

    def test_RSS_actual_monitor_wiring_preserves_unknown(self):
        saved=dict(PID=111,startticks=222)
        measured=lambda _:dict(PID=111,startticks=222,classification='MEASURED',RSS_bytes=1024)
        resource=dict(samples=0,sampled_group_peak_RSS_bytes=0,soft_limit_exceeded=False)
        self.assertTrue(rss.sample_group_rss({111:saved},resource,reader=measured))
        self.assertEqual(resource['last_sample']['group_RSS_bytes'],1024)
        unknown=lambda _:dict(PID=111,startticks=222,classification='LIVE_MISSING_RSS',RSS_bytes=None)
        self.assertTrue(rss.sample_group_rss({111:saved},resource,reader=unknown))
        self.assertIsNone(resource['last_sample']['group_RSS_bytes'])
        self.assertEqual(resource['RSS_unknown_coverage_samples'],1)
        with self.assertRaises(ValueError):rss.sample_group_rss({111:saved},resource,reader=unknown)

    def test_container_and_native_scope(self):
        command=container_command();text=' '.join(command)
        self.assertEqual(command[0],'/public/software/apps/apptainer/1.3.4/bin/apptainer')
        self.assertIn('--rocm',command)
        self.assertNotIn('initial.pt',text)
        self.assertNotIn('original-initial-trainables',text)
        self.assertIn(str(contract.CORE),text)
        self.assertIn('/minicpm-transformers:/swift-runtime/site:',text)
        args=[b'--output_dir',str(contract.ASSET/'continuous/train').encode(),b'--callbacks',b'gemma_domain_continuous_witness']
        self.assertTrue(native_command(args,None))
        args[1]=b'/somebody/else/train'
        self.assertFalse(native_command(args,None))

    def test_terminal_pre_native_failure_is_retained(self):
        import read_terminal as reader
        manifest=dict(bytes=1,SHA256='b'*64)
        intent=dict(publication='a'*40,epoch=100.,request_id=REQUEST,source_manifest=manifest)
        out=dict(actual_updates=None,error=dict(type='ValueError',reason='fixture prior pre-admission failure'))
        def original(path,expected=None,cap=None):
            return (intent,dict(bytes=1,SHA256='c'*64)) if path.name=='intent.json' else (out,dict(bytes=2,SHA256='d'*64))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(reader,'ROOT',root),patch.object(reader,'NODE',socket.gethostname().split('.')[0]),\
                 patch.object(reader,'verify',lambda _:manifest),patch.object(reader,'stable_json',original),\
                 patch.object(contract,'ROOT',root):
                value=reader.consume('a'*40,100.,'b'*64)
            self.assertFalse(value['passed'])
            self.assertEqual(value['planned_updates'],25)
            self.assertEqual(value['controller_failure'],out['error'])
            self.assertTrue((root/'independent-terminal.json').exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Wiring))
    receipt=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        Torch_imported='torch' in sys.modules,model_body_reads=0,native_encode_calls=0,
        Slurm_calls=0,GPU_calls=0,optimizer_updates=0,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')})
    receipt['Windows_bootstrap_fixture_final_execv_intercepted']=os.name=='nt'
    print(json.dumps(receipt,sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
