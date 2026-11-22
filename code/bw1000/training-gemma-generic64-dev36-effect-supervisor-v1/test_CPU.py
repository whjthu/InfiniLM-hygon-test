"""New operational wiring, true local stdlib subprocess and synthetic Slurm/device fixtures."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import contract as c
import controller
import keeper
import launch
import read_terminal
import seal

D=Path(__file__).resolve().parent

class Operations(unittest.TestCase):
    def test_real_nested_package_adoption_ingress_and_mismatch_rejection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for p in D.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    dest=root/p.relative_to(D);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
            spec=importlib.util.spec_from_file_location('fixture_seal',root/'seal.py');sm=importlib.util.module_from_spec(spec);spec.loader.exec_module(sm)
            sm.seal(adopt=True)
            spec=importlib.util.spec_from_file_location('fixture_contract',root/'contract.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertEqual(module.verify('GPU'),module.pin(root/'SOURCE_FILES.json'))
            value=json.loads((root/'core/CLAIM.json').read_bytes());value['GPU_GO']=False
            (root/'core/CLAIM.json').write_text(json.dumps(value),encoding='utf8');sm.seal()
            with self.assertRaisesRegex(ValueError,'flag mismatch'):module.verify('GPU')
        with self.assertRaisesRegex(ValueError,'operative outer flags'):c.verify('GPU')

    def test_single_DCU_step_argv_and_real_accounting_not_srun_boolean(self):
        for phase in ('pre','gpu','post','read'):
            argv=c.step_argv(phase,['fixture'],60)
            self.assertIn('--gres='+('dcu:1' if phase=='gpu' else 'none'),argv)
            self.assertIn('-c'+('8' if phase=='gpu' else '2'),argv)
            row=['1103606.123',c.LABEL+'-'+phase,'COMPLETED','0:0','3',c.NODE,
                 'cpu=8,mem=64000M,node=1,gres/dcu=1' if phase=='gpu' else 'cpu=2,mem=4G,node=1']
            held=dict(reaped=True,termination_needed=False);status=dict(row=row,terminal_visible=True)
            self.assertTrue(controller.phase_passed(phase,0,None,held,status))
            self.assertFalse(controller.phase_passed(phase,0,None,held,dict(row=None,terminal_visible=False)))
            bad=list(row);bad[2]='FAILED';self.assertFalse(controller.phase_passed(phase,0,None,held,dict(row=bad,terminal_visible=True)))
            if phase=='gpu':
                bad=list(row);bad[6]=bad[6].replace('dcu=1','dcu=8')
                self.assertFalse(controller.phase_passed(phase,0,None,held,dict(row=bad,terminal_visible=True)))
        with self.assertRaises(ValueError):c.parse_accounting('1103606.1|truncated')
        with self.assertRaises(ValueError):c.step_argv('gpu',['fixture'],59)

    def test_actual_eight_physical_to_one_assigned_admission_and_bad_paths(self):
        cards=[f'0000:{i:02x}:00.0' for i in range(8)]
        before=dict(request_id=c.REQUEST,step='1103606.123',pci=cards,boot_id='fixture',
            kfd_holder_count=0,full_fd_inventory_performed=True,monotonic_ns=100)
        current=dict(pci=cards,boot_id='fixture');device=dict(pci=cards[2],gpu_id=123)
        keeper.validate_device(before,current,cards[2],device,1000,'1103606.123','2')
        for changes in (dict(kfd_holder_count=1),dict(full_fd_inventory_performed=False),dict(step='1103606.124')):
            bad=dict(before,**changes)
            with self.assertRaises(ValueError):keeper.validate_device(bad,current,cards[2],device,1000,'1103606.123','2')
        with self.assertRaises(ValueError):keeper.validate_device(before,current,'0000:ff:00.0',device,1000,'1103606.123','2')
        with self.assertRaises(ValueError):keeper.validate_device(before,current,cards[2],device,46_000_000_100,'1103606.123','2')
        with self.assertRaises(ValueError):keeper.validate_device(before,current,cards[2],device,1000,'1103606.123','0,1')

    def test_deadlines_include_original_epoch_and_actual_Slurm_floor(self):
        native,cleanup=keeper.work_deadlines(100,130,660,140)
        self.assertEqual((native,cleanup),(735,785))
        self.assertLess(native+5,cleanup);self.assertLess(cleanup,100+900)
        for args in ((100,130,660,751),(100,99,660,140),(100,130,780,140),(float('nan'),130,660,140)):
            with self.assertRaises(ValueError):keeper.work_deadlines(*args)
        self.assertEqual(c.WHOLE,900)

    def test_original_owner_absence_requires_same_host_boot_startticks(self):
        owner=dict(PID=123,startticks=456,host='fixture',boot_id='boot')
        self.assertEqual(read_terminal.original_absent(owner,'fixture','boot',lambda p:dict(startticks=999,state='S')),
                         'ORIGINAL_IDENTITY_ABSENT_PID_REUSED')
        with self.assertRaises(ValueError):read_terminal.original_absent(owner,'other','boot',lambda p:dict(startticks=456,state='Z'))
        with self.assertRaises(ValueError):read_terminal.original_absent(owner,'fixture','boot',lambda p:dict(startticks=456,state='S'))
        def denied(pid):raise PermissionError('fixture')
        with self.assertRaises(PermissionError):read_terminal.original_absent(owner,'fixture','boot',denied)

    def test_real_local_GO_bootstrap_and_injected_transport_only(self):
        with TemporaryDirectory() as tmp:
            marker=Path(tmp)/'actual-child.txt'
            code='from pathlib import Path;Path('+repr(str(marker))+').write_text("ACTUAL_LOCAL_CHILD")'
            # Linux executes the exact real os.execv. Windows _P_OVERLAY does
            # not preserve a waitable subprocess handle; emulate only that
            # final handoff with an actual waited stdlib CPU child in fixture.
            bridge=('import os,subprocess; os.execv=lambda path,args: (_ for _ in ()).throw(SystemExit(subprocess.run(args).returncode));'
                    if os.name=='nt' else '')
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',bridge+keeper.BOOTSTRAP,'-I','-S','-B','-c',code],
                input=b'GO\n',capture_output=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(marker.read_text(),'ACTUAL_LOCAL_CHILD')
            bad=subprocess.run([sys.executable,'-I','-S','-B','-c',bridge+keeper.BOOTSTRAP,'-c','raise RuntimeError("never")'],
                input=b'NO\n',capture_output=True,timeout=5)
            self.assertNotEqual(bad.returncode,0)
            with self.assertRaises(ValueError):launch.launch(transport_start=None,publication='a'*40,local_output=Path(tmp)/'never')
            self.assertFalse((Path(tmp)/'never').exists())

    def test_POSIX_nested_delivery_and_fixed32_ids_exact_runtime_helpers(self):
        body=launch.make_body('a'*40,100,dict(bytes=1,SHA256='a'*64))
        tree=ast.parse(body);root=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='remote_root' for t in n.targets))
        self.assertTrue(root.startswith('/public/home/'));self.assertNotIn('\\',root)
        self.assertIn(b'core/infer_dev36.py',body);self.assertIn(b'p.parent.mkdir(',body)
        self.assertLess(body.index(b'intent.json'),body.index(b'os.execv('))
        import sys as _sys
        _sys.path.insert(0,str(D/'core'));import effect_contract
        for value in (c.DISPATCH_REQUEST,c.REQUEST,effect_contract.READ_REQUEST,effect_contract.SCORE_REQUEST):
            self.assertEqual(len(value),32);self.assertFalse(set(value)-set('0123456789abcdef'))
        old=D.parent/'training-gemma-generic64-control25-inputs-v1'
        for name in ('owned_processes.py','held_child.py','rss_monitor.py','slurm_accounting.py','snapshot.py','devices.py'):
            self.assertEqual((D/name).read_bytes(),(old/name).read_bytes())
        t09=D.parent/'training-hunyuanvideo-cached-native25-target-v3'
        for name in ('runtime_pci.py','assigned_kfd.py'):self.assertEqual((D/name).read_bytes(),(t09/name).read_bytes())

    def test_clean_delivered_operational_import_and_python310_AST(self):
        with TemporaryDirectory() as tmp:
            for p in D.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    target=Path(tmp)/p.relative_to(D);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',
                "import sys;sys.path.insert(0,sys.argv[1]);import contract,controller,gpu_entry,keeper,read_terminal,launch;assert 'torch' not in sys.modules;print('CLEAN_OPERATIONAL_IMPORT_OK')",tmp],
                capture_output=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
        for p in D.rglob('*.py'):ast.parse(p.read_bytes(),feature_version=(3,10))

    def test_transport255_and_callback_failure_are_unknown_without_replay(self):
        self.assertEqual(launch.transport_state(255,dict(reaped=True)),'UNKNOWN_NO_REPLAY')
        self.assertEqual(launch.transport_state(0,dict(reaped=False)),'UNKNOWN_NO_REPLAY')
        self.assertEqual(launch.transport_state(0,dict(reaped=True)),'RETURNED_REQUIRES_ORIGINAL_TERMINAL_CONSUMPTION')
        with TemporaryDirectory() as tmp:
            def refused(context):raise RuntimeError('fixture callback failure, actual network0')
            with patch.object(launch,'verify',return_value=dict(bytes=1,SHA256='a'*64)),patch.object(launch,'make_body',return_value=b'fixture_not_executed'):
                value=launch.launch(transport_start=refused,publication='a'*40,local_output=Path(tmp)/'new')
            self.assertTrue(value['transport_callback_called']);self.assertFalse(value['actual_Popen_returned'])
            self.assertEqual(value['execution_state'],'UNKNOWN_NO_REPLAY')
            self.assertEqual(value['failure']['execution_state'],'UNKNOWN_NO_REPLAY')

if __name__=='__main__':unittest.main()
