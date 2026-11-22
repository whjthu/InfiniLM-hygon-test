"""Real frozen metadata/plan source plus bounded synthetic I/O. No model/remote."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest
from unittest.mock import patch

import control_contract as c
import control_plan as p
import stage
from control_initial import bounded_load

D=Path(__file__).resolve().parent


class Contract(unittest.TestCase):
    def test_actual_original67_and_processor14_subset(self):
        raw=(D/'ORIGINAL_INPUT_FILES.json').read_bytes()
        self.assertEqual(c.sha(raw),c.SOURCE_INPUT_MANIFEST_SHA)
        items=json.loads(raw)['files']
        self.assertEqual(len(items),67)
        self.assertEqual(sum(x['bytes'] for x in items),10964794)
        self.assertEqual(sum(x['name'].endswith('.jpg') for x in items),64)
        original=json.loads((D/'ORIGINAL_PROCESSOR_FILES.json').read_bytes())['files']
        model=json.loads((D/'MODEL_BINDING.json').read_bytes())['files']
        self.assertEqual((len(original),len(model)),(14,16))
        for name,pin in original.items():
            self.assertEqual({k:pin[k] for k in ('bytes','SHA256')},{k:model[name][k] for k in ('bytes','SHA256')})

    def test_real_successful_command_plan_and_posix_target(self):
        plan=p.make_plan((D/'ORIGINAL_PLAN.json').read_bytes(),c.ROOT.as_posix())
        cmd=plan['command'];value=lambda key:cmd[cmd.index(key)+1]
        self.assertEqual(value('--dataset'),c.ROOT.as_posix()+'/view/train800-swift.jsonl')
        self.assertEqual(value('--output_dir'),c.ROOT.as_posix()+'/continuous/train')
        self.assertEqual(value('--model_type'),'gemma3_vision')
        self.assertEqual(value('--gradient_checkpointing'),'false')
        self.assertEqual(value('--max_length'),'512')
        self.assertEqual(value('--gradient_accumulation_steps'),'4')
        self.assertEqual(plan['environment_overrides']['NPROC_PER_NODE'],'8')
        self.assertEqual(plan['environment_overrides']['MASTER_PORT'],'29944')
        self.assertNotIn('--resume_from_checkpoint',cmd)
        self.assertNotIn('gemma_domain_continuous_witness',cmd)

    def test_source_derivation_preserves_update_loss_and_checkpoint_methods(self):
        old=(D/'DOMAIN_CALLBACK_SOURCE.py').read_bytes()
        new=p.derive_callback(old,'a'*64)
        before,after=p.source_functions(old),p.source_functions(new)
        for name in before:
            if name not in ('before','capture_fresh_initial'):self.assertEqual(before[name],after[name])
        self.assertIn(b'gemma_generic_continuous_witness',new)
        self.assertIn(b'restore_target_initial',new)
        self.assertNotIn(b'torch.save(',new)
        with self.assertRaises(ValueError):p.derive_callback(old+b'\n','a'*64)
        with self.assertRaises(ValueError):p.derive_callback(old,'not-a-digest')

    def test_frozen800_schedule_retains_raw64_lines_synthetic_only(self):
        lines=[c.canonical(dict(messages=[dict(role='user',content='fixture'),dict(role='assistant',content='caption')],
            images=[f'fixture-{i}.jpg'])) for i in range(64)]
        raw=b''.join(lines)
        with self.assertRaises(ValueError):stage.make_view(raw)
        with patch.object(stage,'INPUT_SHA',c.sha(raw)),patch.object(stage,'INPUT_BYTES',len(raw)):
            view,ledger=stage.make_view(raw)
            self.assertEqual(view,b''.join(lines[i%64] for i in range(800)))
            self.assertEqual(ledger['exposure_counts'],[13]*32+[12]*32)
            self.assertTrue(ledger['original_line_bytes_retained'])
            self.assertFalse(ledger['dev_or_blind_read'])

    def test_same_fd_weights_only_CPU_load_and_mutation_rejection(self):
        with TemporaryDirectory() as tmp:
            f=Path(tmp)/'synthetic.pt';raw=b'not-a-real-tensor-fixture';f.write_bytes(raw)
            expected=dict(bytes=len(raw),SHA256=c.sha(raw))
            calls=[]
            class FakeTorch:
                @staticmethod
                def load(stream,**kwargs):
                    calls.append(kwargs);self.assertEqual(stream.tell(),0);return {'fixture':True}
            self.assertEqual(bounded_load(FakeTorch,f,expected,lambda:None),{'fixture':True})
            self.assertEqual(calls,[dict(map_location='cpu',weights_only=True)])
            class Changed:
                @staticmethod
                def load(stream,**kwargs):
                    f.write_bytes(raw+b'x');return None
            with self.assertRaises(ValueError):bounded_load(Changed,f,expected,lambda:None)

    def test_clock_rejection_and_clean_only_delivered_imports(self):
        self.assertEqual(c.clock(100,310,200,210),110)
        for args in ((100,310,99,210),(100,310,311,210),(100,311,200,210),(float('nan'),310,200,210)):
            with self.assertRaises(ValueError):c.clock(*args)
        with TemporaryDirectory() as tmp:
            for name in ('control_contract.py','control_plan.py','control_initial.py','stage.py','worker.py'):
                (Path(tmp)/name).write_bytes((D/name).read_bytes())
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',
                "import sys;sys.path.insert(0,sys.argv[1]);import stage,worker,control_initial;assert 'torch' not in sys.modules;print('CLEAN_STDLIB_IMPORT_OK')",tmp],
                capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'CLEAN_STDLIB_IMPORT_OK')
        for path in D.glob('*.py'):ast.parse(path.read_bytes(),feature_version=(3,10))

    def test_operational_posexplicit_transport_and_frozen_ids(self):
        import contract,launch
        for value in (c.TRAINING_REQUEST,c.STAGE_REQUEST,contract.DISPATCH_REQUEST):
            self.assertEqual(len(value),32);self.assertEqual(set(value)-set('0123456789abcdef'),set())
        self.assertEqual(len({c.TRAINING_REQUEST,c.STAGE_REQUEST,contract.DISPATCH_REQUEST}),3)
        claim = json.loads((D / 'CLAIM.json').read_bytes())
        if all(claim[key] is True for key in ('root_GO', 'GPU_GO', 'training_GO')):
            self.assertEqual(contract.verify('GPU'), contract.pin(D / 'SOURCE_FILES.json'))
        with TemporaryDirectory() as tmp:
            denied = Path(tmp)
            claim['GPU_GO'] = False
            (denied / 'CLAIM.json').write_text(json.dumps(claim), encoding='utf-8')
            manifest = {'files': {'CLAIM.json': contract.pin(denied / 'CLAIM.json')}}
            (denied / 'SOURCE_FILES.json').write_text(json.dumps(manifest), encoding='utf-8')
            with patch.object(contract, 'D', denied), self.assertRaisesRegex(ValueError, 'authorization false'):
                contract.verify('GPU')
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                launch.launch(transport_start=None,publication='a'*40,local_output=Path(tmp)/'must-not-create')
            self.assertFalse((Path(tmp)/'must-not-create').exists())
        body=launch.make_body('a'*40,100,dict(bytes=1,SHA256='a'*64))
        tree=ast.parse(body)
        root,=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and
               any(isinstance(t,ast.Name) and t.id=='remote_root' for t in n.targets)]
        self.assertEqual(root,contract.ROOT.as_posix())
        self.assertTrue(root.startswith('/public/home/'));self.assertNotIn('\\',root)
        self.assertIn(b'os.execv(',body)

    def test_stage_accounting_and_compute_device_identity_not_ignored(self):
        import controller,contract,worker
        row=['1103606.999',contract.STAGE_LABEL,'COMPLETED','0:0','25',c.NODE,'cpu=2,mem=8G,node=1']
        self.assertEqual(controller.require_stage_accounting(dict(step=row[0]),[row]),row)
        for changes in ((2,'FAILED'),(5,'wrong-node'),(6,'cpu=2,mem=8G,gres/dcu=1,node=1')):
            wrong=list(row);wrong[changes[0]]=changes[1]
            with self.assertRaises(ValueError):controller.require_stage_accounting(dict(step=row[0]),[wrong])
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'scoped-file';path.write_bytes(b'x')
            good=dict(reads={str(path):dict(file_identity=c.identity(path.stat()))})
            worker.check_stage_identity(good)
            wrong=json.loads(json.dumps(good));wrong['reads'][str(path)]['file_identity'][0]+=1
            with self.assertRaises(ValueError):worker.check_stage_identity(wrong)

    def test_format_and_content_are_independent_full36_metrics(self):
        from content_vs_format import measure,ARMS
        labels=[dict(id=f'synthetic-{i}',chart_id='fixture',group='fixture',kind='depth_change',
                     event_ids=['E01','E02'],answer_type='decimal2',answer='0.00',data_status='QUALIFIED') for i in range(36)]
        def rows(text):return [dict(id=q['id'],text=text,status='COMPLETED',natural_EOS=True,cap_hit=False) for q in labels]
        value=measure(labels,dict(base=rows('0.00'),domain25=rows('FINAL: 1.00'),generic25=rows('FINAL: 0.00')))
        self.assertEqual(value['arms']['base']['content_correct'],36)
        self.assertEqual(value['arms']['base']['strict_FINAL'],0)
        self.assertEqual(value['arms']['domain25']['strict_FINAL'],36)
        self.assertEqual(value['arms']['domain25']['content_correct'],0)
        self.assertEqual(value['domain_vs_generic']['both_parsed_loss'],36)
        self.assertFalse(value['semantic_advantage_or_promotion_inferred'])
        missing=measure(labels,{arm:[] for arm in ARMS})
        self.assertTrue(all(r['denominator']==36 and r['missing']==36 for r in missing['arms'].values()))

    def test_qualified_cleanup_helpers_are_exact_not_reimplemented(self):
        old=D.parents[2]/'code/bw1000/training-gemma-usgs72-continuous25-supervisor-v2'
        for name in ('owned_processes.py','held_child.py','rss_monitor.py','slurm_accounting.py','devices.py'):
            self.assertEqual((D/name).read_bytes(),(old/name).read_bytes())
        import supervisor
        self.assertIn("gemma_generic_continuous_witness",(D/'supervisor.py').read_text(encoding='utf8'))
        self.assertNotIn("b'gemma_domain_continuous_witness'",(D/'supervisor.py').read_text(encoding='utf8'))


if __name__=='__main__':unittest.main()
