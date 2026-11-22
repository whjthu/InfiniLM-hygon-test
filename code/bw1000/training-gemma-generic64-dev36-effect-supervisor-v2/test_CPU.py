"""Only the v2 target-set repair and new scope/package wiring; no old native tests."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'core'))
from target_contract import signature,validate_recipe,PREFIX,SUFFIX,OBSERVED_ADAPTER_CONFIG_SHA256
import effect_contract as effect
import contract as op

class Repair(unittest.TestCase):
    def test_actual_recorded_two_checkpoint_order_only_positive_and_no_mutation(self):
        prior=json.loads((D/'PRIOR_FAILURE.json').read_bytes())
        evidence=prior['adapter_config_evidence'];reference=json.loads((D/'core/STRUCTURAL_BINDING.json').read_bytes())
        self.assertEqual(evidence['expected_target'],reference['target_modules'])
        for arm in ('domain25','generic25'):
            config=evidence['checkpoints'][arm];before=copy.deepcopy(config)
            self.assertEqual(config['SHA256'],OBSERVED_ADAPTER_CONFIG_SHA256)
            self.assertEqual(config['bytes'],1177)
            self.assertNotEqual(config['target_modules'],reference['target_modules'])
            value=validate_recipe(config,reference)
            self.assertEqual(value['unique_projection_alternatives'],7)
            self.assertFalse(value['config_modified']);self.assertEqual(config,before)
            self.assertEqual(signature(config['target_modules']),signature(reference['target_modules']))
        self.assertFalse(prior['engineering_completed']);self.assertEqual(prior['actual_predictions'],0)
        self.assertFalse(prior['dev_gold_read'])

    def test_exact_shell_only_and_unique_seven_alternatives_reject_widening(self):
        actual=json.loads((D/'PRIOR_FAILURE.json').read_bytes())['adapter_config_evidence']['checkpoints']['domain25']['target_modules']
        parts=actual[len(PREFIX):-len(SUFFIX)].split('|')
        self.assertEqual(signature(PREFIX+'|'.join(reversed(parts))+SUFFIX),signature(actual))
        bad=[PREFIX+'|'.join(parts[:-1])+SUFFIX,PREFIX+'|'.join(parts+['v_proj'])+SUFFIX,
             PREFIX+'|'.join(parts[:-1]+[parts[0]])+SUFFIX,PREFIX+'|'.join(parts[:-1]+['other_proj'])+SUFFIX,
             PREFIX+'|'.join(parts)+'|.*'+SUFFIX,actual.replace('(?=\\.)',''),actual+'|.*',actual[1:],actual[:-1],
             actual.replace('model\\.language_model','model.*'),actual.replace('(q_proj|','(?:q_proj|'),['q_proj'],None]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):signature(value)

    def test_original_numeric_recipe_and_expected_target_remain_strict(self):
        prior=json.loads((D/'PRIOR_FAILURE.json').read_bytes())['adapter_config_evidence']
        reference=dict(target_modules=prior['expected_target']);base=prior['checkpoints']['domain25']
        for key,value in (('r',8),('lora_alpha',16),('lora_dropout',0),('bias','all')):
            config=dict(base);config[key]=value
            with self.assertRaises(ValueError):validate_recipe(config,reference)
        with self.assertRaises(ValueError):validate_recipe(base,dict(target_modules='.*'))

    def test_new_four32_ids_namespaces_and_actual_old_checkpoints_unchanged(self):
        values=(effect.REQUEST,effect.READ_REQUEST,effect.SCORE_REQUEST,op.DISPATCH_REQUEST)
        self.assertEqual(len(set(values)),4)
        old={'4d6aeb103c484796b740d4d3e4afe7ac','d3bade6cc7d9415ba5c1bcae777dfe1e',
             '91c4c857ab694548930643a4f37849e9','0e64c73af67f4575adf29abce76edba7'}
        self.assertTrue(set(values).isdisjoint(old))
        for value in values:self.assertEqual(len(value),32);self.assertFalse(set(value)-set('0123456789abcdef'))
        self.assertEqual(effect.ROOT.name,'training-gemma-generic64-dev36-effect-v2')
        self.assertEqual(op.ROOT.name,'training-gemma-generic64-dev36-effect-supervisor-v2')
        self.assertEqual(op.ASSET.as_posix(),effect.ROOT.as_posix());self.assertEqual(op.PORT,29946)
        self.assertEqual(effect.DOMAIN.name,'training-gemma-usgs72-fresh-initial-domain25-v1')
        self.assertEqual(effect.GENERIC.name,'training-gemma-generic64-control25-v1')
        self.assertEqual(effect.GENERIC_GATE['SHA256'],'261fc4ebbd519bb5126243440c3f65691d5b3a795331c320ba37ce5b75d94184')
        self.assertEqual(effect.CAP,256);self.assertEqual(op.WHOLE,900)
        with self.assertRaisesRegex(ValueError,'operative outer flags'):op.verify('GPU')

    def test_functional_delta_is_only_recipe_guard_and_scope_bindings(self):
        old=D.parent/'training-gemma-generic64-dev36-effect-supervisor-v1'
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)}
        before=functions(old/'core/infer_dev36.py');after=functions(D/'core/infer_dev36.py')
        self.assertEqual(before.keys(),after.keys())
        for name in before:
            if name!='generate':self.assertEqual(before[name],after[name])
        for name in ('core/score_after.py','core/read_terminal.py','core/bind_ready.py','core/frozen_score.py',
                     'core/response_protocol.py','core/dev_inputs.py','core/STRUCTURAL_BINDING.json','core/MODEL_BINDING.json',
                     'controller.py','gpu_entry.py','keeper.py','owned_processes.py','held_child.py','rss_monitor.py',
                     'runtime_pci.py','assigned_kfd.py','devices.py','snapshot.py','launch.py','read_terminal.py'):
            self.assertEqual((D/name).read_bytes(),(old/name).read_bytes(),name)
        tree=ast.parse((D/'core/infer_dev36.py').read_bytes())
        generate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
        text=ast.get_source_segment((D/'core/infer_dev36.py').read_text(encoding='utf8'),generate)
        self.assertIn("set(states[arm])==set(expected)",text)
        self.assertIn('torch.equal(actual[target].detach().cpu(),value)',text)
        self.assertIn("configs['domain25']==configs['generic25']",text)
        self.assertIn('OBSERVED_ADAPTER_CONFIG_SHA256',text)

    def test_delivered_v2_imports_new_dependency_with_no_default_target_execution(self):
        with TemporaryDirectory() as tmp:
            for p in D.rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:
                    out=Path(tmp)/p.relative_to(D);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(p.read_bytes())
            code="import sys,pathlib;d=pathlib.Path(sys.argv[1]);sys.path[:0]=[str(d),str(d/'core')];import controller,keeper,launch,infer_dev36,target_contract,score_after,read_terminal;assert pathlib.Path(target_contract.__file__).parent==d/'core';assert 'torch' not in sys.modules;print('V2_SELF_CONTAINED_IMPORT_OK')"
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',code,tmp],capture_output=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
        for path in D.rglob('*.py'):ast.parse(path.read_bytes(),feature_version=(3,10))

if __name__=='__main__':unittest.main()
