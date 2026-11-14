import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('fusion_observer',Path(__file__).resolve().parents[1]/'generation/src/gencuda/zimage_native_fusion_audit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ObserverLifecycle(unittest.TestCase):
    def loader(self):
        class Loader:
            def fuse_lora_to_base_model(self, model, state_dict, alpha=1.0):
                model.append(('native',state_dict,alpha))
                return 'original-return'
        return Loader

    def test_original_call_return_and_argument_identity(self):
        loader=self.loader();original=loader.fuse_lora_to_base_model;events=[];model=[];state={}
        def before(owner,m,s,alpha):
            self.assertIs(m,model);self.assertIs(s,state);events.append('before');return 7
        def after(token,owner,m,s,alpha,seconds):
            self.assertEqual(token,7);self.assertEqual(len(model),1);self.assertGreaterEqual(seconds,0);events.append('after')
        with module.observe_one_fusion(loader,before,after):
            self.assertEqual(loader().fuse_lora_to_base_model(model,state,alpha=1),'original-return')
        self.assertIs(loader.fuse_lora_to_base_model,original);self.assertEqual(events,['before','after'])

    def test_observer_failure_restores_without_native_call(self):
        loader=self.loader();original=loader.fuse_lora_to_base_model;model=[]
        def before(*args):raise ValueError('reject-before-native')
        with self.assertRaisesRegex(ValueError,'reject-before-native'):
            with module.observe_one_fusion(loader,before,lambda *a:None):loader().fuse_lora_to_base_model(model,{})
        self.assertEqual(model,[]);self.assertIs(loader.fuse_lora_to_base_model,original)

    def test_post_failure_restores_and_preserves_native_denominator(self):
        loader=self.loader();original=loader.fuse_lora_to_base_model;model=[]
        def after(*args):raise ValueError('actual-mismatch')
        with self.assertRaisesRegex(ValueError,'actual-mismatch'):
            with module.observe_one_fusion(loader,lambda *a:None,after):loader().fuse_lora_to_base_model(model,{})
        self.assertEqual(len(model),1);self.assertIs(loader.fuse_lora_to_base_model,original)

    def test_no_fusion_is_rejected(self):
        loader=self.loader();original=loader.fuse_lora_to_base_model
        with self.assertRaisesRegex(RuntimeError,'no native'):
            with module.observe_one_fusion(loader,lambda *a:None,lambda *a:None):pass
        self.assertIs(loader.fuse_lora_to_base_model,original)

    def test_second_fusion_never_reaches_original(self):
        loader=self.loader();model=[]
        with self.assertRaisesRegex(RuntimeError,'second'):
            with module.observe_one_fusion(loader,lambda *a:None,lambda *a:None):
                loader().fuse_lora_to_base_model(model,{})
                loader().fuse_lora_to_base_model(model,{})
        self.assertEqual(len(model),1)

    def test_nested_observer_is_rejected(self):
        loader=self.loader();original=loader.fuse_lora_to_base_model
        with module.observe_one_fusion(loader,lambda *a:None,lambda *a:None):
            with self.assertRaisesRegex(RuntimeError,'already installed'):
                with module.observe_one_fusion(loader,lambda *a:None,lambda *a:None):pass
            loader().fuse_lora_to_base_model([],{})
        self.assertIs(loader.fuse_lora_to_base_model,original)


if __name__=='__main__':unittest.main()
