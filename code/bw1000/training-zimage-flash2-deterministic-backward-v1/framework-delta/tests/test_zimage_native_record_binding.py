from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda.zimage_native_record import canonical,validate_binding,run


class NativeBinding(unittest.TestCase):
    def test_parent_checkpoint_bypass_requires_explicit_derived_policy_scope(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            b=self.binding()
            with self.assertRaisesRegex(ValueError,'parent checkpoint'):
                run(b,Path(temp)/'none',mode='resume20',epoch=1,deadline=2,checkpoint='unused',first_resumed_update_only=True,diagnostic_parent_checkpoint=True)
            b['config']['attention_backward_policy']='flash2-deterministic-backward-v1';b['config_canonical_SHA256']=canonical(b['config'])
            with self.assertRaisesRegex(ValueError,'parent checkpoint'):
                run(b,Path(temp)/'none',mode='continuous',epoch=1,deadline=2,diagnostic_parent_checkpoint=True)
            parent=dict(b['config']);parent.pop('attention_backward_policy');parent['learning_rate']=1e-3
            b.update(diagnostic_parent_config=parent,diagnostic_parent_config_SHA256=canonical(parent))
            with self.assertRaises(ValueError):run(b,Path(temp)/'none',mode='resume20',epoch=1,deadline=2,checkpoint='unused',first_resumed_update_only=True,diagnostic_parent_checkpoint=True)
            self.assertFalse((Path(temp)/'none').exists())

    def test_diagnostic_flag_cannot_silently_shorten_continuous_recipe(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            for flag,mode in [(True,'continuous'),(1,'resume20'),(None,'resume20')]:
                output=Path(temp)/'must-not-exist'
                with self.subTest(flag=flag,mode=mode),self.assertRaisesRegex(ValueError,'resumed21 diagnosis'):
                    run(self.binding(),output,mode=mode,epoch=1.,deadline=2.,checkpoint='unused',first_resumed_update_only=flag)
                self.assertFalse(output.exists())
            for flag in (True,1,None):
                with self.subTest(warmup=flag),self.assertRaisesRegex(ValueError,'diagnostic warmup'):
                    run(self.binding(),Path(temp)/'must-not-exist',mode='resume20',epoch=1.,deadline=2.,checkpoint='unused',warmup_before_restored_update=flag)
            for flag in (True,1,None):
                with self.subTest(profile=flag),self.assertRaisesRegex(ValueError,'native profile'):
                    run(self.binding(),Path(temp)/'must-not-exist',mode='resume20',epoch=1.,deadline=2.,checkpoint='unused',profile_resumed_update=flag)

    def binding(self):
        config=dict(model='z-image-turbo',seed=20260911,cards='0',precision='bfloat16',lora_rank=4,
            lora_target_modules='to_q,to_k,to_v,to_out.0,w1,w2,w3',learning_rate=1e-4,weight_decay=.01,
            max_steps=25,warmup_steps=5,max_grad_norm=1.,gradient_checkpointing=True,height=256,width=256,
            checkpoint_steps=[10,20,25],validate_recovery=False,validate_turbo_regression=True)
        return dict(config=config,config_canonical_SHA256=canonical(config),data=dict(images={str(i):{} for i in range(20)},row_order=list(range(20))),
            initial=dict(trainable_tensors=476,trainable_elements=9922560),model=dict(model_files=77,preset_files=1),
            platform='cuda',logical_device_ids=[0],planned_continuous_steps=25,planned_resume_global_steps=[21,22,23,24,25])

    def test_exact_original_schedule_and_extent(self):
        validate_binding(self.binding())
        b=self.binding();b['platform']='hip';validate_binding(b)

    def test_shrunken_recipe_rejected_even_with_new_hash(self):
        for key,value in [('max_steps',1),('height',128),('lora_rank',2),('cards','0,1'),('gradient_checkpointing',False),('warmup_steps',0)]:
            b=self.binding();b['config'][key]=value;b['config_canonical_SHA256']=canonical(b['config'])
            with self.subTest(key=key),self.assertRaises(ValueError):validate_binding(b)

    def test_hidden_configuration_change_requires_new_binding(self):
        b=self.binding();b['config']['another_native_flag']=True
        with self.assertRaisesRegex(ValueError,'complete original'):validate_binding(b)

    def test_partial_data_or_trainable_scope_rejected(self):
        for area,key,value in [('data','images',{'0':{}}),('data','row_order',[0]),('initial','trainable_tensors',475),('initial','trainable_elements',9),('model','model_files',7)]:
            b=self.binding();b[area][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_binding(b)

    def test_nonfresh_resume_schedule_rejected(self):
        b=self.binding();b['planned_resume_global_steps']=[25]
        with self.assertRaisesRegex(ValueError,'schedule'):validate_binding(b)

    def test_boolean_integer_and_multiple_devices_rejected(self):
        b=self.binding();b['config']['lora_rank']=4.;b['config_canonical_SHA256']=canonical(b['config'])
        with self.assertRaises(ValueError):validate_binding(b)
        b=self.binding();b['logical_device_ids']=[0,1]
        with self.assertRaises(ValueError):validate_binding(b)


if __name__=='__main__':unittest.main()
