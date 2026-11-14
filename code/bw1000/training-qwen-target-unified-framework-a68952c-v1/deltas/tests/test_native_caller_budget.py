import unittest
from types import SimpleNamespace as NS
from vlm.runtime.native_caller_budget import describe_capture, plan_capture, validate_capture_conditions


class CaptureBudgetTests(unittest.TestCase):
    def rows(self):
        return [dict(ordinal=i,sequence_length=3363,selected_logit_rows=21,
                     collated_serialized_bytes=32*1024**2,image_tokens=3072) for i in range(1,65)]

    def plan(self, rows):
        return plan_capture(rows,parameters=43646976,tensors=504,hidden_size=4096,vocab_size=151936)

    def test_complete_f32_conservative_plan_fits_unchanged_limits(self):
        p=self.plan(self.rows())
        self.assertLess(p['per_rank_total_upper_bytes'],p['per_rank_total_limit'])
        self.assertLess(p['maximum_artifact_upper_bytes'],p['per_artifact_limit'])
        self.assertEqual(p['full_parameter_and_AdamW_payload_bytes'],15*43646976*4)
        self.assertFalse(p['actual_GPU_serialized_artifact_bytes_qualified'])

    def test_all_rows_are_required_in_order(self):
        for rows in (self.rows()[:-1],self.rows()[::-1]):
            with self.assertRaises(ValueError):self.plan(rows)

    def test_full_logits_fail_instead_of_truncating(self):
        rows=self.rows()
        for row in rows:row['selected_logit_rows']=3363
        with self.assertRaises(ValueError):self.plan(rows)

    def test_oversized_full_payload_report_is_not_an_admission_plan(self):
        rows=self.rows()
        for row in rows:row['selected_logit_rows']=3363
        description=describe_capture(rows,parameters=43646976,tensors=504,
                                     hidden_size=4096,vocab_size=151936)
        self.assertFalse(description['within_recorder_limits'])
        self.assertFalse(description['analytical_sufficiency'])
        self.assertEqual(description['per_rank_excess_upper_bytes'],
                         description['per_rank_total_upper_bytes']-8*1024**3)
        self.assertFalse(description['input_truncation_allowed'])
        self.assertFalse(description['output_truncation_allowed'])
        self.assertEqual(rows[0]['selected_logit_rows'],3363)
        with self.assertRaises(ValueError):self.plan(rows)

    def test_invalid_native_counts_fail(self):
        for key,value in [('sequence_length',4097),('selected_logit_rows',True),
                          ('selected_logit_rows',4000),('image_tokens',0),
                          ('collated_serialized_bytes',2**31)]:
            rows=self.rows();rows[3][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.plan(rows)


class CaptureConditionsTests(unittest.TestCase):
    def fixture(self, case='T04.sft'):
        def forward(self,input_ids=None,attention_mask=None,inputs_embeds=None):return None
        cls=('LLaVAOneVision1_5_ForConditionalGeneration' if case=='T04.sft'
             else 'InternVLForConditionalGeneration')
        core=type(cls,(),{'forward':forward})()
        language=NS(config=NS(hidden_size=4096,vocab_size=151936,use_cache=False,
                              output_hidden_states=False,output_attentions=False))
        core.config=NS(text_config=language.config,use_cache=False,
                       output_hidden_states=False,output_attentions=False)
        core.model=NS(language_model=language)
        core.lm_head=NS(in_features=4096,out_features=151936,
                        weight=NS(shape=(151936,4096),dtype='torch.bfloat16',element_size=lambda:2))
        conditions=dict(case=case,hidden_size=4096,vocab_size=151936,language_cache=False,
                        hidden_state_stacks=False,attention_maps=False)
        if case=='T04.sft':conditions['native_logits_to_keep']=False
        return core,conditions

    def test_original_InternVL_guard_keeps_its_contract(self):
        core,conditions=self.fixture('T03.sft')
        self.assertEqual(validate_capture_conditions(core,conditions),
                         dict(passed=True,**conditions,actual_model='InternVLForConditionalGeneration'))
        core.model.language_model.config.use_cache=True
        with self.assertRaises(ValueError):validate_capture_conditions(core,conditions)

    def test_OneVision_default_or_false_preserves_complete_logits(self):
        core,conditions=self.fixture()
        for option in (None,False):
            with self.subTest(option=option):
                self.assertTrue(validate_capture_conditions(core,conditions,native_logits_option=option)['passed'])
        wrapper=NS(get_base_model=lambda:core)
        self.assertTrue(validate_capture_conditions(wrapper,conditions)['passed'])
        for option in (True,0,'false'):
            with self.subTest(option=option),self.assertRaises(ValueError):
                validate_capture_conditions(core,conditions,native_logits_option=option)

    def test_OneVision_width_cache_hidden_outputs_and_attention_maps_are_rejected(self):
        changes=(('root','use_cache',True),('language','use_cache',True),
                 ('root','output_hidden_states',True),('language','output_hidden_states',True),
                 ('root','output_attentions',True),('language','output_attentions',True),
                 ('language','hidden_size',2048),('language','vocab_size',129280))
        for target,name,value in changes:
            core,conditions=self.fixture();config=core.config if target=='root' else core.model.language_model.config
            setattr(config,name,value)
            with self.subTest(target=target,name=name),self.assertRaises(ValueError):validate_capture_conditions(core,conditions)
        core,conditions=self.fixture();conditions['native_logits_to_keep']=True
        with self.assertRaises(ValueError):validate_capture_conditions(core,conditions)
        core,conditions=self.fixture();core.__class__=type('OtherModel',(),{})
        with self.assertRaises(ValueError):validate_capture_conditions(core,conditions)

    def test_OneVision_actual_head_and_forward_signature_are_checked(self):
        for kind in ('shape','in_features','out_features','float64','float32','float16','element_bytes','signature'):
            core,conditions=self.fixture()
            if kind=='shape':core.lm_head.weight.shape=(4096,151936)
            elif kind=='in_features':core.lm_head.in_features=2048
            elif kind=='out_features':core.lm_head.out_features=32000
            elif kind in ('float64','float32','float16'):core.lm_head.weight.dtype='torch.'+kind
            elif kind=='element_bytes':core.lm_head.weight.element_size=lambda:4
            else:core.forward=lambda input_ids=None,logits_to_keep=0:None
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_capture_conditions(core,conditions)


if __name__=='__main__':unittest.main()
