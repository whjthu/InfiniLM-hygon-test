"""Policy explicitness,call ownership and preservation of nontraining calls."""
from pathlib import Path
import inspect,sys,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda import zimage_attention_policy as policy


class AttentionPolicy(unittest.TestCase):
    def fixture(self,raises=False):
        seen=[];result=object();state=dict(grad=True)
        def original(q,k,v,dropout_p=0.,softmax_scale=None,causal=False,deterministic=False):
            seen.append(deterministic)
            if raises:raise RuntimeError('actual-native-failed')
            return result
        module=types.SimpleNamespace(flash_attn_func=original);torch=types.SimpleNamespace(is_grad_enabled=lambda:state['grad'])
        q=types.SimpleNamespace(requires_grad=True,device=types.SimpleNamespace(type='cuda'),shape=(1,288,30,128),stride=lambda:(1105920,3840,128,1),dtype='BF16')
        return module,torch,original,inspect.signature(original),q,state,seen,result

    def test_absent_flag_is_inert_and_invalid_flags_rejected(self):
        with patch.object(policy,'_resolve_provider',side_effect=AssertionError('must-not-resolve')):
            with policy.apply_training_attention_policy(policy.policy_for_config({})) as ledger:self.assertIsNone(ledger)
        for flag in (None,False,True,'unknown'):
            with self.subTest(flag=flag),self.assertRaises(ValueError):policy.policy_for_config(dict(attention_backward_policy=flag))

    def test_actual_grad_call_forced_and_inference_return_identity_preserved(self):
        m,t,o,s,q,state,seen,result=self.fixture();p=policy.policy_for_config(dict(attention_backward_policy=policy.POLICY_ID))
        with patch.object(policy,'_resolve_provider',return_value=(m,t,o,s)):
            with policy.apply_training_attention_policy(p) as ledger:
                self.assertIs(m.flash_attn_func(q,q,q),result);state['grad']=False
                self.assertIs(m.flash_attn_func(q,q,q),result)
                self.assertEqual(seen,[True,False]);self.assertEqual(ledger['forced_calls'],1);self.assertEqual(ledger['unchanged_calls'],1)
        self.assertIs(m.flash_attn_func,o)

    def test_conflicting_caller_override_is_rejected_and_restored(self):
        m,t,o,s,q,state,seen,result=self.fixture();p=policy.policy_for_config(dict(attention_backward_policy=policy.POLICY_ID))
        with patch.object(policy,'_resolve_provider',return_value=(m,t,o,s)):
            with self.assertRaisesRegex(ValueError,'override'):
                with policy.apply_training_attention_policy(p):m.flash_attn_func(q,q,q,deterministic=False)
        self.assertEqual(seen,[]);self.assertIs(m.flash_attn_func,o)

    def test_native_exception_does_not_leave_policy_installed(self):
        m,t,o,s,q,state,seen,result=self.fixture(True);p=policy.policy_for_config(dict(attention_backward_policy=policy.POLICY_ID))
        with patch.object(policy,'_resolve_provider',return_value=(m,t,o,s)):
            with self.assertRaisesRegex(RuntimeError,'actual-native-failed'):
                with policy.apply_training_attention_policy(p):m.flash_attn_func(q,q,q)
        self.assertEqual(seen,[True]);self.assertIs(m.flash_attn_func,o)

if __name__=='__main__':unittest.main()
