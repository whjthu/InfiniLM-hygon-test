"""Profile annotations preserve update math and own actual dispatch wrappers."""
from contextlib import contextmanager
from pathlib import Path
import sys,types,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda.zimage_native_profile import capture_attention_routes,instrument_profiled_update

calls=[]
def synchronize():calls.append('sync')
def native_update(model,optimizer,data,seed,max_grad_norm):
    loss=model(data)
    loss.backward()
    grad_norm=model.clip_grad_norm_([],max_grad_norm)
    optimizer.step()
    synchronize()
    return loss,grad_norm
def missing_update(model):return model()

class ProfileContract(unittest.TestCase):
    def test_exact_math_order_return_and_phase_annotations(self):
        phases=[]
        @contextmanager
        def record(name):phases.append(name);yield
        loss=types.SimpleNamespace(backward=lambda:calls.append('backward'))
        class Model:
            def __call__(self,data):calls.append('forward');return loss
            def clip_grad_norm_(self,p,v):calls.append('clip');return 7
        optimizer=types.SimpleNamespace(step=lambda:calls.append('step'))
        calls.clear();plain=native_update(Model(),optimizer,{},1,1);baseline=list(calls)
        calls.clear();observed=instrument_profiled_update(native_update,record)(Model(),optimizer,{},1,1)
        self.assertEqual(baseline,calls);self.assertIs(plain[0],observed[0]);self.assertEqual(observed[1],7)
        self.assertEqual(phases,['native.forward','native.backward','native.clip_grad','native.optimizer','native.synchronize'])

    def test_unknown_update_rejected(self):
        with self.assertRaisesRegex(ValueError,'boundaries'):instrument_profiled_update(missing_update,lambda _:None)

    def fixture(self,raises=False):
        result=object()
        def original(*a,**k):
            if raises:raise RuntimeError('native-failed')
            return result
        @contextmanager
        def record(n):yield
        native=types.SimpleNamespace(**{n:original for n in ('torch_sdpa','flash_attention_3','flash_attention_2','sage_attention','xformers_attention')})
        torch=types.SimpleNamespace(nn=types.SimpleNamespace(functional=types.SimpleNamespace(scaled_dot_product_attention=original)),profiler=types.SimpleNamespace(record_function=record),is_grad_enabled=lambda:True)
        tensor=types.SimpleNamespace(shape=(2,4),stride=lambda:(4,1),dtype='BF16',device='cpu',requires_grad=True)
        return native,torch,tensor,original,result

    def test_actual_route_metadata_and_return_identity(self):
        native,torch,q,original,result=self.fixture()
        with capture_attention_routes(native,torch,lambda:None) as rows:
            self.assertIs(native.torch_sdpa(q,q,q,attn_mask=q),result)
            self.assertIs(torch.nn.functional.scaled_dot_product_attention(q,q,q,attn_mask=q),result)
            self.assertEqual(sum(r['calls'] for r in rows()),2);self.assertTrue(all(r['mask']['shape']==[2,4] for r in rows()))
        self.assertIs(native.torch_sdpa,original);self.assertIs(torch.nn.functional.scaled_dot_product_attention,original)

    def test_wrappers_restore_after_native_exception(self):
        native,torch,q,original,result=self.fixture(True)
        with self.assertRaisesRegex(RuntimeError,'native-failed'):
            with capture_attention_routes(native,torch,lambda:None):native.flash_attention_2(q,q,q)
        for name in vars(native):self.assertIs(getattr(native,name),original)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention,original)

if __name__=='__main__':unittest.main()
