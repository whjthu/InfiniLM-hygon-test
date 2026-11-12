"""No tensor backend needed for original-body and restoration checks."""
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'generation/src'))
from gencuda.zimage_first_update import instrument_loss, instrument_update, observe_native_loss


def FlowMatchSFTLoss(pipe, **inputs):
    timestep_id=pipe.sample_timestep()
    timestep=timestep_id+3
    noise=pipe.sample_noise()
    training_target=noise-inputs['value']
    noise_pred=inputs['value']*timestep
    loss=(noise_pred-training_target)**2
    return loss


def no_target(pipe, **inputs):
    noise_pred=3
    loss=2
    return loss


def native_update(model,optimizer,data,seed,max_grad_norm):
    loss=model(data)
    loss.backward()
    grad_norm=model.clip_grad_norm_([],max_grad_norm)
    optimizer.step()
    return dict(loss=loss.value,norm=grad_norm)


class OriginalObservation(unittest.TestCase):
    def test_original_random_calls_and_exact_return_preserved(self):
        calls=[]
        pipe=types.SimpleNamespace(sample_timestep=lambda:(calls.append('timestep') or 2),sample_noise=lambda:(calls.append('noise') or 11))
        captured=[]
        observer=types.SimpleNamespace(prepared=lambda *a:captured.append(a),forward_value=lambda *a:captured.append(a))
        result=instrument_loss(FlowMatchSFTLoss,observer)(pipe,value=4)
        self.assertEqual(result,169);self.assertEqual(calls,['timestep','noise'])
        self.assertEqual(captured[0],({'value':4},11,2,5,7));self.assertEqual(captured[1],(20,169))

    def test_loss_global_restored_on_exception(self):
        native=types.SimpleNamespace(FlowMatchSFTLoss=FlowMatchSFTLoss)
        observer=types.SimpleNamespace(prepared=lambda *a:None,forward_value=lambda *a:None)
        with self.assertRaisesRegex(RuntimeError,'native-failed'):
            with observe_native_loss(native,observer):raise RuntimeError('native-failed')
        self.assertIs(native.FlowMatchSFTLoss,FlowMatchSFTLoss)

    def test_nested_loss_observer_rejected(self):
        native=types.SimpleNamespace(FlowMatchSFTLoss=FlowMatchSFTLoss)
        observer=types.SimpleNamespace(prepared=lambda *a:None,forward_value=lambda *a:None)
        with observe_native_loss(native,observer):
            with self.assertRaisesRegex(RuntimeError,'already installed'):
                with observe_native_loss(native,observer):pass
        self.assertIs(native.FlowMatchSFTLoss,FlowMatchSFTLoss)

    def test_unrecognized_native_function_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity'):instrument_loss(no_target,object())

    def test_real_update_boundary_order_preserved(self):
        calls=[]
        loss=types.SimpleNamespace(value=7,backward=lambda:calls.append('backward'))
        class Model:
            def __call__(self,data):calls.append('forward');return loss
            def clip_grad_norm_(self,p,n):calls.append('clip');return 9
        optimizer=types.SimpleNamespace(step=lambda:calls.append('step'))
        observer=types.SimpleNamespace(gradients=lambda name,*a:calls.append(name),updated=lambda:calls.append('updated'))
        result=instrument_update(native_update)(Model(),optimizer,{},0,1,observer=observer)
        self.assertEqual(result,dict(loss=7,norm=9));self.assertEqual(calls,['forward','backward','preclip','clip','postclip','step','updated'])

    def test_unrecognized_update_boundaries_rejected(self):
        with self.assertRaisesRegex(ValueError,'boundaries'):instrument_update(no_target)


if __name__=='__main__':unittest.main()
