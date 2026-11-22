"""Local mocked-accelerator coverage of the actual Swift observer wrappers."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from observer import Observer
from swift_args import add_observer


X, DY, W = (1,186,4096), (1,186,1024), (1024,4096)


class Handle:
    def __init__(self, items, item):
        self.items, self.item = items, item
        items.append(item)

    def remove(self):
        if self.item in self.items:
            self.items.remove(self.item)


class Tensor:
    dtype, device = 'BF16', SimpleNamespace(index=1)
    is_cuda, requires_grad, _version = True, True, 0

    def __init__(self, shape):
        self.shape = shape
        self.hooks = []
        self.pointer = id(self)
        self.alias = None

    def view_as(self, _):
        view = Tensor(self.shape)
        view.pointer = self.pointer
        self.alias = view
        return view

    def stride(self):
        return (self.shape[-1]*self.shape[-2], self.shape[-1], 1)

    def data_ptr(self):
        return self.pointer

    def storage_offset(self):
        return 0

    def detach(self):
        return self

    def is_contiguous(self):
        return True

    def register_hook(self, hook):
        return Handle(self.hooks, hook)

    def backward_hook(self, gradient):
        for hook in tuple(self.hooks):
            assert hook(gradient) is None


class Base:
    bias, training = None, True

    def __init__(self):
        self.weight = Tensor(W)
        self.weight.requires_grad = False
        self.calls = []
        self.bad_output = False
        self.raise_error = None

    def forward(self, input):
        self.calls.append(input)
        if self.raise_error is not None:
            raise self.raise_error
        return Tensor((1,17,1024) if self.bad_output else DY)


class Root:
    training = True

    def __init__(self, base):
        self.base = base
        self.before, self.after = [], []

    def register_forward_pre_hook(self, hook):
        return Handle(self.before, hook)

    def register_forward_hook(self, hook, *, always_call):
        assert always_call
        return Handle(self.after, hook)

    def __call__(self, value):
        output = None
        for hook in tuple(self.before):
            hook(self, (value,))
        try:
            output = self.base.forward(input=value)
            return output
        finally:
            for hook in tuple(self.after):
                hook(self, (value,), output)


class Trainer:
    def __init__(self):
        self.events = []
        self.state = SimpleNamespace(global_step=22)
        self.loss = object()
        self.upstream = Tensor(DY)
        self.grad = Tensor(X)
        self.backward_count = 0
        self.root_count = 1

    def training_step(self, model, inputs, *, count=None):
        self.events.append(('step', count))
        for _ in range(self.root_count):
            output = model(inputs)
        model.base.forward(inputs)
        output.backward_hook(self.upstream)
        if inputs.alias:
            inputs.alias.backward_hook(self.grad)
        self.backward_count += 1
        self.events.append('native_backward_return')
        return self.loss


class Test(unittest.TestCase):
    def make(self, *, unsupported=False, fail_receipt=False):
        trainer, base = Trainer(), Base()
        root = Root(base)
        observations = []

        @contextmanager
        def fork(devices, enabled):
            self.assertEqual(devices,[1])
            self.assertTrue(enabled)
            trainer.events.append('rng_fork')
            try:
                yield
            finally:
                trainer.events.append('rng_restore')

        def hold(gradient, _torch):
            if unsupported:
                raise ValueError('test unsupported native layout')
            return SimpleNamespace(value=gradient)

        def evaluate(_torch, _linear, value, weight, upstream, output, dx, check):
            self.assertEqual(trainer.events[-2:], ['native_backward_return','rng_fork'])
            self.assertEqual((value.shape,weight.shape,output.shape),(X,W,DY))
            self.assertIs(dx,trainer.grad)
            self.assertIs(upstream.value,trainer.upstream)
            self.assertIs(value.alias.hooks[0].__self__.dx.tensor, dx)
            check()
            trainer.events.append('diagnose')
            return dict(candidate_status='STOP_ALREADY_CONTIGUOUS')

        def emit(value):
            if fail_receipt:
                raise OSError('test receipt path does not exist')
            observations.append(value)

        torch = SimpleNamespace(Tensor=Tensor, bfloat16='BF16', is_grad_enabled=lambda:True,
                               cuda=SimpleNamespace(current_device=lambda:1),
                               random=SimpleNamespace(fork_rng=fork),
                               nn=SimpleNamespace(functional=SimpleNamespace(linear=object())))
        layout = SimpleNamespace(X=X,W=W,DY=DY,OwnedUpstream=hold,evaluate=evaluate)
        observer = Observer(torch,layout,trainer,root,base,time.time()+60,emit)
        observer.install()
        return observer,trainer,root,base,observations

    def test_recorder_runs_once_and_only_after_real_backward_returns(self):
        observer,trainer,root,base,reports = self.make()
        x = Tensor(X)
        out = trainer.training_step(root,x,count=17)
        self.assertIs(out,trainer.loss)
        self.assertEqual(trainer.events,[('step',17),'native_backward_return','rng_fork','diagnose','rng_restore'])
        self.assertEqual(len(base.calls),2)
        self.assertIs(base.calls[0],x.alias)
        self.assertIs(base.calls[1],x)
        self.assertFalse(x.alias.hooks)
        self.assertFalse(root.before or root.after)
        self.assertIsNone(observer.edge)
        self.assertEqual(reports[0]['diagnostics_invoked'],1)
        first = reports[0]['coverage'][0]
        self.assertEqual((first['global_step'],first['root_forwards'],first['leaf_outside_root_calls'],
                          first['native_dx_hooks'],first['native_dy_hooks']),(22,1,1,1,1))
        trainer.training_step(root,Tensor(X))
        self.assertEqual(trainer.backward_count,2)
        self.assertEqual(len(reports),1)

    def test_missing_shape_four_micros_and_native_receipts_unchanged(self):
        _,trainer,root,base,reports = self.make()
        for i in range(6):
            x = Tensor((1,187,4096))
            self.assertIs(trainer.training_step(root,x),trainer.loss)
            self.assertIsNone(x.alias)
        self.assertEqual(trainer.backward_count,6)
        self.assertEqual(len(base.calls),12)
        self.assertEqual(reports[0]['state'],'WINDOW_NO_ELIGIBLE_EDGE')
        self.assertEqual(len(reports[0]['coverage']),4)
        self.assertTrue(all(row['unmatched_leaf_calls']==1 and row['native_step_returned']
                            for row in reports[0]['coverage']))

    def test_unsupported_gradient_or_output_keeps_native_step(self):
        for bad_output in (False, True):
            with self.subTest(bad_output=bad_output):
                _,trainer,root,base,reports = self.make(unsupported=not bad_output)
                base.bad_output = bad_output
                self.assertIs(trainer.training_step(root,Tensor(X)),trainer.loss)
                self.assertEqual(len(base.calls),2)
                micro = reports[0]['coverage'][0]
                self.assertEqual(micro['diagnostic']['candidate_status'],'UNSUPPORTED_NATIVE_EDGE')
                self.assertTrue(micro['native_step_returned'])
                self.assertEqual(reports[0]['diagnostics_invoked'],0)

    def test_native_exception_propagates_and_own_hooks_are_released(self):
        observer,trainer,root,base,reports = self.make()
        failure = RuntimeError('real native leaf test exception')
        base.raise_error = failure
        x = Tensor(X)
        with self.assertRaises(RuntimeError) as observed:
            trainer.training_step(root,x)
        self.assertIs(observed.exception,failure)
        self.assertEqual(len(base.calls),1)
        self.assertFalse(x.alias.hooks)
        self.assertFalse(observer.installed)
        self.assertEqual(reports[0]['state'],'NATIVE_STEP_RAISED')

    def test_failed_receipt_does_not_change_native_loss(self):
        observer,trainer,root,_,_ = self.make(fail_receipt=True)
        self.assertIs(trainer.training_step(root,Tensor(X)),trainer.loss)
        self.assertEqual(observer.result['receipt_error']['type'],'OSError')

    def test_swift_args_and_callback_registry_keep_original_options(self):
        cmd = ['swift_entry.py','sft','--callbacks','native','--external_plugins','original.py','--attn_impl','sdpa',
               '--bf16','true','--gradient_accumulation_steps','4']
        plugin = '/public/home/reviewed/swift_plugin.py'
        actual = add_observer(cmd,plugin)
        self.assertEqual([item for item in actual if item not in (plugin,'qwen_native_vproj_upstream_layout')],cmd)
        with self.assertRaises(ValueError):
            add_observer(actual,plugin)
        swift = ModuleType('swift.callbacks')
        swift.TrainerCallback, swift.callbacks_map = type('TrainerCallback', (), {}), {}
        peft = ModuleType('peft.tuners.lora.layer')
        peft.Linear = Base
        with patch.dict(sys.modules, {'torch':ModuleType('torch'),'swift.callbacks':swift,'peft.tuners.lora.layer':peft}):
            script = runpy.run_path(str(HERE/'swift_plugin.py'))
        self.assertIs(swift.callbacks_map['qwen_native_vproj_upstream_layout'],script['QwenLayoutCallback'])
        prior = HERE.parent/'training-qwen-native-vproj-upstream-layout-source-v1'
        implementation = script['load_layout'](prior)
        self.assertEqual(implementation.X,X)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)
            script['publish'](output,dict(coverage=[1]))
            with self.assertRaises(FileExistsError):
                script['publish'](output,dict(coverage=[2]))
            self.assertEqual(json.loads((output/script['REPORT']).read_bytes()),dict(coverage=[1]))

    def test_swift_callback_native_start_and_other_rank_no_hooks(self):
        trainer, base = Trainer(), Base()
        root = Root(base)
        root.get_base_model = lambda:root
        class Peft:
            active_adapters = ['default']
            disable_adapters = merged = False
            use_dora = {}
            def get_base_layer(self):
                return base
        root.get_submodule = lambda name:SimpleNamespace(
            layers=[SimpleNamespace(get_submodule=lambda _:Peft()) for _ in range(36)])
        trainer.model = root
        torch = ModuleType('torch')
        torch.version = SimpleNamespace(hip='mocked-hip')
        torch.bfloat16 = 'BF16'
        torch.nn = SimpleNamespace(Linear=Base, parallel=SimpleNamespace(DistributedDataParallel=type('DDP',(),{})))
        torch.cuda = SimpleNamespace(current_device=lambda:1)
        swift = ModuleType('swift.callbacks')
        swift.TrainerCallback, swift.callbacks_map = type('TrainerCallback', (), {}), {}
        peft = ModuleType('peft.tuners.lora.layer')
        peft.Linear = Peft
        prior = HERE.parent/'training-qwen-native-vproj-upstream-layout-source-v1'
        with patch.dict(sys.modules, {'torch':torch,'swift.callbacks':swift,'peft.tuners.lora.layer':peft}):
            script = runpy.run_path(str(HERE/'swift_plugin.py'))
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(output_dir=td,bf16=True,per_device_train_batch_size=1,gradient_accumulation_steps=4)
            action = script['QwenLayoutCallback']()
            action.trainer = trainer
            control = object()
            env = dict(RANK='0', LOCAL_RANK='0', WORLD_SIZE='8', QWEN_VPROJ_LAYOUT_SOURCE=str(prior),
                       QWEN_VPROJ_LAYOUT_DEADLINE=str(time.time()+60))
            with patch.dict(os.environ,env):
                self.assertIs(action.on_train_begin(args,trainer.state,control,model=root),control)
                self.assertFalse(root.before or root.after)
            env.update(RANK='1',LOCAL_RANK='1')
            with patch.dict(os.environ,env):
                self.assertIs(action.on_train_begin(args,trainer.state,control,model=root),control)
                self.assertIsNone(action.unavailable)
                self.assertTrue(action.observation.installed)
                self.assertIs(action.on_train_end(args,trainer.state,control),control)
            self.assertFalse(root.before or root.after)
            receipt = json.loads((Path(td)/script['REPORT']).read_bytes())
            self.assertEqual(receipt['state'],'TRAINING_ENDED_BEFORE_ELIGIBLE_EDGE')
            self.assertEqual(receipt['coverage'],[])


if __name__ == '__main__':
    unittest.main()
