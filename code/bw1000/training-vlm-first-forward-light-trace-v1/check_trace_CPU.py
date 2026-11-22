"""New observer fixtures with no Torch/model/device dependencies."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import unittest
from types import SimpleNamespace

from light_trace import install_if_enabled,SCHEMA
import native_bridge
from optin import attach_if_requested


class Handle:
    def __init__(self,rows,item):self.rows,self.item=rows,item
    def remove(self):
        if self.item in self.rows:self.rows.remove(self.item)


class FakeModel:
    def __init__(self,backend):self.pre=[];self.post=[];self.backend=backend;self.choice='UNCHANGED_NATIVE_FIXTURE';self.output=object()
    def register_forward_pre_hook(self,hook,**kwargs):
        assert kwargs==dict(with_kwargs=True);self.pre.append(hook);return Handle(self.pre,hook)
    def register_forward_hook(self,hook,**kwargs):
        assert kwargs==dict(with_kwargs=True,always_call=True);self.post.append(hook);return Handle(self.post,hook)
    def forward(self):
        for hook in self.pre:assert hook(self,(),{}) is None
        self.backend.counter+=1
        for hook in self.post:assert hook(self,(),{},self.output) is None
        return self.output


class Backend:
    def __init__(self):self.counter=0;self.metadata_calls=0
    def check(self):pass
    def rng_snapshot(self):return dict(SHA256=hashlib.sha256(str(self.counter).encode()).hexdigest())
    def metadata(self,*_):
        self.metadata_calls+=1
        return dict(context={'mode':'fixture_train','autocast_dtype':'fixture_bf16'},
                    input_layouts={'args/0':{'shape':[1,3],'dtype':'fixture'}},scale={'configured_accumulation':4})


def config(tmp,phase='resumed'):
    output=Path(tmp)/'trace';output.mkdir()
    return dict(enabled=True,schema=SCHEMA,root_GO=True,trace_GO=True,diagnostic_only=True,
        request_id='9'*32,ranks=[0,1],phase=phase,start_ordinal=81 if phase=='continuous' else 1,
        count=4,output=str(output),scope_root=str(tmp))


class Trace(unittest.TestCase):
    def test_disabled_or_unselected_is_zero_backend_and_zero_hooks(self):
        def forbidden():raise AssertionError('disabled path touched backend')
        for value in (None,{},dict(enabled=False)):
            self.assertIsNone(install_if_enabled(value,rank=None,backend_factory=forbidden))
            self.assertIsNone(native_bridge.install(value,model=object(),trainer=object(),rank=None))
        selected=dict(enabled=True,schema=SCHEMA,root_GO=True,trace_GO=True,diagnostic_only=True,ranks=[0,1])
        self.assertIsNone(install_if_enabled(selected,rank=2,backend_factory=forbidden))
        self.assertNotIn('torch',sys.modules)
        self.assertIsNone(attach_if_requested(environ={},model=object(),trainer=object()))

    def test_selected_hooks_keep_output_choice_and_rng_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Backend();model=FakeModel(b)
            t=install_if_enabled(config(tmp),rank=0,backend_factory=lambda:b).attach(model)
            for _ in range(4):self.assertIs(model.forward(),model.output)
            result=t.finish()
            self.assertTrue(result['complete'])
            self.assertEqual((result['entries_attempted'],result['exits_completed']),(4,4))
            self.assertEqual(b.counter,4) # native fixture advanced once; observer never advanced it
            self.assertEqual(model.choice,'UNCHANGED_NATIVE_FIXTURE')
            self.assertEqual(model.pre,[]);self.assertEqual(model.post,[])
            entries=sorted(t.root.glob('*-entry-private.json'));exits=sorted(t.root.glob('*-exit-private.json'))
            self.assertEqual(len(entries),4);self.assertEqual(len(exits),4)
            self.assertNotEqual(json.loads(entries[0].read_bytes())['RNG'],json.loads(exits[0].read_bytes())['RNG'])
            self.assertEqual(len(list(t.root.glob('context-*-private.json'))),1)

    def test_continuous_selection_skips_first80_without_metadata_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Backend();model=FakeModel(b)
            t=install_if_enabled(config(tmp,'continuous'),rank=1,backend_factory=lambda:b).attach(model)
            for _ in range(80):model.forward()
            self.assertEqual(b.metadata_calls,0)
            for _ in range(4):model.forward()
            self.assertTrue(t.finish()['complete'])
            self.assertEqual(sorted(p.name for p in t.root.glob('*-entry-private.json')),
                ['081-entry-private.json','082-entry-private.json','083-entry-private.json','084-entry-private.json'])

    def test_partial_and_observer_rng_mutation_remain_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Backend();model=FakeModel(b)
            t=install_if_enabled(config(tmp),rank=0,backend_factory=lambda:b).attach(model)
            model.forward();result=t.finish()
            self.assertFalse(result['complete']);self.assertEqual(result['planned_forward_pairs'],4)
            self.assertEqual(result['exits_completed'],1)
        with tempfile.TemporaryDirectory() as tmp:
            class Bad(Backend):
                def metadata(self,*args):self.counter+=1;return super().metadata(*args)
            b=Bad();model=FakeModel(b)
            t=install_if_enabled(config(tmp),rank=0,backend_factory=lambda:b).attach(model)
            with self.assertRaises(ValueError):model.forward()
            failure=json.loads((t.root/'failure-private.json').read_bytes())
            self.assertEqual(failure['planned_forward_pairs'],4)
            self.assertEqual(failure['selected_entries_attempted'],1)
            self.assertEqual(failure['selected_exits_completed'],0)
            self.assertFalse(t.finish()['complete'])

    def test_unapproved_or_changed_selection_rejected(self):
        for change in ({'root_GO':False},{'trace_GO':False},{'count':5},{'start_ordinal':2}):
            with tempfile.TemporaryDirectory() as tmp:
                c=config(tmp);c.update(change)
                with self.subTest(change=change),self.assertRaises(ValueError):
                    install_if_enabled(c,rank=0,backend_factory=Backend)

    def test_native_rng_adapter_exports_hashes_only(self):
        class Keys:
            def tolist(self):return [123,456]
        class RNG:
            cpu=staticmethod(lambda value:value)
            fingerprint=staticmethod(lambda value:dict(SHA256=hashlib.sha256(repr(value).encode()).hexdigest()))
            rng_fingerprint=staticmethod(lambda value:dict(SHA256='b'*64,finite=True,floating_l2=999.))
            @staticmethod
            def rng_values(device):
                return dict(python='RAW_SECRET_RNG',numpy=('MT19937',Keys(),0,0,0.),
                            cpu='RAW_SECRET_RNG',cuda=device())
        backend=object.__new__(native_bridge.NativeBackend)
        backend.rank=1;backend.rng=RNG()
        backend.torch=SimpleNamespace(cuda=SimpleNamespace(get_rng_state=lambda rank:'RAW_SECRET_RNG'))
        value=backend.rng_snapshot()
        self.assertEqual(set(value['components_SHA256']),{'python','numpy','CPU','device_local'})
        self.assertNotIn('RAW_SECRET_RNG',json.dumps(value))
        self.assertNotIn('floating_l2',json.dumps(value))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Trace))
    print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        fixture_only=True,Torch_imported='torch' in sys.modules,GPU_calls=0,remote_calls=0,optimizer_updates=0,
        real_model_or_device_trace_qualified=False,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')}),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
