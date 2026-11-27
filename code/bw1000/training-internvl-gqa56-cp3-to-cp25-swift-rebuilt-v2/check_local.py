import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest

import live_hooks
import prepare
import skip_cpu
import swift_watch as watch


class WitnessTests(unittest.TestCase):
    def test_exact_nested_owner_and_original_accelerate_identity(self):
        def plain(*args):
            return args
        def nested(self,origin_skip_first_batches):
            def skip_first_batches(*args):
                return self,origin_skip_first_batches(*args)
            return skip_first_batches
        owner=object()
        wrapped=nested(owner,plain)
        hf=SimpleNamespace(skip_first_batches=wrapped)
        path=Path(inspect.getsourcefile(wrapped))
        line=wrapped.__code__.co_firstlineno
        self.assertIs(watch.active_skip_callable(hf,plain,owner,path,line),wrapped)
        with self.assertRaisesRegex(ValueError,"CLOSURE_OWNER"):
            watch.active_skip_callable(hf,plain,object(),path,line)
        hf.skip_first_batches=plain
        with self.assertRaisesRegex(ValueError,"LIVE_SWIFT_SKIP_SOURCE"):
            watch.active_skip_callable(hf,plain,owner,path,line)

    def test_new_actual_nested_sampler_is_required(self):
        class Batches(SimpleNamespace):
            def __len__(self):
                return 8
        class Wrapper(SimpleNamespace):
            pass
        class Loader(SimpleNamespace):
            def __len__(self):
                return self.size
        values=dict(rank=3,world_size=8,tp_size=1,total_samples=8,batch_size=1,base_seed=42,curr_seed=42,
                    shuffle=True,drop_last=False,group_by_length=False)
        old,new=(Batches(**values) for _ in range(2))
        common=dict(dataset=object(),num_workers=4,in_order=True)
        base=Loader(**common,batch_sampler=old,size=8)
        rebuilt=Loader(**common,batch_sampler=Wrapper(batch_sampler=new,skip_batches=4),size=4)
        self.assertIs(watch.require_rebuilt(base,rebuilt,old,Wrapper,Batches,3),new)
        rebuilt.batch_sampler.batch_sampler=old
        with self.assertRaisesRegex(ValueError,"FRESH_RAW_SAMPLER"):
            watch.require_rebuilt(base,rebuilt,old,Wrapper,Batches,3)

    def test_cleanup_finishes_inside_original_patch_on_exception(self):
        seen=[]
        live=SimpleNamespace(is_swift=False)
        class Mixin:
            @contextmanager
            def _patch_skip_first_batches(self):
                live.is_swift=True
                seen.append("enter")
                try:
                    yield
                finally:
                    seen.append("native_exit")
                    live.is_swift=False
        class Cursor:
            def detach(self,strict):
                if not live.is_swift or strict:
                    raise AssertionError("Cleanup outside failure scope")
                seen.append("observer_cleanup")
        original=Mixin._patch_skip_first_batches
        here=Path(inspect.getsourcefile(inspect.unwrap(original)))
        watch.wrap_native_context(Mixin,lambda *args,**kw:None,here)
        owner=Mixin()
        with self.assertRaisesRegex(RuntimeError,"prior native failure"):
            with owner._patch_skip_first_batches():
                setattr(owner,watch.SCOPE_ATTR,Cursor())
                raise RuntimeError("prior native failure")
        self.assertIs(Mixin._patch_skip_first_batches,original)
        self.assertFalse(hasattr(owner,watch.SCOPE_ATTR))
        self.assertEqual(seen,["enter","observer_cleanup","native_exit"])

    def test_detach_does_not_replace_foreign_callback(self):
        original,watched,foreign,plain,shim=(lambda:None for _ in range(5))
        cls=type("Sampler",(),{"__iter__":watched})
        hf=SimpleNamespace(skip_first_batches=foreign)
        cursor=SimpleNamespace(hooks=(cls,original,watched,hf,plain,shim))
        live_hooks.detach(cursor,strict=False)
        self.assertIs(cls.__iter__,original)
        self.assertIs(hf.skip_first_batches,foreign)

    def test_candidate_source_one_scope_gpu_disabled(self):
        files=prepare.render()
        self.assertEqual(files,prepare.render())
        claim=json.loads(files["CLAIM.json"])
        self.assertFalse(claim["GPU_GO"])
        self.assertEqual((claim["planned_updates"],claim["planned_input_slots"],claim["placement"]["cpus"]),(22,704,64))
        self.assertEqual(json.loads(files["SWIFT_SKIP_CPU.json"]),skip_cpu.binding())
        self.assertIn(b"skip_cpu.verify(here,pin)",files["owner_gate.py"])
        self.assertIn(b"EXACT_4790_ZERO_GRES_TERMINAL",files["owner_gate.py"])
        self.assertIn(b"port=29997",files["owner_gate.py"])
        self.assertIn(b"EXACT_4726_SKIPPED_EPOCH1_RAW_POSITION",files["series_live.py"])
        self.assertIn(b"swift_watch.wrap_native_context(SwiftMixin,pin)",files["callback.py"])

    def test_actual_4790_receipts_and_wrong_cpu_evidence(self):
        self.assertEqual(prepare.checked_skip_evidence(),skip_cpu.binding())
        raw=json.loads(prepare.SKIP_TRANSPORT.read_bytes())["response"]
        result=dict(raw["output/result.json"]["fields"],request=raw["request"])
        skip_cpu.validate_result(result)
        for key,value in (("actual_native_epoch1_skipped_dataset_fetches",1),
                          ("actual_loader_constructions_per_rank",1),
                          ("production_worker_behavior_verified",True)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,"4790_NATIVE_SKIP_CPU_RESULT"):
                skip_cpu.validate_result(dict(result,**{key:value}))


if __name__=="__main__":
    unittest.main()
