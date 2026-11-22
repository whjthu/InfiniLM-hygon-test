"""CPU recording of the native observation and stop boundary, without Torch."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from itertools import count
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import compare_first as reader
from probe import FirstForward, await_owned_cohort, profile_operators


class Module:
    training=True

    def __init__(self):
        self.hooks=[]

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        item=(hook,with_kwargs)
        self.hooks.insert(0,item) if prepend else self.hooks.append(item)
        return NS(remove=lambda:self.hooks.remove(item))

    def observe(self):
        for hook,with_kwargs in self.hooks:
            hook(self,(),{}) if with_kwargs else hook(self,())


class Linear(Module):
    pass


class Dropout(Module):
    p=.05
    inplace=False


class Model(Module):
    def __init__(self):
        super().__init__()
        self.language=Module()
        self.language.config=NS(_attn_implementation="sdpa")
        self.linear=Linear()
        self.dropout=Dropout()
        self.model=NS(language_model=self.language)

    def get_base_model(self):
        return self

    def named_modules(self):
        return (("",self),("llm",self.language),("q",self.linear),("dropout",self.dropout))

    def parameters(self):
        return [NS(requires_grad=True,grad=None)]


STATE=dict(adapter={"SHA256":"a"},optimizer={"SHA256":"o"},scheduler={"SHA256":"s"})


class Witness:
    def __init__(self):
        self.inputs,self.outputs,self.steps=[],[],[]
        self.rng_loads=[dict(actual={"generation":0})]
        self.expected=STATE
        self.records={}
        self.failed=False

    def training_state(self):
        return STATE

    def detach(self):
        pass

    def write(self,name,value):
        self.records[name]=deepcopy(value)

    def fail(self,error):
        self.failed=True


class Trainer:
    def __init__(self,fixture):
        self.fixture=fixture
        self.state=NS(global_step=20)
        self.model_wrapped=NS(require_backward_grad_sync=False,broadcast_buffers=True)
        self.accelerator=NS(sync_gradients=False)
        self.backward=self.updates=0

    def compute_loss(self):
        f=self.fixture
        f.rng_generation=1
        f.model.observe()
        f.autocast=True
        f.model.language.observe()
        f.model.linear.observe()
        f.autocast=False
        f.rng_generation=2
        f.witness.inputs.append(dict(ordinal=1,global_step=20,runtime_cursor=dict(micro_in_update=1),full_input={"SHA256":"ids"}))
        f.witness.outputs.append(dict(ordinal=1,loss={"SHA256":"loss"},loss_scalar=2.5))

    def training_step(self):
        self.compute_loss()
        self.backward+=1
        self.updates+=1


class Fixture:
    def __init__(self, extra=(), fail_cohort=False):
        self.model=Model()
        self.trainer=Trainer(self)
        self.witness=Witness()
        self.autocast=False
        self.rng_generation=0
        self.cohort_calls=0
        self.fail_cohort=fail_cohort
        self.rank,self.skip_seen=0,True
        cuda=NS(matmul=NS(allow_tf32=False,allow_bf16_reduced_precision_reduction=True),
                math_sdp_enabled=lambda:True,flash_sdp_enabled=lambda:True)
        torch=NS(nn=NS(Linear=Linear,modules=NS(dropout=NS(_DropoutNd=Dropout))),
                 backends=NS(cuda=cuda),is_grad_enabled=lambda:True,is_inference_mode_enabled=lambda:False,
                 is_autocast_enabled=lambda dev:self.autocast,get_autocast_dtype=lambda dev:"BF16",
                 are_deterministic_algorithms_enabled=lambda:False,get_float32_matmul_precision=lambda:"highest")
        events=[NS(key="aten::_scaled_dot_product_attention_math",count=32),NS(key="aten::dropout",count=128),*extra]
        @contextmanager
        def profiler():
            yield NS(key_averages=lambda:events)
        self.observer=FirstForward(self,torch,lambda:dict(generation=self.rng_generation),profiler,self.cohort)
        self.observer.attach()

    def check_time(self):
        pass

    def cohort(self):
        self.cohort_calls+=1
        if self.fail_cohort:
            raise ValueError("missing rank")


class Local(unittest.TestCase):
    def test_first_real_method_return_stops_before_native_backward(self):
        fixture=Fixture()
        with self.assertRaises(SystemExit) as stop:
            fixture.trainer.training_step()
        self.assertEqual(stop.exception.code,0)
        self.assertEqual((fixture.trainer.backward,fixture.trainer.updates,fixture.cohort_calls),(0,0,1))
        self.assertNotIn("compute_loss",fixture.trainer.__dict__)
        self.assertFalse(any(m.hooks for _,m in fixture.model.named_modules()))
        row=fixture.witness.records["first-forward-state.json"]
        self.assertEqual((row["RNG_before_compute_loss"]["generation"],row["root_before_forward"]["RNG"]["generation"],
                          row["RNG_after_forward"]["generation"]),(0,1,2))
        self.assertFalse(row["root_before_forward"]["context"]["autocast_enabled"])
        self.assertTrue(row["language_before_forward"]["context"]["autocast_enabled"])
        self.assertTrue(row["inner_linear"]["context"]["autocast_enabled"])
        self.assertEqual(row["operators"]["selected_SDPA"],{"aten::_scaled_dot_product_attention_math":32})
        self.assertEqual(row["operators"]["native_dropout"],{"aten::dropout":128})

    def test_unknown_SDPA_and_backward_fail_closed(self):
        result=profile_operators([NS(key="aten::scaled_dot_product_attention",count=1)])
        self.assertFalse(result["SDPA_route_observed"])
        error=Fixture(extra=(NS(key="autograd::engine::evaluate_function: AddBackward",count=1),))
        with self.assertRaisesRegex(ValueError,"backward observed"):
            error.trainer.training_step()
        self.assertTrue(error.witness.failed)
        self.assertNotIn("first-forward-stop.json",error.witness.records)
        self.assertEqual(error.trainer.backward,0)

    def test_partial_cohort_retains_first_forward_but_cannot_qualify_stop(self):
        fixture=Fixture(fail_cohort=True)
        with self.assertRaisesRegex(ValueError,"missing rank"):
            fixture.trainer.training_step()
        self.assertIn("first-forward-state.json",fixture.witness.records)
        self.assertNotIn("first-forward-stop.json",fixture.witness.records)

    def test_per_rank_comparison_does_not_invent_historic_context(self):
        fixture=Fixture()
        with self.assertRaises(SystemExit):
            fixture.trainer.training_step()
        row=fixture.witness.records["first-forward-state.json"]
        rows={rank:dict(row,rank=rank) for rank in range(8)}
        historic={rank:dict(input=row["actual_input"]["full_input"],state=STATE,RNG=row["RNG_after_native_restore"],
                           resumed_loss=row["actual_loss"],continuous_loss=dict(row["actual_loss"],loss_scalar=2.49999))
                  for rank in range(8)}
        repeat=deepcopy(rows)
        repeat[7]["operators"]["selected_SDPA"]={"aten::_scaled_dot_product_flash_attention":32}
        result=reader.compare(rows,historic,repeat)
        self.assertTrue(all(r["resumed_loss_match"] and not r["continuous_loss_match"] for r in result["rows"]))
        self.assertEqual(result["rows"][7]["repeat_differences"],["operators"])
        self.assertFalse(result["continuous_context_recovered"] or result["formal_recovery_qualified"])
        repeat[7]["actual_input"]={"full_input":{"SHA256":"wrong"}}
        with self.assertRaisesRegex(ValueError,"repeat input"):
            reader.compare(rows,historic,repeat)

    def test_original_public_byte_pin_and_terminal_no_replay(self):
        path=HERE.parents[2]/"results/bw1000/2026-11-22/training-minicpm-native88-structured-v3-1171-paired-recovery-terminal.json"
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),reader.ORIGINAL_PUBLIC)
        proof=reader.read(path,3271,reader.ORIGINAL_PUBLIC)
        self.assertEqual(proof["original_comparison_SHA256"],reader.ORIGINAL_COMPARISON)
        self.assertIs(proof["original_replay_allowed"],False)

    def test_eight_rank_ready_gate_and_missing_peer_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ranks=[root/f"rank-{rank:02d}" for rank in range(8)]
            for rank in ranks:
                rank.mkdir()
            identity={"step":"fixture","config_SHA256":"a"*64}
            with ThreadPoolExecutor(max_workers=8) as pool:
                readers=[pool.submit(await_owned_cohort,path,rank,identity,lambda:None,seconds=3)
                         for rank,path in enumerate(ranks)]
                for reader in readers:
                    reader.result(timeout=4)
            self.assertEqual(len(list(root.glob("rank-*/first-forward-ready.json"))),8)
        with tempfile.TemporaryDirectory() as directory:
            rank=Path(directory)/"rank-00"
            rank.mkdir()
            tick=count()
            with self.assertRaisesRegex(TimeoutError,"missing original zero-update ranks"):
                await_owned_cohort(rank,0,identity,lambda:None,seconds=1,clock=lambda:next(tick))


if __name__=="__main__":
    unittest.main(verbosity=2)
