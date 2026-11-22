"""Exercise the real caller and reducer using local CPU recording objects."""

import copy
from hashlib import sha256
from itertools import count
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))

import aggregate
import bindings
import run_pair as pair

SOURCE = HERE.parent / "full-model-screen32-v1"
ACTUAL = ROOT / "results/bw1000/2026-11-22/GLM474-t512-native-original1071-host-verified.json"
IDS = tuple(range(1312))
IDS_SHA = bindings.canonical_hash(IDS)
PCI = {rank: f"0000:{rank%8:02x}:00.0" for rank in range(16)}
GOLD = b"\x00\x3f" * bindings.VOCAB
BAD = b"\x00\x40" * bindings.VOCAB
LOAD = dict(context_create_attempts=1, context_creations=1, checkpoint_load_attempts=1, checkpoint_loads=1,
            checkpoint_key_count=bindings.LOAD_KEYS, loaded_parameters=bindings.LOAD_KEYS, all78layers_loaded=True,
            actual_layers=list(range(78)))
COUNTERS = dict(attempted=78,packed=78,rotary_layout=0,tensor_layout=0,address_span=0)
DRIVER = dict(samples=10,min_observed=pair.FLOOR,errors=[])


class RecordingOwner:
    def __init__(self, rank, fail=False, fallback=False):
        self.rank, self.past, self.ready, self.ctx = rank, 0, True, 56789
        self.calls = []
        self.resets = 0
        self.fail, self.fallback = fail, fallback

    def owned(self):
        return self.ctx is not None

    def geometry(self):
        return dict(TP=16, PP=1, rows=1, capacity_per_row=32768, world_rank=self.rank, tp_rank=self.rank,
                    root_output_rank=0, local_attention_heads=4, local_index_heads=2, loaded_parameters=bindings.LOAD_KEYS,
                    Model_READY=True, weights_loaded=True, Model_constructed=True, cache_allocated=True,
                    poisoned=False, replicated_embedding_BF16_bytes=0, embedding_BF16_local_bytes=118947840,
                    embedding_vocab_begin=self.rank*9680, embedding_vocab_end=(self.rank+1)*9680)

    def forward(self, n, past, ids, pos, logits, stream, *, phase, prefill512):
        self.calls.append((n,past,ids,pos,logits,stream,phase,prefill512))
        if prefill512 and self.fail:
            self.last_prefill512 = dict(COUNTERS,attempted=8,packed=8)
            raise RuntimeError("recorded Native failure")
        self.past += n
        if prefill512:
            return dict(COUNTERS,packed=77,rotary_layout=1) if self.fallback else dict(COUNTERS)

    def reset(self):
        self.resets += 1
        self.past = 0

    def close(self):
        self.ctx, self.ready = None, False


class RecordingIO:
    def __init__(self, rank, changed=False, poisoned=False):
        self.rank, self.changed, self.poisoned = rank, changed, poisoned
        self.reads, self.poison = 0, 0

    def physical_pci(self):
        return PCI[self.rank]

    def pointers(self, past, n):
        return 100000+8*past, 200000+8*past, 300000, 400000

    def verify_inputs(self, ids):
        assert ids == IDS

    def poison_root_logits(self):
        self.poison += 1

    def read_root_logits(self):
        self.reads += 1
        if self.poisoned:
            return b"\x80\x7f" * bindings.VOCAB
        return BAD if self.changed and self.reads == 6 else GOLD


def invoke(rank, *, io=None, ctx=None, pci=None, mem=(pair.FLOOR, 64*1024**3), stages=None):
    owner = ctx or RecordingOwner(rank)
    buffers = io or RecordingIO(rank)
    ticks = count(10,10)
    with patch.object(pair,"PROMPT_SHA",IDS_SHA):
        result = pair.run_loaded(owner,buffers,IDS,LOAD,rank=rank,native_sha=bindings.NATIVE_SHA,
                                 expected_pci=pci or PCI[rank], gate=lambda label: None, memory=lambda label: mem,
                                 deadline=lambda label: None,
                                 emit=lambda evidence: stages.append(copy.deepcopy(evidence)) if stages is not None else None,
                                 clock=lambda: next(ticks))
    return owner, buffers, result


def finished(rank, **kwargs):
    _, _, result = invoke(rank,**kwargs)
    result.update(state="CLOSED",normal_close=True,load_progress=LOAD,memory_after_close=result["final_memory"],
                  driver_memory=DRIVER)
    return result


class Local(unittest.TestCase):
    def test_pinned_original_artifacts(self):
        actual = bindings.read(ACTUAL,bindings.NATIVE_CPU_EVIDENCE_SHA,bindings.NATIVE_CPU_EVIDENCE_SIZE,
                               bindings.NATIVE_CPU_EVIDENCE_SIZE)
        self.assertEqual(bindings.verify_cpu(actual)["strong_exports"],285)
        public = json.loads(bindings.read(SOURCE/"INPUT_BINDING.json",bindings.INPUT_SHA,bindings.INPUT_SIZE,
                                          bindings.INPUT_SIZE))
        case = public["private_plan"]["requests"][0]
        self.assertEqual((case["ordinal"],case["prompt_tokens"],case["prompt_IDs_SHA256"]),
                         (1,1312,bindings.PROMPT_SHA))
        self.assertTrue(callable(bindings.load_checkpoint_function(SOURCE/"checkpoint.py")))
        rendered = HERE.parent / "fresh-kv-prefill512-caller-source-v1" / "render.py"
        namespace = {"__file__": str(rendered), "__name__": "local_reviewed_render"}
        exec(compile(rendered.read_bytes(),str(rendered),"exec"),namespace)
        typed = namespace["sources"]()["typed_binding.py"]
        self.assertEqual((sha256(typed).hexdigest(),len(typed)),(bindings.TYPED_SHA,bindings.TYPED_SIZE))
        scope = json.loads((HERE/"SCOPE.json").read_bytes())
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{32}",scope["reserved_source_identity"]))
        self.assertEqual(scope["fragments"],[list(chunk) for chunk in pair.CHUNKS])
        self.assertFalse(scope["GPU_GO"] or scope["Model_GO"])

    def test_all_nine_mounted_file_readers_with_exact_config_and_index_sizes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            model = root/"model"
            model.mkdir()
            config = model/"config.json"
            index = model/"model.safetensors.index.json"
            plan = root/"private-plan.json"
            native = root/"native.so"
            recipe = root/"recipe.json"

            def save(path, value, size):
                data = json.dumps(value,separators=(",",":")).encode()
                self.assertLess(len(data),size)
                raw = data.ljust(size,b" ")
                path.write_bytes(raw)
                return sha256(raw).hexdigest()

            config_sha = save(config,{"num_hidden_layers":78,"vocab_size":154880},bindings.CONFIG_SIZE)
            mapping = {f"model.weight_{n}":f"model-{n % bindings.SHARDS}.safetensors" for n in range(bindings.INDEX_KEYS)}
            index_sha = save(index,{"weight_map":mapping},bindings.INDEX_SIZE)
            row = dict(ordinal=1,case_id="fixture",suite="GSM",prompt_tokens=len(IDS),prompt_IDs_SHA256=IDS_SHA,
                       original_private_SHA256="0"*64,body_SHA256="1"*64,max_new_tokens=4096,prefill_window=4096)
            plan_sha = save(plan,dict(planned=32,requests=[dict(row,prompt_token_ids=IDS,body={"temperature":0.0}),
                            *[{"ordinal":n} for n in range(2,33)]]),bindings.PLAN_SIZE)
            public_sha = save(root/"INPUT_BINDING.json",dict(model_path=model.as_posix(),model_config_SHA256=config_sha,
                              model_index_SHA256=index_sha,model_index_keys=bindings.INDEX_KEYS,model_checkpoint_files=bindings.SHARDS,
                              private_plan=dict(SHA256=plan_sha,bytes=bindings.PLAN_SIZE,requests=[row]),
                              selected_ordinals=list(range(1,33))),bindings.INPUT_SIZE)
            for name in ("checkpoint.py","typed_binding.py"):
                shutil_bytes = ((SOURCE/name).read_bytes() if name == "checkpoint.py"
                                else b"t"*bindings.TYPED_SIZE)
                (root/name).write_bytes(shutil_bytes)
            elf = b"\x7fELF".ljust(bindings.NATIVE_SIZE,b"\x00")
            native.write_bytes(elf)
            native_sha = sha256(elf).hexdigest()
            cpu = json.loads(ACTUAL.read_bytes())
            cpu["native"]["SHA256"] = native_sha
            cpu_sha = save(root/"1071.json",cpu,bindings.NATIVE_CPU_EVIDENCE_SIZE)
            recipe_sha = save(recipe,dict(linked_library=str(native),expected_providers={str(n):"0"*64 for n in range(71)}),
                              bindings.RECIPE_SIZE)
            with patch.multiple(bindings,MODEL=model,CONFIG_SHA=config_sha,INDEX_SHA=index_sha,PLAN=plan,PLAN_SHA=plan_sha,
                                INPUT_SHA=public_sha,PROMPT_SHA=IDS_SHA,NATIVE=native,NATIVE_SHA=native_sha,
                                NATIVE_CPU_EVIDENCE_SHA=cpu_sha,RECIPE=recipe,RECIPE_SHA=recipe_sha,
                                TYPED_SHA=sha256((root/"typed_binding.py").read_bytes()).hexdigest()), \
                 patch.object(bindings.ctypes,"CDLL",side_effect=AssertionError("unexpected Native load")), \
                 patch.object(bindings,"module",side_effect=AssertionError("unexpected pinned source execution")):
                evidence=bindings.verify_mounted_inputs(root)
                self.assertEqual((len(evidence["files"]),evidence["prompt_tokens"],evidence["expected_provider_paths"]),(9,1312,71))
                self.assertEqual(evidence["files"]["config"]["bytes"],29464)
                self.assertEqual(evidence["files"]["checkpoint_index"]["bytes"],11359251)
                self.assertFalse(evidence["native_loaded"] or evidence["checkpoint_weights_read"])
                short=config.read_bytes()[:-1]
                config.write_bytes(short)
                with patch.object(bindings,"CONFIG_SHA",sha256(short).hexdigest()):
                    with self.assertRaisesRegex(ValueError,"SHA/size differs: config.json"):
                        bindings.verify_mounted_inputs(root)

    def test_same_context_full_schedule_and_unchanged_later_fragments(self):
        owner, io, result = invoke(0)
        self.assertEqual((owner.resets, io.poison, io.reads), (3,6,6))
        self.assertEqual([(r[1],r[0]) for r in owner.calls],list(pair.CHUNKS)*2)
        self.assertEqual([r[-1] for r in owner.calls], [False,False,False,True,False,False])
        self.assertEqual([r[2:7] for r in owner.calls[:3]], [r[2:7] for r in owner.calls[3:]])
        self.assertTrue(result["root_logits_equal"] and result["all_local_layers_packed"])

    def test_real_scope_entry_closes_once(self):
        owner = RecordingOwner(2)
        stages = []
        def qualified_load(_, **kwargs):
            kwargs["load_report"].update(LOAD)
            return owner
        ticks = count(10,10)
        with patch.multiple(pair,PROMPT_SHA=IDS_SHA,load_input=lambda _: IDS,load_owner=qualified_load,
                            physical_pci=lambda device: PCI[device],
                            DeviceIO=lambda *args: RecordingIO(2)):
            report = pair.run_owner(node_index=0,device=2,expected_pci=PCI[2],world_uid=b"a",tp_uid=b"b",torch=None,
                                    safe_open=None,native=SimpleNamespace(SHA256=bindings.NATIVE_SHA),
                                    input_binding=None,loader_path=None,gate=lambda label: None,
                                    memory=lambda label: (pair.FLOOR,64*1024**3),deadline=lambda label: None,
                                    memory_summary=lambda: DRIVER, clock=lambda: next(ticks),
                                    emit=lambda item: stages.append(copy.deepcopy(item)),mark=lambda *a, **kw: None)
        self.assertEqual((stages[0]["state"],report["state"],owner.ctx),("BOUND","CLOSED",None))
        self.assertTrue(report["normal_close"])

    def test_failed_candidate_preserves_attempts_and_never_resets_on_error(self):
        owner = RecordingOwner(0,fail=True)
        stages = []
        with self.assertRaisesRegex(RuntimeError,"recorded Native failure"):
            invoke(0,ctx=owner,stages=stages)
        result = stages[-1]
        self.assertEqual((result["state"],result["attempted_forwards"],result["successful_forwards"],owner.resets),
                         ("FAILED",4,3,2))
        self.assertEqual(result["arms"][1]["fragments"][0]["observed_route_counts"]["packed"],8)
        self.assertEqual(len(result["plans"]),6)

    def test_identity_and_floor_and_finite_output_fail_closed(self):
        for kwargs,message in [
            ({"pci":"0000:7f:00.0"},"PCI"),
            ({"mem":(pair.FLOOR-1,64*1024**3)},"floor"),
            ({"io":RecordingIO(0,poisoned=True)},"complete finite"),
        ]:
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):
                invoke(0,**kwargs)

    def test_all_owner_denominators_and_cpu_recorded_faults(self):
        reports = {rank: finished(rank) for rank in range(16)}
        with patch.object(aggregate,"PROMPT_SHA",IDS_SHA):
            result = aggregate.summarize(reports,PCI)
            self.assertTrue(result["prefill_pair_evidence_pass"])
            self.assertEqual((result["owners_verified"], result["forwards_planned"], result["route_layers_packed"]),
                             (16,96,1248))
            fallback = dict(reports)
            fallback[7] = finished(7,ctx=RecordingOwner(7,fallback=True))
            result = aggregate.summarize(fallback,PCI)
            self.assertEqual((result["owners_verified"],result["route_layers_packed"]),(16,1247))
            self.assertFalse(result["prefill_pair_evidence_pass"])
            changed = dict(reports)
            changed[0] = finished(0,io=RecordingIO(0,changed=True))
            self.assertEqual(aggregate.summarize(changed,PCI)["rank0_bitwise_fragments"],2)
            low = dict(reports)
            low[3] = dict(low[3],driver_memory=dict(DRIVER,min_observed=pair.FLOOR-1))
            self.assertIn("8GiB",aggregate.summarize(low,PCI)["rejected"][0]["reason"])
            del reports[11]
            self.assertEqual(aggregate.summarize(reports,PCI)["rejected"],[{"rank":11,"reason":"MISSING"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
