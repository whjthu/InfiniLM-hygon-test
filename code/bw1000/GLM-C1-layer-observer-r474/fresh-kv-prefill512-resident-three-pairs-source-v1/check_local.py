"""Run the actual new caller and reducer with an independent CPU recording context."""

from copy import deepcopy
from hashlib import sha256
from itertools import count
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import caller
from aggregate import summarize

base=caller.prior
IDS=tuple(range(1312))
HASH=base.canonical_hash(IDS)
PCI={rank:f"0000:{rank%8:02x}:00.0" for rank in range(16)}
GOLD=b"\x00\x3f"*base.VOCAB
BAD=b"\x00\x40"*base.VOCAB
COUNTS=dict(attempted=78,packed=78,rotary_layout=0,tensor_layout=0,address_span=0)
LOAD=dict(context_create_attempts=1,context_creations=1,checkpoint_load_attempts=1,checkpoint_loads=1,
          loaded_parameters=base.LOAD_KEYS,checkpoint_key_count=base.LOAD_KEYS,
          all78layers_loaded=True,actual_layers=list(range(78)))
DRIVER=dict(samples=10,min_observed=base.FLOOR,errors=[])


class Owner:
    def __init__(self, rank, fail_at=None, fallback_at=None):
        self.rank,self.past,self.ready,self.ctx=rank,0,True,2468
        self.fail_at,self.fallback_at=fail_at,fallback_at
        self.calls=[]
        self.resets=self.closes=self.loads=0

    def owned(self):
        if not self.ctx:
            raise ValueError("closed")

    def geometry(self):
        return dict(TP=16,PP=1,rows=1,capacity_per_row=32768,world_rank=self.rank,tp_rank=self.rank,
                    root_output_rank=0,local_attention_heads=4,local_index_heads=2,loaded_parameters=base.LOAD_KEYS,
                    Model_READY=True,weights_loaded=True,Model_constructed=True,cache_allocated=True,poisoned=False,
                    replicated_embedding_BF16_bytes=0,embedding_BF16_local_bytes=118947840,
                    embedding_vocab_begin=self.rank*9680,embedding_vocab_end=(self.rank+1)*9680)

    def forward(self, n, past, ids, pos, logits, stream, *, phase, prefill512):
        self.calls.append((n,past,ids,pos,logits,stream,phase,prefill512,self.ctx))
        if len(self.calls)==self.fail_at:
            self.last_prefill512=dict(COUNTS,attempted=8,packed=8)
            raise RuntimeError("recorded Native forward failure")
        self.past+=n
        if prefill512:
            return dict(COUNTS,packed=77,rotary_layout=1) if len(self.calls)==self.fallback_at else dict(COUNTS)

    def reset(self):
        self.resets+=1
        self.past=0

    def close(self):
        self.closes+=1
        self.ctx,self.ready=None,False


class Buffers:
    def __init__(self, rank, wrong=()):
        self.rank,self.wrong=rank,wrong
        self.audits=self.reads=self.poison=0

    def physical_pci(self):
        return PCI[self.rank]

    def pointers(self, past, n):
        return (10000+past*8,20000+past*8,30000,40000)

    def verify_inputs(self, ids):
        if ids!=IDS:
            raise ValueError("fixture original input differs")
        self.audits+=1

    def poison_root_logits(self):
        self.poison+=1

    def read_root_logits(self):
        self.reads+=1
        return BAD if self.reads in self.wrong else GOLD


def execute(owner, *, io=None, reports=None, provider=None, memory=base.FLOOR):
    rank=owner.rank
    io=io or Buffers(rank)
    phases=[]
    ticks=count(rank+1,rank+1)

    def fixture_load(_, *, gate, load_report, **kwargs):
        owner.loads+=1
        load_report.update(LOAD)
        gate("prefill_pair_created")
        return owner

    with patch.multiple(base,PROMPT_SHA=HASH,DeviceIO=lambda *a:io,physical_pci=lambda dev:PCI[rank],load_owner=fixture_load), \
         patch.object(caller.binding,"load_input",return_value=IDS):
        result=caller.run_owner(node_index=rank//8,device=rank%8,expected_pci=PCI[rank],world_uid=b"w",tp_uid=b"t",
                                torch=None,safe_open=None,input_binding=None,loader_path=None,
                                native=SimpleNamespace(SHA256=base.NATIVE_SHA),providers=caller.PROVIDERS if provider is None else provider,
                                gate=phases.append,memory=lambda label:(memory,64*1024**3),memory_summary=lambda:DRIVER,
                                deadline=lambda label:None,emit=lambda row:reports.append(deepcopy(row)) if reports is not None else None,
                                mark=lambda *a,**kw:None,clock=lambda:next(ticks))
    return result,phases


def reduce(records):
    with patch.object(base,"PROMPT_SHA",HASH):
        return summarize(records,PCI,IDS)


class Local(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort={r:execute(Owner(r))[0] for r in range(16)}

    def test_records_terminal_original_scope_as_source_evidence(self):
        result=HERE.parents[3]/"results/bw1000/2026-11-22/GLM474-prefill512-resident-v5-original1157-terminal.json"
        raw=result.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(),caller.PRIOR_RESULT_SHA)
        baseline=json.loads(raw)["independent_actual_evidence"]
        self.assertEqual((baseline["forwards_completed"],baseline["route_layers_packed"],baseline["rank0_bitwise_fragments"]),
                         (96,1248,3))
        self.assertTrue(baseline["pair_evidence_pass"])

    def test_real_caller_one_context_no_extra_warmup_and_ordered_gates(self):
        owner=Owner(0)
        io=Buffers(0)
        result,phases=execute(owner,io=io)
        self.assertEqual(phases,list(caller.PHASES))
        self.assertEqual((owner.loads,owner.closes,owner.resets,result["attempted_forwards"]),(1,1,7,18))
        self.assertEqual([i+1 for i,row in enumerate(owner.calls) if row[7]],[4,7,16])
        self.assertEqual({row[-1] for row in owner.calls},{2468})
        self.assertEqual([row[:2] for row in owner.calls],[(n,past) for past,n in caller.CHUNKS]*6)
        self.assertEqual((io.audits,io.reads,io.poison),(7,18,18))
        self.assertEqual([[(i["past"],i["bitwise_equal_to_pair_first"]) for i in r["fragments"]]
                          for r in result["arms"] if r["position"]==2],
                         [[(past,True) for past,n in caller.CHUNKS]]*3)

    def test_complete_clock_position_report_and_full_denominators(self):
        result=reduce(self.cohort)
        self.assertTrue(result["prefill_numerical_evidence_pass"] and result["rank0_OFF_outputs_stable"])
        self.assertEqual((result["owner_forwards_planned"],result["route_layers_planned"],result["root_bitwise_fragments_planned"]),
                         (288,3744,9))
        self.assertEqual([(row["returned"],row["packed"],row["rank0_bitwise_fragments"]) for row in result["pairs"]],
                         [(96,1248,3)]*3)
        timing=result["timing_evidence"]
        self.assertEqual([row["pair"] for row in timing["positions"][1]["ON"]],[2])
        self.assertEqual([row["pair"] for row in timing["positions"][2]["ON"]],[1,3])
        self.assertEqual(timing["pairs"][1]["arms"][0]["slowest_rank"]["sum_three_forward_ns"],dict(rank=15,ns=48))
        self.assertEqual(timing["median_ON_over_OFF"]["first_forward_ns"],1)
        self.assertFalse(result["decode_qualified"] or result["model_speed_win"])

    def test_reverse_position_comparison_and_repeated_off_drift_veto_timing(self):
        changed=execute(Owner(0),io=Buffers(0,wrong=(7,)))[0]
        result=reduce(dict(self.cohort)|{0:changed})
        self.assertEqual([r["rank0_bitwise_fragments"] for r in result["pairs"]],[3,2,3])
        self.assertTrue(result["rank0_OFF_outputs_stable"])
        self.assertIsNone(result["timing_evidence"])
        drift=execute(Owner(0),io=Buffers(0,wrong=(7,10)))[0]
        result=reduce(dict(self.cohort)|{0:drift})
        self.assertEqual([r["rank0_bitwise_fragments"] for r in result["pairs"]],[3,3,3])
        self.assertFalse(result["rank0_OFF_outputs_stable"])
        self.assertIsNone(result["timing_evidence"])

    def test_native_failure_and_missing_owner_keep_original_plan(self):
        broken=Owner(11,fail_at=7)
        saved=[]
        with self.assertRaisesRegex(RuntimeError,"recorded Native"):
            execute(broken,reports=saved)
        failed=saved[-1]
        self.assertEqual((failed["attempted_forwards"],failed["successful_forwards"],broken.resets,broken.closes),(7,6,3,0))
        self.assertEqual(failed["arms"][2]["fragments"][0]["observed_route_counts"]["attempted"],8)
        records=dict(self.cohort)|{11:failed}
        del records[7]
        result=reduce(records)
        self.assertEqual([r["returned"] for r in result["pairs"]],[90,84,84])
        self.assertEqual([r["attempted"] for r in result["pairs"]],[90,85,84])
        self.assertEqual(result["owner_forwards_planned"],288)
        self.assertEqual(len(result["rejected"]),2)
        self.assertIsNone(result["timing_evidence"])

    def test_fallback_provider_and_floor_do_not_qualify(self):
        fallback=execute(Owner(11,fallback_at=7))[0]
        result=reduce(dict(self.cohort)|{11:fallback})
        self.assertEqual([r["packed"] for r in result["pairs"]],[1248,1247,1248])
        self.assertFalse(result["prefill_numerical_evidence_pass"])
        self.assertIsNone(result["timing_evidence"])
        bad=Owner(3)
        with self.assertRaisesRegex(ValueError,"provider"):
            execute(bad,provider=dict(caller.PROVIDERS,confirmed_providers=70))
        self.assertEqual(bad.loads,0)
        with self.assertRaisesRegex(ValueError,"8GiB"):
            execute(bad,memory=base.FLOOR-1)


if __name__=="__main__":
    unittest.main(verbosity=2)
