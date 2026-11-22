"""Exercise the fixed changeovers, unchanged unit work and independent pulse comparison."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import unittest

import contract as c
import launch
import package
from package import REPO, source
from scope import LABEL, REQUEST, PARENTS, RUN
import transition_read as reader
import test_report_training80_backlog as dispatch_fixture


PLAN = dict(request=REQUEST,campaign_epoch=1000.,shared_start_epoch=1082.,deadline_epoch=1480.)
CARDS = [f"0000:{i:02x}:00.0" for i in range(8)]


def inventory():
    return {f"mock{p}n{i:02d}":dict(parent=parent,PCI=CARDS) for p,parent in enumerate(PARENTS) for i in range(40)}


def pulse(plan, phase):
    binding=reader.schedule(plan)[phase]
    request=binding["request"]
    identities=inventory()
    steps={parent:parent+".200" for parent in PARENTS}
    points=[]
    for seq,target in enumerate(binding["target_epoch_ns"]):
        nonce=f"{seq+1:032x}"
        quiet=(35 if phase=="aligned" else 25) if seq==2 else 0
        reports=[dict(node=node,parent=item["parent"],step=steps[item["parent"]],sequence=seq,nonce=nonce,
                      request=request,busy=dict.fromkeys(CARDS,0 if i<quiet else 90),failed_pci=[])
                 for i,(node,item) in enumerate(identities.items())]
        points.append(dict(sequence=seq,target_epoch_ns=target,controller_send_epoch_ns=target+1_000,
                           complete=True,valid_samples=640,missing_nodes=[],sample_span_upper_bound_ns=8_000_000,
                           worker_reports=reports,nonce=nonce))
    return dict(request=request,complete=True,counter="sysfs_gpu_busy_percent",GPU_requested=0,
                clock="controller_monotonic_send_to_last_receipt_upper_bound",planned_nodes=80,
                expected_unique_node_PCI=640,observed_unique_node_PCI=640,expected_samples=3200,valid_samples=3200,
                planned_pulse_interval_ns=4*c.NANO,points=points,accounting_qualified=True,failures=[],
                actual_accounting={parent:[handle,f"training80-busy-pulse-{request}-{parent}","COMPLETED","0:0",
                                           "20","fixture","cpu=40,mem=10G,node=40"] for parent,handle in steps.items()})


class Stagger(unittest.TestCase):
    def test_all_nine_combinations_use_the_same_five_complete_segments(self):
        for cohort in range(9):
            rows=c.grid(PLAN,cohort)
            offsets=[start//c.NANO-1000 for _,start,_ in rows]
            first=c.FIRST_ACTIVE[cohort//3]
            self.assertEqual(offsets[:3],[first,first+64,first+128])
            self.assertEqual(offsets[3:],[339+c.TRANSITION_SHIFTS[cohort%3],403+c.TRANSITION_SHIFTS[cohort%3]])
            self.assertTrue(all(end-start==60*c.NANO for _,start,end in rows))
            self.assertLessEqual(rows[-1][2],1471*c.NANO)
            self.assertEqual([c.counters(mode,10) for mode,_,_ in rows],[c.counters("prelude",10)]*5)
        self.assertEqual(c.counters("aligned_before",1)["stream_event_waits"],0)
        self.assertEqual(c.counters("staggered_after",1)["d2h_payload_bytes"],761856)

    def test_assigns_whole_shards_and_balances_selected_nodes(self):
        shards={}
        for i in range(31):
            nodes=[f"mock{i}n{j}" for j in range(3 if i<12 else 2)]
            shards[f"label{i:02d}"]={"nodes":nodes,"parent":PARENTS[i%2]}
        assigned=c.assign_cohorts(shards)
        load=[sum(len(shards[label]["nodes"]) for label,cohort in assigned.items() if cohort%3==group)
              for group in range(3)]
        self.assertEqual(set(shards),set(assigned))
        self.assertEqual(sum(load),74)
        self.assertLessEqual(max(load)-min(load),3)
        self.assertTrue(all(c.setup_cohort(cohort) in (1,2) for cohort in assigned.values()))

    def test_aligned_and_staggered_physical_samples_on_the_same_devices(self):
        plan=dict(PLAN,selected=dict(list(inventory().items())[:74]),cohort_by_shard={"a":1,"b":2,"c":3})
        term=dict(request=REQUEST,campaign_epoch=1000.,planned_nodes=74,planned_cards=592,
                  device_qualified=True,valid_cards=592,completed_arms=2960,
                  actual_busy_card_samples=35520,nodes=plan["selected"])
        pulses={phase:pulse(plan,phase) for phase in ("aligned","staggered")}
        result=reader.summarize(plan,term,pulses)
        self.assertEqual(result["qualified_physical_samples"],6400)
        self.assertEqual(result["paired_sampled_transition_delta"]["worst_selected_nonzero_delta"],80)
        self.assertFalse(result["formal_sustained_busy_qualified"])
        incomplete=deepcopy(pulses)
        incomplete["aligned"]["points"][2]["sample_span_upper_bound_ns"]=250_000_001
        result=reader.summarize(plan,term,incomplete)
        self.assertEqual(result["qualified_physical_samples"],3200)
        self.assertIsNone(result["paired_sampled_transition_delta"])
        term["actual_busy_card_samples"]-=1
        self.assertIsNone(reader.summarize(plan,term,pulses)["paired_sampled_transition_delta"])

    def test_staged_sources_and_original_pins_build(self):
        _, files=source(REPO)
        self.assertIn("transition_pulse.py",files)
        self.assertIn("transition_read.py",files)
        self.assertNotIn(b'StreamWaitEvent',files["worker.py"])
        manifest=json.loads(files["SOURCE_FILES.json"])["files"]
        self.assertEqual(sha256(files["contract.py"]).hexdigest(),manifest["contract.py"]["SHA256"])
        for name,raw in files.items():
            if name.endswith(".py"):
                compile(raw,name,"exec")

    def test_whole_generated_packet_and_requests_reach_real_local_queue(self):
        self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{32}",REQUEST))
        root=Path(__file__).resolve().parent
        manifests=[json.loads((root/name).read_bytes()) for name in
                   ("ROOT_CLAIM.json","SCOPE.json","SOURCE_FILES.json")]
        identities=[doc["request"] for doc in manifests]
        self.assertEqual(identities+[launch.REQUEST,reader.REQUEST],[REQUEST]*5)
        self.assertEqual(LABEL,"O01-BF16-dependent-transition-stagger-fleet-v1")
        self.assertEqual([doc["label"] for doc in manifests[:2]],[LABEL,LABEL])
        self.assertEqual(RUN,LABEL+"-"+REQUEST)
        from controller import REQUEST as controller_request, RUN as controller_run
        self.assertEqual((controller_request,controller_run),(REQUEST,RUN))
        fleet=dispatch_fixture.BacklogTest()
        fleet.setUp()
        self.addCleanup(fleet.doCleanups)
        fleet.now=time.monotonic()
        fleet.anchor=time.time()-fleet.now
        for row in fleet.snapshot["nodes"].values():
            row.update(PCIs=CARDS,device_probe_step=row["parent"]+".900",kfd_process_count=0,
                       own_device_fd_process_count=0,probe_error=None)
        snapshot=fleet.verified_snapshot()
        plan,body,members=package.prepare("a"*40,1000.,snapshot,set())
        self.assertEqual(plan["request"],REQUEST)
        self.assertEqual((len(plan["selected"]),len(plan["shards"])),(74,31))
        self.assertEqual({cohort%3 for cohort in plan["cohort_by_shard"].values()},{0,1,2})
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{32}",r["request"]) for r in plan["shards"].values()))
        self.assertTrue(all(name.startswith("O01-dependent-transition-") for name in plan["shards"]))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{32}",r["request"]) for r in reader.schedule(plan).values()))
        self.assertLessEqual(len(body),131072)
        compile(body,"<original frozen transport>","exec")
        self.assertIn(package.TRANSITION_EVIDENCE.as_posix(),members)
        queue=launch.Queue(fleet.repo/"transition-fixture.sqlite")
        self.addCleanup(queue.db.close)
        with queue.db:
            handles=launch.reserve_wave(queue,plan,"a"*40,snapshot["local_receipt"],
                                        time.monotonic()+10,"f"*64,launch.DIRECTORY)
        self.assertEqual(len(handles),31)


if __name__=="__main__":
    unittest.main()
