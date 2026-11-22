"""Exercise actual scope boundaries and two-sided relay using local CPU fixtures."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/"full-model-screen32-v1"
sys.path[:0] = [str(HERE),str(ORIGINAL)]

import contract as c
import host
import node
import worker


SOURCE = "a"*64
CLAIM = dict(request=c.ID,label=c.LABEL,parent=c.PARENT,nodes=c.NODES,source_SHA256=SOURCE,
             caller_SHA256=c.PAIR_MANIFEST,ports=c.PORTS,DCU=16,GPU_CPU_per_node=64,GPU_memory_MiB_per_node=224000,
             keeper_CPU=2,keeper_memory_MiB=8192,construct_seconds=c.CONSTRUCT,load_seconds=c.LOAD,work_seconds=c.WORK,
             inside_seconds=c.INSIDE,GPU_seconds=c.GPU,keeper_seconds=c.KEEP,whole_seconds=c.WHOLE,max_GPU_steps=1,
             GPU_GO=True,Model_GO=False,keeper_node="f12r2n13",data_IPs=["10.1.2.30","10.1.2.31"],root_admission_SHA256="b"*64)


def row(role, state="COMPLETED"):
    ordinal = {"keeper":11001,"pre":11002,"device":11003,"post":11004}[role]
    return [f"{c.PARENT}.{ordinal}",f"{c.LABEL}-{role}",state,"0:0" if state=="COMPLETED" else "1:0","2",
            "f12r2n13" if role=="keeper" else "f13r1n[08-09]",
            {"device":"cpu=128,mem=437.50G,node=2,gres/dcu=16","keeper":"cpu=2,mem=8G,node=1"}
            .get(role,"cpu=2,mem=2048M,node=2")]


def receipt(role,state="COMPLETED"):
    return dict(row=row(role,state),state="TERMINAL",resources_bound=True,returncode=0 if state=="COMPLETED" else 1)


class Local(unittest.TestCase):
    def test_contract_and_admission_fixed(self):
        c.validate_claim(CLAIM,SOURCE)
        for patching in [{"nodes":["a02r3n15","a02r3n16"]},{"request":c.ID[:-1]},{"parent":"910371"},
                         {"ports":[43843,43844]},{"GPU_GO":False},{"data_IPs":["127.0.0.1","10.1.2.4"]}]:
            with self.subTest(patching=patching),self.assertRaises(ValueError):
                c.validate_claim(dict(CLAIM,**patching),SOURCE)
        for now in (0,1000+c.WHOLE):
            with self.assertRaises(ValueError):
                c.budget(1000,c.WHOLE,now=now)
        self.assertLess(c.LOAD,c.WORK)
        self.assertLess(c.WORK+15,c.INSIDE)
        self.assertLess(c.INSIDE+15,c.GPU)
        self.assertLess(c.GPU+100,c.KEEP)
        for key in ("ready_socket_parent.py","slurm_accounting.py","ENV.sh","checkpoint.py","INPUT_BINDING.json"):
            c.pinned(ORIGINAL/key,c.STAGED[key])

    def test_exact_commands_and_preexisting_duplicate_label(self):
        epoch=time.time()-.2
        argv=host.command("device",CLAIM,Path("orig"),epoch)
        self.assertIn("--nodelist=f13r1n08,f13r1n09",argv)
        self.assertIn("--gres=dcu:8",argv)
        self.assertIn("--cpus-per-task=64",argv)
        for role in ("pre","post","keeper"):
            self.assertIn("--gres=none",host.command(role,CLAIM,Path("orig"),epoch))
        other = ["910371.906","old","COMPLETED","0:0","1","a02r3n15","cpu=1"]
        with self.assertRaisesRegex(ValueError,"duplicate"):
            host.account("device",read=lambda *_:["|".join(row("device"))]*2)
        self.assertIsNone(host.account("device",read=lambda *_:["|".join(other)]))
        host.matching_resources(row("device"),"device",CLAIM,hosts=lambda _:set(c.NODES))
        bad=row("device")
        bad[6]="cpu=128,mem=448000M,node=2,gres/dcu=8"
        with self.assertRaisesRegex(ValueError,"DCU"):
            host.matching_resources(bad,"device",CLAIM,hosts=lambda _:set(c.NODES))

    def test_real_bidirectional_phase_gate(self):
        phase=c.PHASES[0]
        run={"request":c.ID,"source":SOURCE}
        def records(n):
            return [dict(run=run,phase=phase,node=c.NODES[n],rank=n*8+d,device=d,uid=42,PID=500+n*10+d,startticks=100)
                    for d in range(8)]
        frames=worker.Frames(time.time()-.2)
        with patch.object(worker.os,"getuid",lambda:42,create=True):
            left,right=socket.socketpair()
            with left,right,ThreadPoolExecutor(max_workers=2) as pool:
                one=pool.submit(worker.exchange,left,frames,0,phase,run,records(0))
                two=pool.submit(worker.exchange,right,frames,1,phase,run,records(1))
                self.assertEqual(one.result(timeout=2),two.result(timeout=2))
                epoch=time.time()-.2
                receive=pool.submit(worker.close_peer,right,1,epoch)
                worker.close_peer(left,0,epoch)
                receive.result(timeout=2)
            bad=records(0)
            bad[1]["run"]={"request":"unknown"}
            with self.assertRaisesRegex(ValueError,"wrong owner"):
                worker.verify_records(bad,0,phase,run)

    def test_exact_saved_process_post_and_stale_device(self):
        with tempfile.TemporaryDirectory() as dir:
            out=Path(dir)
            identity=dict(node=c.NODES[0],PID=2,startticks=7)
            step=row("device")[0]
            c.put(out/"owner-00.json", dict(identity=identity,run={"step":step}))
            with self.assertRaisesRegex(ValueError,"still live"):
                node.reconcile_processes(out,c.NODES[0],step,start=lambda _:7)
            self.assertEqual(node.reconcile_processes(out,c.NODES[0],step,start=lambda _:8),[(2,7)])
            with self.assertRaisesRegex(ValueError,"foreign"):
                node.reconcile_processes(out,c.NODES[0],"910371.906",start=lambda _:8)
        before=dict(node=c.NODES[0],address="10.1.2.30",boot="b",PCI=["00"],PCI_to_KFD={"00":1},ports=c.PORTS,
                    holders=0,kernel_holders=0,unreadable_FD=0,monotonic_ns=5)
        node.hardware_matches(before,dict(before,monotonic_ns=6),gate=True)
        for changed in [dict(before,holders=1),dict(before,unreadable_FD=1),dict(before,monotonic_ns=10**11)]:
            with self.assertRaises(ValueError):
                node.hardware_matches(before,changed,gate=True)

    def keeper(self, gpu):
        requested=[]
        before=dict(node=c.NODES[0],address="10.1.2.30",boot="b",PCI=["00"],PCI_to_KFD={"00":1},ports=c.PORTS,
                    holders=0,kernel_holders=0,unreadable_FD=0,monotonic_ns=5)
        def perform(_folder,role,_intent,_deadline):
            requested.append(role)
            return gpu if role=="device" else receipt(role)
        with tempfile.TemporaryDirectory() as dir:
            out=Path(dir)
            c.put(out/"intent.json",dict(claim=CLAIM,epoch=time.time()-.1))
            with patch.multiple(host,placement=lambda *_:None,step_identity=lambda:row("keeper")[0],
                                verify_stage=lambda *_:None,proc_start=lambda _:155,perform=perform,
                                evidence=lambda *_:{c.NODES[0]:before},consume_pair=lambda *_:{"prefill_pair_evidence_pass":True}), \
                 patch.object(host.socket,"gethostname",lambda:CLAIM["keeper_node"]), \
                 patch.object(host.os,"sched_getaffinity",lambda _:{1,2},create=True):
                if host.success(gpu):
                    host.keep(out)
                else:
                    with self.assertRaises(ValueError):
                        host.keep(out)
            return requested,c.read_json(out/"keeper-watch.json")

    def test_failure_and_unknown_lifecycles_have_distinct_cleanup(self):
        phases,report=self.keeper(receipt("device"))
        self.assertEqual(phases,["pre","device","post"])
        self.assertTrue(report["resource_clear"] and report["prefill_pair_evidence_pass"])
        phases,report=self.keeper(receipt("device","FAILED"))
        self.assertEqual(phases,["pre","device","post"])
        self.assertTrue(report["resource_clear"])
        self.assertFalse(report["prefill_pair_evidence_pass"])
        self.assertEqual(len(report["owner_denominator"]),16)
        self.assertTrue(all(row["state"]=="MISSING" for row in report["owner_denominator"]))
        phases,report=self.keeper(dict(state="UNKNOWN",row=None))
        self.assertEqual(phases,["pre","device"])
        self.assertFalse(report["resource_clear"])
        self.assertEqual(report["forwards_planned"],96)


if __name__=="__main__":
    unittest.main(verbosity=2)
