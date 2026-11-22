"""Exercise actual scope boundaries and two-sided relay using local CPU fixtures."""

from concurrent.futures import ThreadPoolExecutor
import builtins
from contextlib import redirect_stdout
import errno
from io import StringIO
import json
from pathlib import Path
import socket
import symtable
import sys
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent/"full-model-screen32-v1"
PAIR = HERE.parent/"fresh-kv-prefill512-resident-three-pairs-source-v1"
sys.path[:0] = [str(HERE),str(ORIGINAL),str(PAIR)]

from caller import binding as bindings, PHASES as CALLER_PHASES
import contract as c
import host
import node
import worker


SOURCE = "a"*64
CARDS = {name:[f"0000:{i+16+8*index:02x}:00.0" for i in range(8)] for index,name in enumerate(c.NODES)}
CLAIM = dict(request=c.ID,label=c.LABEL,parent=c.PARENT,nodes=c.NODES,source_SHA256=SOURCE,
             caller_SHA256=c.PAIR_MANIFEST,ports=c.PORTS,DCU=16,GPU_CPU_per_node=64,GPU_memory_MiB_per_node=224000,
             keeper_CPU=2,keeper_memory_MiB=8192,construct_seconds=c.CONSTRUCT,load_seconds=c.LOAD,work_seconds=c.WORK,
             inside_seconds=c.INSIDE,GPU_seconds=c.GPU,keeper_seconds=c.KEEP,whole_seconds=c.WHOLE,max_GPU_steps=1,
             GPU_GO=True,Model_GO=False,keeper_node="f12r2n13",data_IPs=["10.1.2.30","10.1.2.31"],
             PCI_by_node=CARDS,root_admission_SHA256="b"*64)


def row(role, state="COMPLETED"):
    ordinal = {"keeper":11001,"pre":11002,"device":11003,"post":11004}[role]
    return [f"{c.PARENT}.{ordinal}",f"{c.LABEL}-{role}",state,"0:0" if state=="COMPLETED" else "1:0","2",
            "f12r2n13" if role=="keeper" else "f13r1n[08-09]",
            {"device":"cpu=128,mem=437.50G,node=2,gres/dcu=16","keeper":"cpu=2,mem=8G,node=1"}
            .get(role,"cpu=2,mem=2048M,node=2")]


def receipt(role,state="COMPLETED"):
    return dict(row=row(role,state),state="TERMINAL",resources_bound=True,returncode=0 if state=="COMPLETED" else 1)


def snapshot(name,phase):
    index = c.NODES.index(name)
    pci = CARDS[name]
    status,topology = node.read_topology(phase,pci,lambda:dict(zip(pci,range(1,9))))
    return dict(phase=phase,node=name,actual_step=row(phase)[0],address=CLAIM["data_IPs"][index],
                boot="fixture-boot",PCI=pci,PCI_to_KFD=topology,KFD_topology_status=status,ports=c.PORTS,
                holders=0,kernel_holders=0,unreadable_FD=0,monotonic_ns={"pre":5,"device":6,"post":7}[phase])


def mounted_data():
    return dict(files=bindings.mounted_input_pins(),layers=78,vocab=bindings.VOCAB,index_keys=bindings.INDEX_KEYS,
                shards=bindings.SHARDS,prompt_tokens=1312,expected_provider_paths=71,native_loaded=False,
                checkpoint_weights_read=False,SDK_calls=0,DCU=0)


class Local(unittest.TestCase):
    def test_contract_and_admission_fixed(self):
        self.assertEqual(c.PHASES,CALLER_PHASES)
        self.assertEqual(c.FORWARDS_PER_OWNER,18)
        c.validate_claim(CLAIM,SOURCE)
        for patching in [{"nodes":["a02r3n15","a02r3n16"]},{"request":c.ID[:-1]},{"parent":"910371"},
                         {"ports":[43843,43844]},{"GPU_GO":False},{"data_IPs":["127.0.0.1","10.1.2.4"]},
                         {"PCI_by_node":{}},{"request":"2bf7ebc97ef74cb9a8d185f10e3d762a"},
                         {"request":"a9bec1b423424abc8240db63df2473fa"},
                         {"request":"41e1f4ae471d46f39d9130145ac4ea35"},
                         {"request":"e1549a5272f54350a567a927c9f10f39"},
                         {"request":"e1549a5272f54350a567a927c9f10f397"}]:
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
        proof = HERE.parents[3]/"results/bw1000/2026-11-22/GLM474-prefill512-resident-v1-zero-GRES-topology-admission.json"
        record = json.loads(c.pinned(proof,c.V1_DIAGNOSTIC_SHA))
        self.assertEqual([(r[0],r[2],r[3]) for r in record["rows"]],
                         [("1103606.1087","FAILED","1:0"),("1103606.1088","COMPLETED","0:0")])
        self.assertFalse(record["GPU_submission_attempted"] or record["keeper_intent"])
        self.assertEqual({v["KFD_topology"]["errno"] for v in record["receipts"]["diagnostic-admission.json"]["nodes"].values()},{1})

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

    def test_production_global_names_are_bound(self):
        def scan(table):
            return {item.get_name() for item in table.get_symbols() if item.is_global() and item.is_referenced()} | {
                name for scope in table.get_children() for name in scan(scope)
            }
        ambient = set(vars(builtins)) | {"__file__","__name__","__package__","__spec__","__loader__","__cached__"}
        for name in ("contract.py","host.py","node.py","worker.py"):
            table = symtable.symtable((HERE/name).read_text(encoding="utf8"),name,"exec")
            bound = {item.get_name() for item in table.get_symbols() if item.is_assigned() or item.is_imported()}
            self.assertEqual(scan(table)-bound-ambient,set(),name)
        self.assertEqual(host.HOST,c.HOST)

    def test_actual_launch_and_status_cli_with_fake_slurm(self):
        visible = []
        commands = []
        node_tokens = {"f13r1n[08-09]":c.NODES,"f12r2n13":[CLAIM["keeper_node"]],
                       "f12r2n13,f13r1n[08-09]":[CLAIM["keeper_node"],*c.NODES]}

        def query(argv,seconds=10):
            commands.append(argv)
            if argv[0] == "squeue":
                return ["RUNNING|f12r2n13,f13r1n[08-09]"]
            if argv[:3] == ["scontrol","show","step"]:
                return []
            if argv[:3] == ["scontrol","show","hostnames"]:
                return node_tokens[argv[3]]
            if argv[0] == "sacct":
                return ["|".join(row(role)) for role in visible]
            self.fail(f"unknown external action {argv}")

        def check_output(argv,timeout,text):
            self.assertTrue(text)
            return "\n".join(query(argv,timeout))

        with tempfile.TemporaryDirectory() as dir:
            root = Path(dir)
            (root/"results").mkdir()
            admission = root/"admission.json"
            claim_path = root/"claim.json"
            c.put(admission,{"actual_fake_placement":True})
            c.put(claim_path,dict(CLAIM,root_admission_SHA256=c.sha(admission.read_bytes())))
            claim_pin = c.sha(claim_path.read_bytes())
            launch = ["host.py","launch","--claim",str(claim_path),"--claim-sha256",claim_pin,
                      "--source-sha256",SOURCE,"--admission",str(admission)]
            with patch.multiple(host,HOST=root,verify_stage=Mock(),proc_start=lambda _:777), \
                 patch.object(host.subprocess,"check_output",check_output), \
                 patch.object(host.subprocess,"Popen",return_value=Mock(pid=24680)) as popen:
                stdout = StringIO()
                with patch.object(sys,"argv",launch),redirect_stdout(stdout):
                    host.main()
                self.assertEqual(json.loads(stdout.getvalue())["pid"],24680)
                out = root/"results"/c.LABEL
                intent = c.read_json(out/"intent.json")
                self.assertEqual(intent["claim"]["request"],c.ID)
                self.assertEqual(intent["claim_SHA256"],claim_pin)
                submitted = c.read_json(out/"keeper-submission.json")
                self.assertEqual((submitted["pid"],submitted["startticks"]),(24680,777))
                self.assertEqual(submitted["intent_SHA256"],c.sha((out/"intent.json").read_bytes()))
                argv = popen.call_args.args[0]
                self.assertIn("--job-name="+c.LABEL+"-keeper",argv)
                self.assertIn("--gres=none",argv)
                self.assertEqual(argv[-2:],["keep",str(out)])
                with patch.object(sys,"argv",launch),self.assertRaises(FileExistsError):
                    host.main()
                self.assertEqual(popen.call_count,1)
                stdout = StringIO()
                with patch.object(sys,"argv",["host.py","status"]),redirect_stdout(stdout):
                    host.main()
                self.assertEqual(json.loads(stdout.getvalue())["state"],"PENDING_OR_UNKNOWN")
                visible.extend(("keeper","device","post"))
                c.put(out/"keeper-watch.json",dict(step=row("keeper")[0],resource_clear=True,
                      prefill_pair_evidence_pass=True,device=receipt("device"),post=receipt("post")))
                stdout = StringIO()
                with patch.object(sys,"argv",["host.py","status"]),redirect_stdout(stdout):
                    host.main()
                result = json.loads(stdout.getvalue())
                self.assertEqual((result["state"],result["keeper"][0]),("TERMINAL",row("keeper")[0]))
                self.assertTrue(result["release"] and result["prefill_pair_evidence_pass"])
            self.assertTrue(any(argv[0]=="sacct" for argv in commands))

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
        name = c.NODES[0]
        before, leased = snapshot(name,"pre"), snapshot(name,"device")
        node.hardware_matches(before,leased,gate=True)
        for changed in [dict(leased,holders=1),dict(leased,unreadable_FD=1),dict(leased,monotonic_ns=10**11)]:
            with self.assertRaises(ValueError):
                node.hardware_matches(before,changed,gate=True)

    def test_zero_gres_scope_keeps_permission_denials_separate(self):
        pci = CARDS[c.NODES[0]]
        read = Mock(side_effect=PermissionError(errno.EPERM,"observed zero-GRES KFD restriction"))
        for phase in ("pre","post"):
            self.assertEqual(node.read_topology(phase,pci,read),("UNAVAILABLE_ZERO_GRES",None))
        read.assert_not_called()
        with self.assertRaises(PermissionError) as raised:
            node.read_topology("device",pci,read)
        self.assertEqual(raised.exception.errno,errno.EPERM)
        read.assert_called_once()

    def test_zero_gres_mounted_entry_and_exact_two_node_precondition(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            pre = receipt("pre")
            step = pre["row"][0]
            with patch.object(bindings,"verify_mounted_inputs",return_value=mounted_data()) as mounted, \
                 patch.multiple(worker,verify_stage=Mock(),step_identity=lambda:step), \
                 patch.dict(worker.os.environ,{"SLURM_GPUS_ON_NODE":"0","SLURM_STEP_GPUS":"N/A"},clear=True):
                for hostname in c.NODES:
                    with patch.object(worker.socket,"gethostname",lambda:hostname):
                        worker.main("mounted-preflight",str(out),SOURCE)
                self.assertEqual(mounted.call_count,2)
                with patch.object(worker.socket,"gethostname",lambda:c.NODES[0]), \
                     patch.dict(worker.os.environ,{"SLURM_STEP_GPUS":"0"}):
                    with self.assertRaisesRegex(ValueError,"zero-GRES"):
                        worker.main("mounted-preflight",str(out),SOURCE)
                self.assertEqual(mounted.call_count,2)
            binding = {"claim":CLAIM}
            checked = host.inspect_mounted_preflight(out,pre,binding)
            self.assertEqual({name:info["files"] for name,info in checked.items()},{name:9 for name in c.NODES})
            path = out/f"mounted-pre-{c.NODES[1]}.json"
            corrupt = c.read_json(path)
            corrupt["files"]["config"]["bytes"] = 16384
            c.put(path,corrupt,replace=True)
            with self.assertRaisesRegex(ValueError,"both exact"):
                host.inspect_mounted_preflight(out,pre,binding)

    def test_exact_full_leased_gate_and_negative_launch(self):
        name, peer = c.NODES
        step = row("device")[0]
        for mapping in [{}, dict.fromkeys(CARDS[name],1),dict(zip(CARDS[peer],range(1,9)))]:
            with self.assertRaisesRegex(ValueError,"KFD/DRM"):
                node.read_topology("device",CARDS[name],lambda:mapping)
        with tempfile.TemporaryDirectory() as dir:
            out = Path(dir)
            epoch = time.time()-.2
            c.put(out/"intent.json",dict(claim=CLAIM,epoch=epoch))
            with patch.multiple(node,step_identity=lambda:step,
                                snapshot_mode=Mock(side_effect=PermissionError(errno.EPERM,"leased access denied"))), \
                 patch.object(node.socket,"gethostname",lambda:peer),patch.object(node.os,"execv") as enter:
                with self.assertRaises(PermissionError):
                    node.main("device",out)
                enter.assert_not_called()
            neg = c.read_json(out/f"device-entry-{peer}.json")
            self.assertEqual((neg["state"],neg["errno"]),("REJECTED",1))
            with self.assertRaisesRegex(ValueError,"node rejected"):
                node.wait_both_leases(out,step,epoch)
            for hostname in c.NODES:
                c.put(out/f"pre-{hostname}.json",snapshot(hostname,"pre"))
                c.put(out/f"device-{hostname}.json",snapshot(hostname,"device"))
                c.put(out/f"device-entry-{hostname}.json",dict(node=hostname,actual_step=step,state="VERIFIED"),
                      replace=(hostname==peer))
            self.assertEqual(set(node.wait_both_leases(out,step,epoch)),set(c.NODES))
            self.assertEqual(set(worker.verify_lease_pins(out,{"claim":CLAIM},{"step":step})),set(c.NODES))
            bad_claim = dict(CLAIM,PCI_by_node={hostname:CARDS[name] for hostname in c.NODES})
            with self.assertRaisesRegex(ValueError,"per-card"):
                worker.verify_lease_pins(out,{"claim":bad_claim},{"step":step})

    def test_leased_reader_waits_for_complete_atomic_marker(self):
        first,second = c.NODES
        step = row("device")[0]
        epoch = time.time()-.2
        writing, reader_waiting, finish = Event(),Event(),Event()
        opened = []
        actual_fdopen, actual_link, actual_read = c.os.fdopen,c.os.link,node.read_json

        class DelayedWrite:
            def __init__(self,stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self,*exc):
                return self.stream.__exit__(*exc)

            def __getattr__(self,name):
                return getattr(self.stream,name)

            def write(self,raw):
                writing.set()
                if not finish.wait(5):
                    raise TimeoutError("fixture delayed writer was never released")
                return self.stream.write(raw)

        with tempfile.TemporaryDirectory() as dir:
            out=Path(dir)
            marker=out/f"device-entry-{second}.json"
            for name in c.NODES:
                c.put(out/f"pre-{name}.json",snapshot(name,"pre"))
                c.put(out/f"device-{name}.json",snapshot(name,"device"))
            c.put(out/f"device-entry-{first}.json",dict(node=first,actual_step=step,state="VERIFIED"))

            def fdopen(*args):
                stream=actual_fdopen(*args)
                opened.append(stream)
                return DelayedWrite(stream)

            def observed_read(path, *args):
                try:
                    return actual_read(path,*args)
                except FileNotFoundError:
                    if path == marker:
                        reader_waiting.set()
                    raise

            def publish(src,dst):
                self.assertTrue(opened[0].closed)
                return actual_link(src,dst)

            with patch.object(c.os,"fdopen",fdopen),patch.object(c.os,"link",publish), \
                 patch.object(node,"read_json",observed_read), \
                 ThreadPoolExecutor(max_workers=2) as pool:
                writer = pool.submit(c.put,marker,dict(node=second,actual_step=step,state="VERIFIED"))
                try:
                    self.assertTrue(writing.wait(2))
                    pending = pool.submit(node.wait_both_leases,out,step,epoch)
                    self.assertTrue(reader_waiting.wait(2))
                    self.assertFalse(pending.done())
                    self.assertEqual(len(list(out.glob("."+marker.name+".tmp-*"))),1)
                    with self.assertRaises(FileNotFoundError):
                        c.read_json(marker)
                finally:
                    finish.set()
                writer.result(timeout=2)
                leases = pending.result(timeout=2)
                self.assertTrue(opened[0].closed)
                self.assertEqual(set(leases),set(c.NODES))
                self.assertFalse(list(out.glob(".*.tmp-*")))

    def test_publish_refuses_duplicate_and_preserves_strict_read(self):
        with tempfile.TemporaryDirectory() as dir:
            out=Path(dir)
            path=out/"record.json"
            original = {"step":row("device")[0],"writing":False}
            c.put(path,original)
            original_bytes = path.read_bytes()
            with self.assertRaises(FileExistsError):
                c.put(path,{"step":"wrong"})
            self.assertEqual(path.read_bytes(),original_bytes)
            self.assertFalse(list(out.glob(".*.tmp-*")))
            c.put(path,dict(original,writing=True),replace=True)
            self.assertTrue(c.read_json(path)["writing"])
            self.assertFalse(list(out.glob(".*.tmp-*")))
            path.write_bytes(b"")
            with self.assertRaisesRegex(ValueError,"bounded JSON size"):
                c.read_json(path)
            path.write_bytes(b"{bad json")
            with self.assertRaises(json.JSONDecodeError):
                c.read_json(path)

    def test_post_binds_positive_leased_receipt_or_clean_negative(self):
        name = c.NODES[0]
        pre,leased,post = (snapshot(name,phase) for phase in ("pre","device","post"))
        step = leased["actual_step"]
        self.assertEqual(node.verify_post_binding(pre,post,leased,step),step)
        self.assertIsNone(node.verify_post_binding(pre,post,None,step))
        for invalid in [dict(leased,actual_step="1103606.1087"),dict(leased,boot="other"),
                        dict(leased,KFD_topology_status="UNAVAILABLE_ZERO_GRES",PCI_to_KFD=None)]:
            with self.assertRaises(ValueError):
                node.verify_post_binding(pre,post,invalid,step)

    def keeper(self, gpu, *, mounted_ok=True):
        requested=[]
        def perform(_folder,role,_intent,_deadline):
            requested.append(role)
            if role == "pre" and mounted_ok:
                for name in c.NODES:
                    c.put(_folder/f"mounted-pre-{name}.json",dict(mounted_data(),node=name,
                          state="MOUNTED_INPUTS_VERIFIED_ZERO_GRES",actual_step=row("pre")[0],source_SHA256=SOURCE))
            return gpu if role=="device" else receipt(role)
        def evidence(_folder,phase,_receipt):
            return {name:dict(snapshot(name,phase),prior_leased_KFD_step=None,prior_leased_KFD_SHA256=None)
                    for name in c.NODES}
        with tempfile.TemporaryDirectory() as dir:
            out=Path(dir)
            c.put(out/"intent.json",dict(claim=CLAIM,epoch=time.time()-.1))
            with patch.multiple(host,placement=lambda *_:None,step_identity=lambda:row("keeper")[0],
                                verify_stage=lambda *_:None,proc_start=lambda _:155,perform=perform,
                                evidence=evidence,consume_pair=lambda *_:{"prefill_numerical_evidence_pass":True}), \
                 patch.object(host.socket,"gethostname",lambda:CLAIM["keeper_node"]), \
                 patch.object(host.os,"sched_getaffinity",lambda _:{1,2},create=True):
                if host.success(gpu) and mounted_ok:
                    host.keep(out)
                else:
                    with self.assertRaises(ValueError if mounted_ok else FileNotFoundError):
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
        self.assertEqual(report["forwards_planned"],288)
        phases,report=self.keeper(receipt("device"),mounted_ok=False)
        self.assertEqual(phases,["pre"])
        self.assertEqual(report["GPU_steps"],0)


if __name__=="__main__":
    unittest.main(verbosity=2)
