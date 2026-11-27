"""Run the actual singleton controller with captured scheduler and worker calls."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import controller
import bridge
import prepare


REQ,PUB="7"*32,"a"*40


class HandoffTest(unittest.TestCase):
    def invoke(self, mode=None):
        files=prepare.build(REQ)
        pins=json.loads(files["pci.json"])[controller.NODE]
        owner=json.loads(prepare.make_owner(REQ,PUB,time.time(),pins,
            queue_excluded_nodes=[controller.NODE],original_hold_conflicts=[]))
        events=[]
        uid=1234
        probe_request=controller.digest(f"{REQ}:same-lock-n06-pre".encode())[:32]
        probe_label="training80-admit-"+probe_request
        gpu_label=f"training80-dcu-standby-{REQ[:12]}-{controller.NODE}"
        probe=controller.PARENT+".9001"
        app=controller.PARENT+".9002"
        phase=0
        children=None
        gpu_argv=[]
        sys.path.insert(0,str(prepare.REPO/"tools"))
        import collect_training80_dispatch_snapshot as original_cpu

        @contextmanager
        def lock():
            events.append("lock")
            yield
            events.append("unlock")

        def proof_snapshot(groups,*,owner_uid,run_query):
            nonlocal phase
            phase+=1
            events.append(f"scheduler-{phase}")
            return dict(parents={"stub":dict(owned_by_observer=True,ledger_exact=True,
                                              step_locations_exact=True,state="RUNNING")})

        def cpu(argv,**kwargs):
            events.append("cpu")
            self.assertIn("--gres=none",argv)
            self.assertNotIn("--overlap",argv)
            self.assertEqual(json.loads(argv[-1]),[43843,43844])
            row=dict(request_id=probe_request,parent=controller.PARENT,node=controller.NODE,step=probe,PCIs=pins,
                     ports_checked=[43843,43844],observer_uid=uid,kfd_process_count=int(mode=="kfd"),
                     own_device_fd_process_count=0,occupied_requested_ports=[43843] if mode=="port" else [])
            return subprocess.CompletedProcess(argv,0,json.dumps(row).encode(),b"")

        def query(*argv):
            if argv[:3]==("scontrol","show","step"):
                events.append("visible")
                return [f"Name={gpu_label} StepId={app} NodeList={controller.NODE} State=RUNNING "
                        f"TRES=cpu=8,gres/dcu={7 if mode=='wrong-visible' else 8},mem=16G,node=1"]
            self.assertEqual(argv[0],"sacct")
            if argv[2]=="-S":
                if phase==2:
                    return [f"{probe}|{probe_label}"] + ([f"{app}|{gpu_label}"] if mode=="duplicate" else [])
                return []
            stamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            return ["|".join([probe,probe_label,"COMPLETED","0:0","0",controller.NODE,
                     "cpu=1,mem=256M,node=1" if mode!="bad-tres" else "cpu=1,gres/dcu=1,mem=256M,node=1",stamp,stamp])]

        def popen(argv,**kwargs):
            events.append("gpu-popen")
            self.assertIn("--threads=1",argv)
            self.assertIn("--gres=dcu:8",argv)
            self.assertIn("--nodelist="+controller.NODE,argv)
            self.assertTrue(kwargs["start_new_session"])
            gpu_argv.append(argv)
            return SimpleNamespace(pid=999999)

        before=lambda query,handle:{key:True for key in controller.PROOF_FIELDS if key!="raw_singleton_timeout"}
        match=lambda raw,handle:phase!=2 or mode!="other-step"
        with TemporaryDirectory() as temp:
            path=Path(temp).resolve()
            for name in ("historical-4625.json","standby_worker.py","collect_training80_dispatch_snapshot.py"):
                (path/name).write_bytes(files[name])
            original={"__name__":"pinned_standby"}
            with patch.dict(sys.modules,{"fcntl":SimpleNamespace()}):
                exec(compile(files["standby_controller.py"],"standby_controller.py","exec"),original)
            old_bridge={"__name__":"pinned_bridge"}
            exec(compile(files["original_bridge.py"],"original_bridge.py","exec"),old_bridge)
            children=old_bridge["StandbyChildren"]()
            source=dict(pci=pins,groups={},membership=SimpleNamespace(snapshot=proof_snapshot),collector=original_cpu,
                        allocation=SimpleNamespace(confirm_stale_step=before,stale_step_candidate=match),
                        standby=original,worker_SHA256=prepare.PUBLISHED["standby_worker.py"][1])
            with (patch.object(controller,"RESULTS",path),patch.object(controller,"sync_directory"),
                  patch.object(controller.os,"getuid",return_value=uid,create=True),patch.object(subprocess,"Popen",popen)):
                call=lambda:controller.execute(path,REQ,PUB,owner,source,children,query=query,run_cpu=cpu,lock_factory=lock)
                if mode:
                    with self.assertRaises(ValueError):
                        call()
                else:
                    self.assertEqual(call()["actual_GPU_step"],app)
                root=path/f"training80-dcu-standby-{REQ}"
                self.assertEqual((root/"intent.json").is_file(),mode in (None,"wrong-visible"))
        return events,gpu_argv

    def test_exact_once_train_partition_lock_and_native_wrapper(self):
        events,calls=self.invoke()
        self.assertEqual(events,["lock","scheduler-1","cpu","scheduler-2","gpu-popen","visible","unlock"])
        self.assertEqual(len(calls),1)

    def test_conflicts_keep_gpu_unsent_and_late_unknown_has_one_intent(self):
        for mode in ("other-step","kfd","port","bad-tres","duplicate"):
            with self.subTest(mode=mode):
                self.assertEqual(self.invoke(mode)[1],[])
        self.assertEqual(len(self.invoke("wrong-visible")[1]),1)

    def test_frozen_manifest_and_original_bridge_cpu_route(self):
        first=prepare.build(REQ)
        self.assertEqual(first,prepare.build(REQ))
        self.assertEqual(set(first)-{"SOURCE_FILES.json"},bridge.MEMBERS)
        command=bridge.control_argv(REQ,"c"*64,"d"*64)
        self.assertIn("--nodelist=f12r2n13",command)
        self.assertIn("--gres=none",command)
        self.assertEqual(command[-4:],["compute",REQ,"c"*64,"d"*64])


if __name__=="__main__":
    unittest.main()
