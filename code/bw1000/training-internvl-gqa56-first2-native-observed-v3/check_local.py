"""Verify the immutable v3 renderer and gate its one-shot controller offline."""
import ast
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import unittest

import owner_gate as g
import render


PUB,CLAIM,MANIFEST="a"*40,"b"*64,"c"*64
NOW=g.record_epoch("2026-11-27T17:20:11")


def documents():
    release=dict(request=g.REQUEST,parent="1103606",node="f13r1n14",holder_step=g.HOLDER,accounting=[
        g.HOLDER,"training80-dcu-standby-9fd58db691fe-f13r1n14","COMPLETED","0:0","5","f13r1n14",
        "cpu=8,gres/dcu=8,mem=16G,node=1","2026-11-27T17:20:00","2026-11-27T17:20:05"],
        expected_file_SHA256=g.EXPECTED_SHA,worker_file_SHA256=g.WORKER_SHA,
        worker_returned=True,numbered_steps=[],queued_steps=[],observed_epoch=NOW-2)
    owner=dict(request=g.REQUEST,label=g.LABEL,publication=PUB,claim_SHA256=CLAIM,manifest_SHA256=MANIFEST,
        placement=g.PLACEMENT.copy(),zero_gres_step=g.CPU_ROW[0],zero_gres_result_SHA256=g.CPU_SHA,
        release_evidence_SHA256=sha256(render.encode(release)).hexdigest(),
        issued_epoch=NOW-1,expires_epoch=NOW+119,GPU_GO=True,replay_allowed=False)
    return release,owner


class Guards(unittest.TestCase):
    def test_exact_owner_and_release_chain(self):
        release,owner=documents()
        checks=[]
        here=Path("/stage")
        blobs={"owner-authority.json":render.encode(owner),"holder-release.json":render.encode(release)}
        def pin(path,*,expected,cap):
            checks.append((path,expected))
            if path==g.CPU_STAGED:
                self.assertEqual(expected,g.CPU_SHA)
                return b"fixture"
            raw=blobs[path.name]
            self.assertEqual(expected,sha256(raw).hexdigest())
            self.assertLessEqual(len(raw),cap)
            return raw
        wanted=sha256(blobs["owner-authority.json"]).hexdigest()
        self.assertEqual(g.load(here,wanted,PUB,CLAIM,MANIFEST,pin,NOW),owner)
        self.assertEqual([path for path,_ in checks],[here/"owner-authority.json",here/"holder-release.json",g.CPU_STAGED])
        with self.assertRaisesRegex(ValueError,"OWNER_PIN_REQUIRED"):
            g.load(here,None,PUB,CLAIM,MANIFEST,pin,NOW)

    def test_expired_wrong_port_prior_holder_and_cpu_fail_closed(self):
        release,owner=documents()
        with self.assertRaisesRegex(ValueError,"FRESH_OWNER_WINDOW"):
            g.validate_owner(owner,PUB,CLAIM,MANIFEST,owner["expires_epoch"])
        owner["placement"]["port"]=29967
        with self.assertRaisesRegex(ValueError,"EXPLICIT_OWNER_AUTHORITY"):
            g.validate_owner(owner,PUB,CLAIM,MANIFEST,NOW)
        release["accounting"][2]="RUNNING"
        with self.assertRaisesRegex(ValueError,"STANDBY_TERMINAL_ROW"):
            g.validate_release(release,NOW)

        def run(argv,**kwargs):
            self.assertEqual(kwargs["env"]["TZ"],"UTC")
            self.assertEqual(argv,["sacct","-nP","-j","1103606.4701","-o",g.COLS])
            return subprocess.CompletedProcess(argv,0,("|".join(g.CPU_ROW)+"\n").encode(),b"")
        self.assertEqual(g.verify_cpu_accounting(run),g.CPU_ROW)
        def gpu(argv,**kwargs):
            return subprocess.CompletedProcess(argv,0,("|".join(g.CPU_ROW).replace("cpu=2","cpu=2,gres/dcu=1")).encode(),b"")
        with self.assertRaisesRegex(ValueError,"ORIGINAL_4701_ZERO_GRES"):
            g.verify_cpu_accounting(gpu)

    def test_new_renderer_keeps_observed_slots_cp2_and_aligns_ports(self):
        files=render.render()
        self.assertEqual(files,render.render())
        scope=files["scope.py"].decode()
        control=files["control_once.py"].decode()
        native=files["native.py"].decode()
        node=files["node.py"].decode()
        self.assertIn("29987",scope)
        self.assertIn("29987",control)
        self.assertNotIn("29967",control)
        self.assertIn("intent_sha",control)
        self.assertIn("controller-owned.json",control)
        self.assertEqual(control.count("validate_launch(launch)"),2)
        self.assertIn("verify_accounting=True",control)
        self.assertIn("load_owner(HERE",node)
        self.assertIn('f"{scratch}:/tmp:rw"',node)
        self.assertIn('dir="/tmp"',node)
        self.assertIn('HF_HOME="/tmp/hf",HF_DATASETS_CACHE="/tmp/datasets"',native)
        self.assertIn("planned_update_slots=2,original_schedule_steps=25",native)
        self.assertIn("planned_input_slots=64",native)
        self.assertIn("seal_checkpoint",native)
        self.assertIn("live_cursor.confirmed",files["callback.py"].decode())
        ast.parse(node)
        claim=json.loads(files["CLAIM.json"])
        self.assertFalse(claim["GPU_GO"] or claim["root_GO"])
        self.assertEqual(claim["placement"],g.PLACEMENT)
        self.assertEqual(claim["prior_zero_gres_result_SHA256"],g.CPU_SHA)


if __name__=="__main__":
    unittest.main()
