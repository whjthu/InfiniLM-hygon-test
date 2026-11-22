"""Local protocol fixtures for the newly scoped shard01 suffix relay."""

from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import endpoint
import relay


HERE = Path(__file__).resolve().parent


class Suffix(unittest.TestCase):
    def setUp(self):
        self.original_plan = json.loads((HERE/"PLAN.json").read_bytes())
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        prior = root / "prior"
        prior.mkdir()
        self.prefix, self.tail = b"pinned durable prefix", b"strict remaining suffix"
        self.plan = dict(self.original_plan, target=str(root/"new"), bytes=len(self.prefix+self.tail),
                         SHA256=sha256(self.prefix+self.tail).hexdigest(), prefix_bytes=len(self.prefix),
                         network_suffix_bytes=len(self.tail))
        binding = {key:self.plan[key] for key in ("file","bytes","SHA256")}
        binding.update(request="original", publication="b"*40)
        (prior/"shard.incomplete").write_bytes(self.prefix)
        self.prior = prior
        names = {
            "intent.json": binding,
            "terminal.json": dict(binding, role="receiver", passed=False, written_bytes=len(self.prefix),
                                 durable_bytes=len(self.prefix)),
            "retained-prefix.json": dict(binding, durable_bytes=len(self.prefix),
                                         prefix_SHA256=sha256(self.prefix).hexdigest()),
        }
        for name,data in names.items():
            (prior/name).write_bytes(json.dumps(data).encode())
        self.plan["prior"] = {
            "target":str(prior),"request":"original","publication":"b"*40,
            "intent_SHA256":sha256((prior/"intent.json").read_bytes()).hexdigest(),
            "terminal_SHA256":sha256((prior/"terminal.json").read_bytes()).hexdigest(),
            "prefix_receipt_SHA256":sha256((prior/"retained-prefix.json").read_bytes()).hexdigest(),
            "prefix_SHA256":sha256(self.prefix).hexdigest(),
        }
        self.binding = {key:self.plan[key] for key in ("request","file","bytes","SHA256")}
        self.binding.update(publication="a"*40,resume=dict(request="original",offset=len(self.prefix),
                                                           prefix_SHA256=sha256(self.prefix).hexdigest()))

    def receive(self, wire):
        output = io.StringIO()
        native_id = endpoint.identity

        def same_local_metadata(info):
            # Windows may defer path timestamps after this fixture's deliberate mutation.
            return {"size":native_id(info)["size"]}

        if os.name == "nt":
            with patch.object(endpoint,"identity",same_local_metadata), patch.object(
                    endpoint,"open_regular",lambda path:path.open("rb")), patch.object(endpoint,"sync_directory"), patch.object(
                    endpoint.sys,"stdin",SimpleNamespace(buffer=io.BytesIO(wire))), redirect_stdout(output):
                status=endpoint.receive(self.plan,self.binding)
        else:
            with patch.object(endpoint.sys,"stdin",SimpleNamespace(buffer=io.BytesIO(wire))), redirect_stdout(output):
                status=endpoint.receive(self.plan,self.binding)
        return status, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_source_commit_publishes_only_the_exact_new_file(self):
        wire=endpoint.frame(self.binding)+self.tail+endpoint.COMMIT
        status,rows=self.receive(wire)
        self.assertEqual(status,0)
        self.assertEqual(rows[0]["role"],"receiver_ready")
        self.assertEqual(rows[0]["durable_bytes"],len(self.prefix))
        self.assertTrue(rows[1]["passed"])
        self.assertEqual((Path(self.plan["target"])/self.plan["file"]).read_bytes(),self.prefix+self.tail)
        self.assertEqual((self.prior/"shard.incomplete").read_bytes(),self.prefix)

    def test_receiver_does_not_publish_without_source_commit(self):
        status,rows=self.receive(endpoint.frame(self.binding)+self.tail)
        self.assertEqual(status,1)
        self.assertFalse((Path(self.plan["target"])/self.plan["file"]).exists())
        self.assertEqual(rows[-1]["failure_code"],"MISSING_SOURCE_COMMIT_OR_TRAILER")

    def test_bad_retained_bytes_fail_before_receiver_ready(self):
        (self.prior/"shard.incomplete").write_bytes(b"x"*len(self.prefix))
        status,rows=self.receive(endpoint.frame(self.binding)+self.tail+endpoint.COMMIT)
        self.assertEqual(status,1)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["failure_code"],"TARGET_PREFIX_VALUES")

    def test_sender_only_forwards_suffix_after_frame(self):
        source=self.prior.parent/"source"
        source.write_bytes(self.prefix+self.tail)
        stream=io.BytesIO()
        normalize=(lambda info: {"size":info.st_size}) if os.name=="nt" else endpoint.identity
        opener=(lambda path:path.open("rb")) if os.name=="nt" else endpoint.open_regular
        plan=dict(self.plan,source_file=str(source),source_stat=normalize(source.lstat()))
        with patch.object(endpoint,"open_regular",opener), patch.object(endpoint,"identity",normalize), patch.object(
                endpoint.sys,"stdout",SimpleNamespace(buffer=stream)):
            endpoint.send(plan,self.binding)
        self.assertEqual(stream.getvalue(),endpoint.frame(self.binding)+self.tail)
        self.assertEqual(source.read_bytes(),self.prefix+self.tail)

    def test_current_published_evidence_and_range(self):
        plan=self.original_plan
        path=relay.REPO
        official=json.loads((path/plan["official_evidence"]["path"]).read_bytes())
        evidence=json.loads((path/plan["prior"]["independent_evidence"]["path"]).read_bytes())
        local=json.loads((path/plan["prior"]["local_terminal"]["path"]).read_bytes())
        prior=relay.validate_scope(plan,official,evidence,local)
        self.assertEqual(plan["bytes"]-prior["target_proof"]["offset"],552912848)
        wrong=deepcopy(evidence)
        wrong["independent_zero_gres_accounting"][2]="RUNNING"
        with self.assertRaisesRegex(ValueError,"TARGET_PREFIX_PROOF"):
            relay.validate_scope(plan,official,wrong,local)

        def fixture_transport(role, command):
            self.assertEqual(role,"source")
            args=shlex.split(command)
            compile(args[-1],"<fixture endpoint>","exec")
            return ["fixture-transport",command]

        binding=dict(self.binding, bytes=plan["bytes"], resume=prior["target_proof"])
        argv=relay.command(plan,binding,"sender",1800,fixture_transport)
        self.assertEqual(argv[0],"fixture-transport")


if __name__=="__main__":
    unittest.main()
