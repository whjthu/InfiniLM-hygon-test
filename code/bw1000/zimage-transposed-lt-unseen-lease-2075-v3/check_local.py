"""Verify the frozen device delta and successful/failed local admission fixtures."""

import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare


ROLLING = "training80-rolling-current-independent-two-point-42ef067b70bb4b828154f4af191793ed.json"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def functions(raw, names, scope):
    body = [node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), "<actual HOST renderer>", "exec"), scope)
    return scope


class Check(unittest.TestCase):
    def test_real_predecessor_records_and_exact_runtime(self):
        prior = prepare.prior_evidence()
        self.assertTrue(prior["four_attempts_never_sent_payload"])
        self.assertEqual(len(prior["transport_replies_SHA256"]), 4)
        version = prepare.frozen_renderer()
        rendered = version.source_files()
        claim = json.loads((prepare.HERE / "CLAIM.json").read_bytes())
        self.assertEqual(prepare.digest(rendered["worker.py"]), claim["fixed_source"]["device_worker_SHA256"])
        self.assertEqual(prepare.digest(rendered["SOURCE_FILES.json"]),
                         claim["fixed_source"]["device_stage_manifest_SHA256"])
        self.assertEqual(version.STANDBY, "1103606.2075")
        self.assertNotIn(prepare.V2_REQUEST.encode(), rendered["worker.py"])

    def test_static_old_snapshot_and_runtime_freshness(self):
        prior = prepare.prior_evidence()
        version = prepare.frozen_renderer()
        old = prepare.LOCAL / ROLLING
        with self.assertRaisesRegex(ValueError, "historical"):
            prepare.handoff(old, prepare.RENEWAL_SNAPSHOT_SHA)
        mock = json.loads(prepare.pinned(old, prepare.RENEWAL_SNAPSHOT_SHA))
        offset = 1000
        for phase in ("first", "second"):
            workers = mock[phase][prepare.LEASE_REQUEST]["workers"]
            for row in workers[prepare.NODE]["rows"].values():
                row["epoch"] += offset
        mock["epoch"] += offset
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "fixture.json"
            snapshot.write_bytes(version.encode(mock))
            authority = prepare.handoff(snapshot, prepare.digest(snapshot.read_bytes()))
            files = prepare.execution_files(version, authority, prior)
            version.stage(root / "fixture-envelope", files)
            claim = json.loads(files["DISPATCH.json"])
            self.assertEqual(claim["handoff_step"], prepare.LEASE)
            self.assertEqual(claim["files"]["entry.py"]["SHA256"], prepare.digest(files["entry.py"]))
            helper = functions(files["entry.py"], {"verify_fresh_handoff"}, {"require": need,
                               "time": SimpleNamespace(time=lambda: authority["observed_epoch"] + 20)})
            helper["verify_fresh_handoff"](claim)
            helper["time"].time = lambda: authority["observed_epoch"] + 301
            with self.assertRaisesRegex(ValueError, "expired"):
                helper["verify_fresh_handoff"](claim)

    def test_old_namespace_accounting_and_standby_terminal_gate(self):
        version = prepare.frozen_renderer()
        fresh = dict(step=prepare.LEASE, name=prepare.LEASE_LABEL, observed_epoch=prepare.RENEWAL_EPOCH + 100,
                     SHA256="a" * 64)
        generated = prepare.execution_files(version, fresh, prepare.prior_evidence())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            done = f"{prepare.OLD_STANDBY}|{prepare.OLD_STANDBY_LABEL}|COMPLETED|0:0|10547|{prepare.NODE}|cpu=8"
            old = f"1103606.99|{prepare.V2_LABEL}-gpu|COMPLETED|0:0|12|{prepare.NODE}|gres/dcu=1"
            scope = functions(generated["entry.py"], {"verify_retired_predecessor"},
                              {"RESULTS": root, "require": need, "query": lambda *_: done, "PARENT": prepare.PARENT,
                               "COLUMNS": ["1"] * 7, "NODE": prepare.NODE})
            verify = scope["verify_retired_predecessor"]
            verify()
            scope["query"] = lambda *_: done + "\n" + old
            with self.assertRaisesRegex(ValueError, "accounting requires reconciliation"):
                verify()
            scope["query"] = lambda *_: done.replace("COMPLETED", "RUNNING")
            with self.assertRaisesRegex(ValueError, "standby terminal"):
                verify()
            scope["query"] = lambda *_: done
            (root / (prepare.V2_LABEL + "-run")).mkdir()
            with self.assertRaisesRegex(ValueError, "namespace requires reconciliation"):
                verify()


if __name__ == "__main__":
    unittest.main()
