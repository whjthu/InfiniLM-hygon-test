import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import close_graph64_original_backfill as v1
import close_graph64_v2_original_backfill as v2
import test_close_graph64_original_backfill as fixture


class Graph64V2Test(unittest.TestCase):
    def test_campaigns_cannot_cross_capture_boundaries(self):
        first = fixture.encode(dict(request=v1.REQUEST, replay_allowed=False))
        second = fixture.encode(dict(request=v2.REQUEST, replay_allowed=False))
        self.assertEqual(v1.terminal_receipt(first)["request"], v1.REQUEST)
        self.assertEqual(v2.terminal_receipt(second)["request"], v2.REQUEST)
        with self.assertRaises(ValueError):
            v1.terminal_receipt(second)
        with self.assertRaises(ValueError):
            v2.terminal_receipt(first)
        self.assertEqual(v1.PLAN_SHA, "7dafcd7567f1290b022c3a0ec07242eea5375860c3615c9035e0cc0a95afe08b")

    def test_source_envelope_and_claim_bind_actual_v2_files(self):
        expected = "1c7fc6f2337ebed380c58d01b92fb72ec24334633807a9642902419d82bd1ba4"
        self.assertEqual(v2.frozen_source(v2.REPO), expected)
        raw = (v2.REPO / v2.PACKAGE / "ROOT_CLAIM.json").read_bytes()
        self.assertEqual(json.loads(raw)["request"], v2.REQUEST)
        with patch.object(v2, "SOURCE_MANIFEST_SHA", "a"*64), self.assertRaises(ValueError):
            v2.frozen_source(v2.REPO)

    def test_mixed_v2_fixture_releases_only_the_independent_terminal_shards(self):
        with patch.object(fixture, "graph", SimpleNamespace(**v2._checker)):
            scope = fixture.OriginalGraph64Test()
            scope.setUp()
            self.addCleanup(scope.doCleanups)
            scope.test_mixed_outcomes_close_only_independently_proven_shards()
            for raw, in scope.queue.db.execute('SELECT result FROM runs WHERE state="TERMINAL"'):
                result = json.loads(raw)
                self.assertEqual(result["evidence"]["campaign_request"], v2.REQUEST)
                self.assertEqual(result["evidence"]["controller_SHA256"], v2.CONTROLLER_SHA)
                self.assertEqual(result["evidence"]["local_plan_SHA256"], v2.PLAN_SHA)


if __name__ == "__main__":
    unittest.main()
