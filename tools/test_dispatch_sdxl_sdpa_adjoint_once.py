import ast
from contextlib import closing
from copy import deepcopy
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dispatch_sdxl_sdpa_adjoint_once as entry
from report_training80_backlog import plan, read_queue_holds
import test_report_training80_backlog as fixtures
from tuning_queue import Queue


class SDXLDispatchTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.BacklogTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.repo = self.fixture.repo.resolve()
        source_board = json.loads(entry.BACKLOG.read_bytes())
        [job] = [item for item in source_board['items'] if item['id'] == entry.IDENT]
        _, members = entry.scope(entry.REPO, job)
        parent = (entry.REPO/entry.PACKAGE).parent/'training-sdxl-frozen-conv-dgrad-fp32-v2'
        source_names = json.loads((parent/'SOURCE_FILES.json').read_bytes())['files']
        exact = members | {(parent/name).relative_to(entry.REPO).as_posix() for name in (*source_names, 'SOURCE_FILES.json')}
        for name in exact:
            path = self.repo/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((entry.REPO/name).read_bytes())
        self.board_path = self.repo/entry.BACKLOG.relative_to(entry.REPO)
        self.board = dict(minimum_ready_training=5, items=[job])
        self.save_board()
        self.database = self.repo/entry.QUEUE_DB.relative_to(entry.REPO)
        Queue(self.database).db.close()
        self.intent = self.repo/entry.INTENT
        self.now = self.fixture.now
        clock = patch.object(entry, 'time', SimpleNamespace(monotonic=lambda: self.now,
                             time=lambda: self.now+self.fixture.anchor))
        clock.start()
        self.addCleanup(clock.stop)
        self.fixture.snapshot['nodes'][entry.NODE].update(
            PCIs=[f'0000:{n:02x}:00.0' for n in range(1,9)], kfd_process_count=0,
            own_device_fd_process_count=0, device_probe_step=f'{entry.PARENT}.707', probe_error=None)
        self.snapshot = self.fixture.verified_snapshot()
        self.members = members | {self.board_path.relative_to(self.repo).as_posix()}

    def save_board(self):
        self.board_path.write_bytes(json.dumps(self.board).encode())

    def publication(self, repo, publication, members):
        self.assertEqual((repo, publication), (self.repo, 'b'*40))
        self.assertEqual(set(members), self.members)

    def run_dispatch(self, transport, *, check=None, snapshot=None):
        return entry.dispatch('b'*40, transport, snapshot=self.snapshot if snapshot is None else snapshot,
                              repo=self.repo, backlog=self.board_path, database=self.database,
                              check_publication=self.publication if check is None else check)

    def forbid(self, *_):
        self.fail('transport called for a refused request')

    def test_actual_bootstrap_is_bound_to_durable_unknown_before_callback(self):
        def transport(script, _out, _err, seconds):
            payload = ast.literal_eval(ast.parse(script).body[0].value)
            self.assertEqual(seconds, 120)
            self.assertEqual(payload['target'], entry.TARGET)
            self.assertEqual(payload['manifest_SHA256'], entry.MANIFEST)
            self.assertEqual(len(payload['members']), 18)
            [hold] = read_queue_holds(self.database)
            self.assertEqual((hold['state'], hold['nodes'], hold['direct_request']), ('UNKNOWN',[entry.NODE],entry.REQUEST))
            evidence = json.loads(self.intent.read_bytes())
            self.assertEqual(evidence['device_admission']['device_probe_step'], f'{entry.PARENT}.707')
            return SimpleNamespace(returncode=0)

        result = self.run_dispatch(transport)
        self.assertEqual((result['remote_outcome'],result['transport_returncode']), ('UNKNOWN',0))
        report = plan(self.snapshot,self.board,repo=self.repo,holds=read_queue_holds(self.database),
                      now=self.now,boot_anchor=self.fixture.anchor)
        self.assertEqual(report['rows'][0]['candidate_state'], 'reconcile')
        self.assertEqual(report['rows'][0]['request'], entry.REQUEST)

    def test_timeout_retains_node_and_refuses_replay(self):
        def transport(*_):
            raise subprocess.TimeoutExpired('fixture', 1)

        self.assertEqual(self.run_dispatch(transport)['transport_error'], 'TimeoutExpired')
        with self.assertRaisesRegex(ValueError,'already has a local scope'):
            self.run_dispatch(self.forbid)
        self.assertEqual(read_queue_holds(self.database)[0]['nodes'],[entry.NODE])

    def test_other_original_scope_prevents_reserving_node(self):
        self.board['items'].append(dict(id='already-reserved',state='reconcile',placement={entry.NODE:1}))
        self.save_board()
        with self.assertRaisesRegex(ValueError,'another original request'):
            self.run_dispatch(self.forbid)
        self.assertFalse(self.intent.exists())

    def test_bad_receipt_and_board_drift_refuse_before_intent(self):
        snap = deepcopy(self.snapshot)
        snap['nodes'][entry.NODE]['device_probe_step'] = '1103606.708'
        with self.assertRaisesRegex(ValueError,'collector identity or payload differs'):
            self.run_dispatch(self.forbid,snapshot=snap)

        def check(*args):
            self.publication(*args)
            self.board['items'][0]['next_action'] = 'superseded'
            self.save_board()

        with self.assertRaisesRegex(ValueError,'backlog changed'):
            self.run_dispatch(self.forbid,check=check)
        self.assertFalse(self.intent.exists())

    def test_expiry_before_arming_leaves_durable_non_submission(self):
        admission = entry.device_admission
        count = 0

        def expire(*args):
            nonlocal count
            count += 1
            if count == 3:
                self.now += 100
            return admission(*args)

        with patch.object(entry, 'device_admission', side_effect=expire):
            result = self.run_dispatch(self.forbid)
        self.assertEqual(result['remote_outcome'], 'NOT_ATTEMPTED')
        [hold] = read_queue_holds(self.database)
        self.assertEqual((hold['state'],hold['nodes']), ('RETIRED_PRELAUNCH',[]))

    def test_expiry_after_arming_retains_unknown_reservation(self):
        admission = entry.device_admission
        count = 0

        def expire(*args):
            nonlocal count
            count += 1
            if count == 4:
                self.now += 100
            return admission(*args)

        with patch.object(entry, 'device_admission', side_effect=expire):
            result = self.run_dispatch(self.forbid)
        self.assertEqual(result['remote_outcome'], 'UNKNOWN')
        self.assertFalse(result['transport_attempted'])
        [hold] = read_queue_holds(self.database)
        self.assertEqual((hold['state'],hold['nodes']), ('UNKNOWN',[entry.NODE]))

    def test_original_terminal_cleanup_releases_node_while_preserving_failure(self):
        result = self.run_dispatch(lambda *_: SimpleNamespace(returncode=255))
        done = dict(request_id=entry.REQUEST,intent=entry.INTENT,parent=entry.PARENT,terminal=True,
                    post_clean=True,evidence_SHA256='d'*64,planned=1,completed=0,failed=1,rejected=0,
                    canceled=0,missing=0,qualification='validation',keeper_passed=None,
                    placement_by_parent={entry.PARENT:{entry.NODE:1}})
        queue = Queue(self.database)
        with closing(queue.db):
            with self.assertRaisesRegex(ValueError,'requires actual terminal and post'):
                queue.finish(result['queue_run'],dict(done,post_clean=False))
            queue.finish(result['queue_run'],done)
        holds = read_queue_holds(self.database)
        self.assertEqual(holds[0]['state'], 'TERMINAL')
        report = plan(self.snapshot,self.board,repo=self.repo,holds=holds,now=self.now,
                      boot_anchor=self.fixture.anchor)
        self.assertNotIn(entry.IDENT,report['original_launch_holds'])
        self.assertEqual(report['rows'][0]['candidate_state'],'terminal')
        self.assertEqual(report['outstanding_training_ready'],0)


if __name__ == '__main__':
    unittest.main()
