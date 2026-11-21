from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import dispatch_internvl_usgs72_cpu_once as dispatch


class InternVLAuditQueueTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.board = self.root/'board.json'
        self.db = self.root/'flow.sqlite3'
        self.queue = dispatch.Queue(self.db)
        self.addCleanup(self.queue.db.close)
        SOURCE = dispatch.REPO/dispatch.PACKAGE
        target = self.root/dispatch.PACKAGE
        target.mkdir(parents=True)
        for entry in SOURCE.iterdir():
            if entry.is_file():
                (target/entry.name).write_bytes(entry.read_bytes())
        configured = json.loads(dispatch.BACKLOG.read_bytes())
        [row] = [job for job in configured['items'] if job['id'] == dispatch.IDENT]
        evidence = self.root/row['evidence']['path']
        evidence.parent.mkdir(parents=True)
        evidence.write_bytes((dispatch.REPO/row['evidence']['path']).read_bytes())
        self.board.write_text(json.dumps({'items':[row]}),encoding='utf8')
        self.claim = dispatch.scope(self.root,dispatch.PACKAGE_MANIFEST)
        epoch = time.time()
        self.context = dict(request_id=dispatch.REQUEST,publication='b'*40,epoch=epoch,deadline=epoch+300,
                            payload_SHA256='c'*64,source_manifest_SHA256=dispatch.PACKAGE_MANIFEST,
                            target_command=dispatch.TARGET_COMMAND,scope=dispatch.SCOPE)
        self.original = dict(self.context, GPU=0,MF_SHA256=dispatch.PACKAGE_MANIFEST,
                             target_command=list(dispatch.TARGET_COMMAND))
        path = self.root/dispatch.INTENT
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self.original),encoding='utf8')

    def register(self, context=None):
        return dispatch.register(self.queue,self.root,self.board,self.claim,
                                 self.context if context is None else context)

    def reader(self):
        post = dict(same_host=True,same_boot=True,held_child_reaped=True,controller_device_FDs=[],
                    owned_descendants_remaining=0,owned_device_FD_holders={})
        control = dict(held_child_reaped=True,owned_processes_remaining=0,cleanup_errors={},post=post)
        return dict(independent_metadata_consumed=True,request_id=dispatch.REQUEST,publication=self.original['publication'],
                    epoch=self.original['epoch'],deadline=self.original['deadline'],claim=self.claim,
                    source_manifest=dict(SHA256=dispatch.PACKAGE_MANIFEST),actual_CPU_steps=1,replay_allowed=False,
                    actual_accounting=['1087656.801',Path(dispatch.SCOPE).name,'COMPLETED','0:0','29',dispatch.NODE,
                                       'cpu=8,mem=12G,node=1'],controllers={k:deepcopy(control) for k in ('srun-control','container-control')},
                    planned_rows=72,completed_rows=72,native_call_counts={'failed_rows':0},qualification_passed=True)

    def test_reservation_precedes_authenticated_callback(self):
        events=[]

        def launcher(pub, manifest, *, transport_start):
            transport_start(self.context)
            return 255

        def start(context):
            [row] = dispatch.queue_holds(self.queue.db)
            events.append(row)
            self.assertEqual(row['state'], 'DISPATCHING')
            self.assertEqual(row['nodes'], [dispatch.NODE])
            self.assertEqual(context,self.context)

        with patch.object(dispatch.runpy,'run_path',return_value={'main':launcher}):
            result=dispatch.dispatch(self.context['publication'],dispatch.PACKAGE_MANIFEST,start,
                                     repo=self.root,backlog=self.board,database=self.db)
        self.assertEqual(result['launcher_rc'],255)
        self.assertFalse(result['terminal_verified'])
        self.assertEqual(result['queue_run'],events[0]['run'])
        with self.assertRaisesRegex(ValueError,'already claimed'):
            self.register()

    def test_mismatched_or_expired_context_cannot_claim(self):
        for context in (dict(self.context,payload_SHA256='d'*64),dict(self.context,scope='/wrong')):
            with self.assertRaisesRegex(ValueError,'original CPU request'):
                self.register(context)
        with self.assertRaisesRegex(ValueError,'epoch expired'):
            dispatch.register(self.queue,self.root,self.board,self.claim,self.context,wall=lambda:self.context['epoch']+291)
        self.assertEqual(dispatch.queue_holds(self.queue.db),[])

    def test_sqlite_conflict_blocks_original_request(self):
        self.queue.db.execute('INSERT INTO candidates VALUES(?,?)',('existing','{}'))
        self.queue.db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)',('existing','existing','component','UNKNOWN','{}',
                              json.dumps([dispatch.NODE]),None,None,0))
        with self.assertRaisesRegex(ValueError,'another request'):
            self.register()

    def test_successful_independent_post_releases_only_the_original_hold(self):
        self.register()
        reader=self.reader()
        for blocked in (dict(reader,actual_CPU_steps=None),dict(reader,controllers={})):
            with self.assertRaises((KeyError,ValueError)):
                dispatch.finish_payload(self.original,blocked,'f'*64,self.claim)
        path=self.root/'readout.json'
        raw=json.dumps(reader).encode()
        path.write_bytes(raw)
        outcome=dispatch.finish(path,sha256(raw).hexdigest(),repo=self.root,database=self.db)
        self.assertEqual((outcome['state'],outcome['passed']),('TERMINAL',True))
        self.assertEqual(dispatch.queue_holds(self.queue.db)[0]['nodes'],[])
        self.assertTrue((self.root/dispatch.INTENT).is_file())

    def test_clean_failed_step_keeps_real_failure_and_missing_denominator(self):
        result=self.reader()
        result['qualification_passed']=False
        result['actual_accounting'][2:4]=['FAILED','1:0']
        result.update(completed_rows=25,native_call_counts={'failed_rows':1})
        payload=dispatch.finish_payload(self.original,result,'f'*64,self.claim)
        self.assertEqual((payload['completed'],payload['failed'],payload['missing']),(25,1,46))
        self.assertFalse(payload['passed'])


if __name__=='__main__':
    unittest.main()
