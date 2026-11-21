import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import dispatch_glm_i128_compile_once as dispatch
from report_training80_backlog import read_original_holds, read_queue_holds
from tuning_queue import Queue


class RoutedCompileTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name).resolve()
        (self.repo/'local-only').mkdir()
        source = dispatch.REPO
        self.backlog = self.repo/dispatch.BACKLOG.relative_to(source)
        self.db = self.repo/'queue.sqlite3'
        Queue(self.db).db.close()
        self.publish = '1'*40
        files = {dispatch.BACKLOG.relative_to(source).as_posix(), dispatch.AUDIT['path']}
        for arm in dispatch.ARMS:
            branch = dispatch.scope(source,arm)
            files.add(branch['row']['evidence']['path'])
            relative = branch['source'].relative_to(source)
            known = json.loads((source/relative/'SOURCE_FILES.json').read_bytes())['files_SHA256']
            files.update((relative/name).as_posix() for name in (*known,'SOURCE_FILES.json'))
        for relative in files:
            dst = self.repo/relative
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/relative,dst)

    def local_source(self, repo, expected, publication):
        self.assertEqual((repo,publication),(self.repo,self.publish))
        source = expected['source']
        raw = (source/'SOURCE_FILES.json').read_bytes()
        names = json.loads(raw)['files_SHA256']
        return {name:(source/name).read_bytes() for name in (*names,'SOURCE_FILES.json')}, sha256(raw).hexdigest()

    def invoke(self, arm, transport, **kwargs):
        return dispatch.dispatch(arm,self.publish,transport,repo=self.repo,backlog=self.backlog,database=self.db,
                                 publication_inputs=kwargs.get('publication_inputs',self.local_source))

    def test_both_original_scopes_reserve_independently_and_keep_unknown_after_transport(self):
        for arm in dispatch.ARMS:
            expected = dispatch.scope(self.repo,arm)
            def transport(body,_,err,seconds):
                self.assertEqual(seconds,60)
                self.assertTrue(any(item['state']=='DISPATCHING' and item['direct_request']==expected['request']
                                    for item in read_queue_holds(self.db)))
                intent = json.loads((self.repo/expected['intent']).read_bytes())
                self.assertEqual(intent['body_SHA256'],sha256(body).hexdigest())
                payload = ast.literal_eval(ast.parse(body).body[0].value)
                self.assertEqual(payload['request'],expected['request'])
                err.write(b'fixture response lost')
                return SimpleNamespace(returncode=255)
            result = self.invoke(arm,transport)
            self.assertEqual(result['remote_outcome'],'UNKNOWN')
            self.assertIsNone(result['root_numbered_step'])
            with self.assertRaisesRegex(ValueError,'without replay'):
                self.invoke(arm,lambda *_: self.fail('cannot invoke original again'))
        observed = read_queue_holds(self.db)
        self.assertEqual({tuple(row['nodes']) for row in observed},{('f13r1n08',),('f13r1n09',)})
        self.assertTrue(all(row['state']=='UNKNOWN' for row in observed))

    def test_owned_process_response_does_not_make_numbered_step_or_terminal(self):
        expected = dispatch.scope(self.repo,'gate')
        def transport(_,out,err,seconds):
            intent = str(dispatch.PROJECT/'results'/(expected['label']+'-intent.json'))
            out.write(json.dumps(dict(request=expected['request'], publication=self.publish,
                manifest_SHA256=expected['manifest_SHA256'],GPU=False,epoch=1000.,deadline=1300.,
                host='fixture',pid=96,pgid=96,startticks=121,intent=intent)).encode())
            return SimpleNamespace(returncode=0)
        result = self.invoke('gate',transport)
        self.assertEqual(result['remote_outcome'],'OWNED_SRUN_STARTED')
        self.assertFalse(result['terminal_verified'])
        self.assertIsNone(result['root_numbered_step'])
        self.assertEqual(read_queue_holds(self.db)[0]['state'],'UNKNOWN')

    def test_competing_intent_and_legacy_namespace_block_without_calling_transport(self):
        expected = dispatch.scope(self.repo,'down')
        row = dict(id='fixture-prior', kind='validation', state='reconcile', placement={expected['node']:1})
        backlog = json.loads(self.backlog.read_bytes())
        self.backlog.write_text(json.dumps(dict(backlog,items=[*backlog['items'],row])),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'already holds'):
            self.invoke('down',lambda *_: self.fail('competing request'))
        self.assertFalse((self.repo/expected['intent']).exists())
        self.backlog.write_text(json.dumps(backlog),encoding='utf8')
        marker = self.repo/expected['intent']
        marker.parent.mkdir()
        marker.write_text('{}',encoding='utf8')
        groups = json.loads(dispatch.TRAINING80_LEDGER.read_bytes())['groups']
        nodes = {node for group in groups.values() for members in group['allocations'].values() for node in members}
        holds,_,_ = read_original_holds(self.repo,backlog['items'],nodes)
        self.assertEqual(holds[expected['row']['id']]['nodes'],[expected['node']])
        with self.assertRaisesRegex(ValueError,'without replay'):
            self.invoke('down',lambda *_: self.fail('legacy request'))
        self.assertEqual(read_queue_holds(self.db),[])

    def test_sidecar_bytes_must_match_before_any_reservation(self):
        def tampered(repo, expected, publication):
            files, mf = self.local_source(repo,expected,publication)
            files['receive.py'] += b'\n'
            return files, mf
        with self.assertRaisesRegex(ValueError,'source bytes'):
            self.invoke('gate',lambda *_: self.fail('bad source'),publication_inputs=tampered)
        self.assertEqual(read_queue_holds(self.db),[])


if __name__ == '__main__':
    unittest.main()
