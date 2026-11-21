import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import stage_gemma_render72_target_data as stage


class StageGemmaTargetTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.job = f'JobId={stage.PARENT} UserId=fixture(1000) JobState=RUNNING'
        self.rows = [f'{stage.PARENT}|fixture|RUNNING|0:0|10|pool|cpu=2560,node=40,gres/dcu=320']

    def original(self, state='COMPLETED'):
        return f'{stage.PARENT}.999|{stage.CPU_LABEL}|{state}|0:0|10|node|cpu=4,mem=8G'

    def query(self, command, *_):
        return self.job if command == 'scontrol' else '\n'.join(self.rows)

    def check(self):
        return stage.observe(self.root, self.query, 1000)

    def test_clear_requires_current_owned_parent_with_full_accounting(self):
        self.assertEqual(self.check()['state'], 'ORIGINAL_NAMESPACES_CLEAR')
        for job in (self.job.replace('(1000)', '(2000)'), self.job.replace('RUNNING', 'COMPLETED')):
            with patch.object(self, 'job', job), self.assertRaisesRegex(ValueError, 'owned parent'):
                self.check()
        self.rows[0] = self.rows[0].replace('320', '312')
        with self.assertRaisesRegex(ValueError, 'allocation not confirmed'):
            self.check()

    def test_in_flight_and_terminal_steps_both_require_reconciliation(self):
        for state in ('RUNNING', 'COMPLETED', 'TIMEOUT'):
            self.rows[1:] = [self.original(state)]
            report = self.check()
            self.assertEqual(report['cpu_accounting_row'][2], state)
            self.assertEqual(report['state'], 'RECONCILE_ORIGINAL')
        self.rows.append(self.original())
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.check()
        self.rows[1:] = [self.original().replace(f'{stage.PARENT}.999', '1234.999')]
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.check()

    def test_pristine_and_partial_namespaces_are_distinct(self):
        for name in (stage.CPU_LABEL, stage.STAGE_LABEL):
            with self.subTest(name=name):
                path = self.root / name
                path.mkdir()
                self.assertEqual(self.check()['state'], 'RECONCILE_ORIGINAL')
                path.rmdir()

    def test_transfer_hash_and_second_observation_precede_install(self):
        payload = b'fake payload'

        def never(*_):
            self.fail('must not install an unknown or misbound request')

        (self.root / stage.CPU_LABEL).mkdir()
        with self.assertRaisesRegex(ValueError, 'original request visible'):
            stage.receive(io.BytesIO(payload), self.root, 0, self.check, {}, never)
        (self.root / stage.CPU_LABEL).rmdir()
        with self.assertRaisesRegex(ValueError, 'private packet differs'):
            stage.receive(io.BytesIO(payload), self.root, 0, self.check, {}, never)

        class ConcurrentSubmit(io.BytesIO):
            def read(stream, count):
                self.rows.append(self.original('RUNNING'))
                return super().read(count)

        with patch.object(stage, 'PACKET_SHA', stage.sha256(payload).hexdigest()), \
                patch.object(stage, 'PACKET_BYTES', len(payload)), \
                self.assertRaisesRegex(ValueError, 'request changed during transfer'):
            stage.receive(ConcurrentSubmit(payload), self.root, 0, self.check, {}, never)
        self.assertFalse((self.root / stage.STAGE_LABEL).exists())

    def test_real_published_receiver_with_private_packet_when_available(self):
        inputs, install = stage.published_package()
        self.assertEqual(len(inputs['files']), 16)
        local = stage.REPO / 'local-only/training-gemma-render72-target-data-stage-v1-original/payload.private.bin'
        if not local.is_file():
            self.skipTest('private packet stays outside Git')

        def simulate(stream, path, bound_inputs, epoch):
            return install(stream, path, bound_inputs, epoch, local_test=True)

        with local.open('rb') as payload:
            result = stage.receive(payload, self.root, time.time()-0.1, self.check, inputs, simulate)
        self.assertEqual(result['completed_files'], 16)
        self.assertEqual(result['actual_bytes_written'], 1118352)
        self.assertFalse(result['READY'])
        audit = json.loads((self.root / stage.STAGE_LABEL / 'stage-admission.json').read_bytes())
        self.assertEqual(audit['original_cpu_request'], stage.CPU_REQUEST)
        self.assertEqual(audit['preceding_accounting_SHA256'], [self.check()['accounting_SHA256']]*2)

        # An actual local-test receipt cannot serve as target evidence.
        self.assertEqual(stage.inspect(self.root, self.check, inputs)['state'], 'RECONCILE_ORIGINAL')

    def test_inspect_releases_only_the_data_gate_for_a_complete_target_receipt(self):
        root = self.root / stage.STAGE_LABEL
        root.mkdir()
        (root/'sample').write_bytes(b'frozen')
        pin = dict(bytes=6, SHA256=stage.sha256(b'frozen').hexdigest())
        inputs = dict(files=[dict(target_relative='sample', **pin)])
        shared = dict(passed=True, state='GEMMA72_TRAIN_DATA_STAGED', request_id=stage.STAGE_REQUEST,
                      READY=False, planned_files=1, completed_files=1, actual_bytes_written=6, epoch=100.1)
        (root/'stage-outcome.json').write_text(json.dumps(dict(shared, actual_target_installation=True,
            local_test=False, installed={'sample': pin})), encoding='utf-8')
        admission = dict(shared, packet_SHA256=stage.PACKET_SHA, package_commit=stage.PACKAGE_COMMIT,
                         original_cpu_request=stage.CPU_REQUEST, preceding_accounting_SHA256=['a'*64, 'b'*64])
        (root/'stage-admission.json').write_text(json.dumps(admission), encoding='utf-8')
        self.assertEqual(stage.inspect(self.root, self.check, inputs)['state'], 'STAGING_VERIFIED_CPU_PREREQUISITES_PENDING')
        (root/'sample').write_bytes(b'drift!')
        self.assertEqual(stage.inspect(self.root, self.check, inputs)['state'], 'RECONCILE_ORIGINAL')


if __name__ == '__main__':
    unittest.main()
