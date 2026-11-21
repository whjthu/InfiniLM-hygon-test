from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import read_gemma_render72_native_cpu_original as reader
from control_gemma_render72_native_cpu import REPO


class GemmaCPUReaderTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base/'run'
        self.data = self.base/'data'
        self.publication = 'a'*40
        self.epoch = 1795220000.0
        self.account = [f'{reader.PARENT}.789', reader.LABEL, 'COMPLETED', '0:0', '63', reader.NODE,
                        'cpu=4,mem=8G,node=1']
        self.train = b''.join(f'{{"fixture":{i}}}\n'.encode() for i in range(72))
        self.write(self.data/reader.TRAIN_FILE, self.train)
        self.capture = [dict(ordinal=i, source_line_SHA256=sha256(line).hexdigest(), tokens=200+i,
                             shifted_supervised_tokens=9, encoded={'SHA256': '1'*64}, collated={'SHA256': '2'*64})
                        for i, line in enumerate(self.train.splitlines(keepends=True), 1)]
        for name in reader.PINS:
            if name not in (reader.MEMBERSHIP, reader.TERM_HELPER, reader.ROSTER):
                self.write(self.root/'src'/Path(name).name, (REPO/name).read_bytes())
        self.intent = dict(request=reader.REQUEST, publication=self.publication, epoch=self.epoch, parent=reader.PARENT,
                           node=reader.NODE, whole_deadline=self.epoch+480, worker_deadline=self.epoch+390,
                           replay_allowed=False, source_SHA256=reader.PINS, **reader.PLANNED)
        self.write(self.root/'intent.json', self.intent)
        self.save_worker()

    @staticmethod
    def write(path, data):
        raw = data if type(data) is bytes else (json.dumps(data)+'\n').encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return raw

    def save_worker(self, **extra):
        raw = b''.join((json.dumps(item)+'\n').encode() for item in self.capture)
        self.write(self.root/'audit/native-records-private.jsonl', raw)
        worker = dict(passed=True, model_loaded=False, actual_rank_loader_observed=False, training_ready=False,
                      epoch=self.epoch, deadline=self.epoch+390, private_capture_SHA256=sha256(raw).hexdigest(),
                      **reader.PLANNED)
        worker.update(extra)
        raw = self.write(self.root/'audit/result.json', worker)
        self.outcome = dict(request=reader.REQUEST, passed=True, srun_return_code=0, failure_type=None,
                            target_training_admitted=False, optimizer_updates=0, granted_dcu=0,
                            accounting_columns=reader.COLUMNS, planned=reader.PLANNED,
                            terminal=dict(row=self.account, terminal_visible=True),
                            worker_receipt=dict(bytes=len(raw), SHA256=sha256(raw).hexdigest()))
        self.write(self.root/'outcome.json', self.outcome)

    def accounting(self, row=None):
        header = f'{reader.PARENT}|parent|RUNNING|0:0|123|nodes|cpu=2560,node=40,gres/dcu=320'
        return (header+'\n'+('|'.join(row)+'\n' if row else '')).encode()

    def verify(self, row=None, publication=None):
        return reader.verify(self.root, self.accounting(self.account if row is None else row),
                             self.publication if publication is None else publication,
                             data=self.data, training_sha=sha256(self.train).hexdigest())

    def test_actual_source_payload_and_72_rows_close_without_training(self):
        report = self.verify()
        self.assertTrue(report['passed'])
        self.assertEqual((report['actual_verified_rows'], report['observed_encode_calls'],
                          report['observed_collator_calls']), (72, 144, 72))
        self.assertEqual(report['total_shifted_supervised_tokens'], 648)
        self.assertEqual(report['maximum_row_tokens'], 272)
        self.assertFalse(report['training_READY'])
        self.assertFalse(report['original_replay_allowed'])

    def test_nonterminal_failed_or_wrong_resources_do_not_read_any_artifact(self):
        with patch.object(reader, 'read_file', side_effect=AssertionError('no artifact reading')):
            for state, resource, expected in (
                    ('RUNNING', self.account[-1], 'ORIGINAL_ACCOUNTING_UNKNOWN'),
                    ('TIMEOUT', self.account[-1], 'ORIGINAL_TERMINAL_UNQUALIFIED'),
                    ('COMPLETED', 'cpu=4,mem=8G,node=1,gres/dcu=1', 'ORIGINAL_TERMINAL_UNQUALIFIED')):
                row = [*self.account[:2], state, *self.account[3:-1], resource]
                self.assertEqual(self.verify(row)['state'], expected)
            self.assertEqual(self.verify([])['state'], 'ORIGINAL_ACCOUNTING_UNKNOWN')

    def test_missing_and_duplicate_slurm_steps_fail_closed(self):
        raw = self.accounting(self.account)
        with self.assertRaisesRegex(ValueError, 'more than once'):
            reader.phase_row(raw+('|'.join(self.account)+'\n').encode())
        wrong = self.account[:]
        wrong[5] = 'foreign-node'
        with self.assertRaisesRegex(ValueError, 'location differs'):
            reader.phase_row(self.accounting(wrong))

    def test_rejects_wrong_publication_worker_deadline_and_saved_accounting(self):
        self.assertIn('intent differs', self.verify(publication='b'*40)['failed_metadata_check'])
        self.save_worker(deadline=self.epoch+391)
        self.assertIn('result or scope differs', self.verify()['failed_metadata_check'])
        self.save_worker()
        self.outcome['terminal']['row'] = [*self.account[:4], '70', *self.account[5:]]
        self.write(self.root/'outcome.json', self.outcome)
        self.assertIn('live Slurm', self.verify()['failed_metadata_check'])

    def test_rejects_self_consistent_hashes_for_missing_inputs_and_invalid_records(self):
        original = self.capture[:]
        for bad, fragment in ((original[:-1], '72 ordered rows'),
                              ([dict(original[0], ordinal=2), *original[1:]], 'identity or order'),
                              ([dict(original[0], tokens=513), *original[1:]], 'token')):
            self.capture = bad
            self.save_worker()
            result = self.verify()
            self.assertFalse(result['passed'])
            self.assertIn(fragment, result['failed_metadata_check'])


if __name__ == '__main__':
    unittest.main()
