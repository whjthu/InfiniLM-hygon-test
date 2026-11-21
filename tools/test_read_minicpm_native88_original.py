import json
from pathlib import Path
import tempfile
import unittest

import read_minicpm_native88_original as audit


PARENT = b'1103606|original-parent|RUNNING|0:0|10|node-set|cpu=2560,node=40,gres/dcu=320'


class Native88ReaderTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pub = 'a'*40
        self.epoch = 1795000000.0
        self.row = ['1103606.700',audit.LABEL,'COMPLETED','0:0','40',audit.NODE,'cpu=8,mem=12G,node=1']
        self.reader_calls = 0

    def outcome(self, row=None):
        value = dict(request=audit.REQUEST,publication=self.pub,epoch=self.epoch,deadline=self.epoch+300,
                     accounting_columns=','.join(audit.COLUMNS),
                     terminal=dict(terminal_visible=True,row=self.row if row is None else row))
        (self.root/'outcome.json').write_text(json.dumps(value),encoding='utf-8')

    def metadata(self,context):
        self.reader_calls+=1
        self.assertEqual(context.source_manifest_sha256,audit.MANIFEST)
        self.assertEqual(context.claim_sha256,audit.CLAIM)
        self.assertEqual(context.epoch,self.epoch)
        return dict(state='MINICPM_USGS88_NATIVE_CPU_INDEPENDENT_METADATA_PASS',passed=True,
                    request=audit.REQUEST,publication=self.pub,actual_CPU_step=self.row[0],
                    actual_step_elapsed_seconds=40,completed_rows=88,actual_GPU_allocation=0,
                    actual_optimizer_updates=0,training_READY=False)

    def verify(self,row=True,loader=None):
        raw=PARENT+b'\n'+('|'.join(self.row).encode() if row else b'')
        return audit.verify(self.root,raw,self.pub,self.epoch,load_reader=loader or (lambda:self.metadata))

    def test_c0_uses_original_metadata_reader_with_bound_current_row(self):
        self.outcome()
        result=self.verify()
        self.assertTrue(result['passed'])
        self.assertEqual(result['actual_verified_rows'],88)
        self.assertEqual(result['independent_accounting']['row'],self.row)
        self.assertEqual(self.reader_calls,1)
        self.assertFalse(result['training_READY'])

    def test_missing_or_running_accounting_does_not_open_original_result(self):
        self.assertEqual(self.verify(row=False)['state'],'ORIGINAL_ACCOUNTING_UNKNOWN')
        self.row[2]='RUNNING'
        self.assertEqual(self.verify()['state'],'ORIGINAL_ACCOUNTING_UNKNOWN')
        self.assertEqual(self.reader_calls,0)

    def test_terminal_failure_preserves_original_row_without_reading_new_metadata(self):
        self.row[2:4]=['CANCELLED','0:15']
        result=self.verify()
        self.assertFalse(result['passed'])
        self.assertTrue(result['independent_accounting']['terminal'])
        self.assertEqual(result['independent_accounting']['row'],self.row)
        self.assertEqual(self.reader_calls,0)

    def test_positive_saved_row_drift_or_device_allocation_cannot_pass(self):
        other=self.row[:]
        other[0]='1103606.701'
        self.outcome(other)
        self.assertIn('does not bind',self.verify()['reader_error'])
        self.row[6]+=',gres/dcu=1'
        self.outcome()
        self.assertIn('resource or elapsed',self.verify()['reader_error'])
        self.assertEqual(self.reader_calls,0)

    def test_exact_reader_loads_without_executing_target_capture(self):
        self.assertTrue(callable(audit.source_reader()))


if __name__ == '__main__':
    unittest.main()
