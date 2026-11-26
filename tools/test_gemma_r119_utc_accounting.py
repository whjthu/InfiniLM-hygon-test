"""Local fixtures for the R119 accounting UTC boundary, with no Slurm invocation."""

from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


API=runpy.run_path(str(Path(__file__).with_name('gemma_r119_utc_accounting.py')))
STEPS=('1103606.3504','1103606.3505','1103606.4010','1103606.4079')
ROWS=[
    [STEPS[0],'fixture-historic-gpu','FAILED','1:0','10','f13r1n12','cpu=64,gres/dcu=8,mem=224000M,node=1','2026-11-25T15:45:01','2026-11-25T15:45:11'],
    [STEPS[1],'fixture-historic-post','COMPLETED','0:0','0','f13r1n12','cpu=2,mem=4G,node=1','2026-11-25T15:45:13','2026-11-25T15:45:13'],
    [STEPS[2],'fixture-later-holder','COMPLETED','0:0','101','f13r1n12','cpu=8,gres/dcu=8,mem=16G,node=1','2026-11-26T00:00:00','2026-11-26T00:01:41'],
    [STEPS[3],'fixture-later-post','COMPLETED','0:0','0','f13r1n12','cpu=2,mem=4G,node=1','2026-11-26T00:01:45','2026-11-26T00:01:45'],
]


class AccountingTest(unittest.TestCase):
    def test_real_producer_argv_with_injected_process(self):
        calls=[]
        def run(argv, **kwargs):
            calls.append((argv,kwargs))
            raw='\n'.join('|'.join(row) for row in reversed(ROWS))+'\n'
            return SimpleNamespace(returncode=0, stdout=raw.encode(), stderr=b'')
        doc=API['collect_exact'](STEPS,run=run)
        self.assertEqual(doc['rows'],ROWS)
        argv,arguments=calls.pop()
        self.assertEqual(argv[:6],['env','TZ=UTC','LC_ALL=C','SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S','sacct','-nP'])
        self.assertEqual(argv[-4:],['-j',','.join(STEPS),'--format',
                            'JobIDRaw,JobName%120,State,ExitCode,ElapsedRaw,NodeList,AllocTRES,Start,End'])
        self.assertTrue(arguments['capture_output'])
        self.assertEqual(arguments['timeout'],10)
        self.assertFalse(calls)

    def test_eight_hour_display_offset_rejected_without_repair(self):
        tz_local=deepcopy(ROWS)
        for row in tz_local:
            row[7:9]=[(datetime.fromisoformat(s)+timedelta(seconds=28800)).isoformat() for s in row[7:9]]
        expected=API['evidence'](ROWS,STEPS)
        observed=API['evidence'](tz_local,STEPS)
        self.assertEqual(API['compare'](expected,observed)['different_columns'],['End','Start'])
        self.assertFalse(API['compare'](expected,observed)['timestamp_offset_repair_applied'])
        self.assertFalse(API['compare'](expected,observed)['passed'])
        observed['TZ']='Asia/Shanghai'
        with self.assertRaisesRegex(ValueError,'EXPLICIT_UTC'):
            API['compare'](expected,observed)

    def test_every_column_and_exact_digest(self):
        expected=API['evidence'](ROWS,STEPS)
        names={1:'other-job',2:'FAILED',3:'2:0',5:'f13r1n03',6:'cpu=2,mem=8G,node=1'}
        for column,new in names.items():
            with self.subTest(column=column):
                changed=deepcopy(ROWS)
                changed[1][column]=new
                observed=API['evidence'](changed,STEPS)
                self.assertEqual(API['compare'](expected,observed)['different_columns'],[API['COLUMNS'][column]])
        self.assertTrue(API['compare'](expected,API['evidence'](ROWS,reversed(STEPS)))['passed'])
        corrupted=deepcopy(expected)
        corrupted['rows'][1][6]='cpu=2,mem=8G,node=1'
        with self.assertRaisesRegex(ValueError,'EXACT_CANONICAL_ROWS_HASH'):
            API['validate'](corrupted)

    def test_no_unknown_duplicate_or_unbounded_row(self):
        for index,new in ((0,STEPS[0]),(1,'cut+'),(2,'RUNNING'),(4,'999'),(7,'Unknown'),(8,'2026-11-26T00:00:00+08:00')):
            with self.subTest(index=index):
                changed=deepcopy(ROWS)
                changed[1][index]=new
                with self.assertRaises(ValueError):
                    API['evidence'](changed,STEPS)
        with self.assertRaises(ValueError):
            API['evidence'](ROWS[:-1],STEPS)

    def test_exact_envelope_file_bytes(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'evidence.json'
            doc=API['evidence'](ROWS,STEPS)
            raw=json.dumps(doc).encode()
            path.write_bytes(raw)
            self.assertEqual(API['read_pinned'](path,sha256(raw).hexdigest()),doc)
            with self.assertRaisesRegex(ValueError,'EXACT_ENVELOPE_SHA256'):
                API['read_pinned'](path,'0'*64)


if __name__=='__main__':
    unittest.main()
