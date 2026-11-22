"""Pure mock input/native contracts, not actual72 data or processor qualification."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from transform import DATA,TRAIN_PIN,KINDS,PREFIX,canonical,convert_rows
from audit_protocol import audit_rows,check_sample
import worker

D=Path(__file__).resolve().parent

def fixture():
    return [dict(id=f'fixture-{chart}-{kind}',chart_id=f'chart-{chart}',group=f'group-{chart//4}',kind=kind,
        event_ids=[f'E{i:02d}' for i in range(1,7)],images=[DATA+f'/train/media/{chart:03d}.png'],
        messages=[dict(role='user',content=PREFIX+'unchanged fixture question '+kind+'\n'),
                  dict(role='assistant',content='FINAL: 0.00')]) for chart in range(12) for kind in KINDS]

def fp(value):return dict(SHA256=hashlib.sha256(canonical(value)).hexdigest(),finite=True)

class Template:
    def __init__(self,mode='good'):self.mode=mode
    def encode(self,row):
        value=dict(input_ids=[1,2,90,91],labels=[-100,-100,90,91])
        if self.mode=='pixel':value['pixel_values']=None
        if self.mode=='changed':value['input_ids']=[1,3,90,91]
        return value
    def data_collator(self,rows):
        row=rows[0]
        return dict(input_ids=[row['input_ids']],labels=[row['labels']],attention_mask=[[1]*len(row['input_ids'])])

class Contracts(unittest.TestCase):
    def test_mock72_only_marker_and_image_removed_exact_question_and_FINAL(self):
        source=fixture();before=copy.deepcopy(source);rows,meta=convert_rows(source)
        self.assertEqual(source,before);self.assertEqual(len(rows),72)
        self.assertEqual((meta['source_images'],meta['source_groups'],meta['source_charts']),(12,3,12))
        self.assertEqual(meta['output_images'],0);self.assertEqual(len(meta['witness']),72)
        for original,value in zip(source,rows):
            expected=copy.deepcopy(original);expected.pop('images');expected['messages'][0]['content']=expected['messages'][0]['content'][len(PREFIX):]
            self.assertEqual(value,expected);self.assertEqual(value['messages'][1],original['messages'][1])
        self.assertEqual([r['id'] for r in source],[r['id'] for r in rows])
        self.assertEqual(TRAIN_PIN['bytes'],54246)

    def test_mock_bad_media_prefix_duplicate_ID_and_changed_label_reject(self):
        mutations=[lambda r:r[0]['messages'][0].update(content='wrong '+PREFIX+'question'),
            lambda r:r[0]['messages'][0].update(content=PREFIX+'x <image>'),
            lambda r:r[0]['images'].__setitem__(0,DATA+'/dev/media/000.png'),
            lambda r:r[0].update(id=r[1]['id']),
            lambda r:r[0]['messages'][1].update(content='0.00')]
        for mutate in mutations:
            rows=fixture();mutate(rows)
            with self.assertRaises(ValueError):convert_rows(rows)
        with self.assertRaises(ValueError):convert_rows(fixture()[:-1])

    def test_mock_native_full72_positive_and_actual_call_counts(self):
        rows,_=convert_rows(fixture());events=[]
        result=audit_rows(rows,Template(),Template(),fp,[90,91],{90},lambda:None,lambda k,v:events.append(k))
        self.assertTrue(result['passed']);self.assertEqual((result['planned'],result['attempted'],result['completed'],result['failed'],result['missing']),(72,72,72,0,0))
        self.assertEqual(events.count('full_encode_attempt'),72);self.assertEqual(events.count('capped_encode_attempt'),72)
        self.assertEqual(events.count('collator_attempt'),72)
        self.assertTrue(all(r['pixel_fields_absent'] and r['shifted_supervised_tokens']==2 for r in result['records']))

    def test_mock_pixel_even_None_or_truncation_keeps72_failed_denominator(self):
        rows,_=convert_rows(fixture())
        for full,capped in ((Template('pixel'),Template('pixel')),(Template(),Template('changed'))):
            result=audit_rows(rows,full,capped,fp,[90,91],{90},lambda:None,lambda *_:None)
            self.assertFalse(result['passed']);self.assertEqual((result['attempted'],result['completed'],result['failed'],result['missing']),(72,0,72,0))
        with self.assertRaises(ValueError):check_sample([1,2,90,91],[-100,-100,90,-100],[90,91],{90})
        with self.assertRaises(ValueError):check_sample([1,2,90,91],[-100,-100,90,91],[90,91],{99})

    def test_source_and_versions_original511_separate_new_ABI_unknown(self):
        sha=hashlib.sha256((D/'SOURCE_FILES.json').read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'not adopted'):worker.verify(sha)
        self.assertEqual(len(worker.REQUEST),32)
        self.assertEqual(hashlib.sha256((D/'MODEL_PROCESSOR_FILES.json').read_bytes()).hexdigest(),worker.PROCESSOR_SHA)
        self.assertEqual(len(json.loads((D/'MODEL_PROCESSOR_FILES.json').read_bytes())['files']),14)
        self.assertEqual(hashlib.sha256((D/'FRAMEWORK_FILES.json').read_bytes()).hexdigest(),worker.FRAMEWORK_SHA)
        self.assertEqual(len(json.loads((D/'FRAMEWORK_FILES.json').read_bytes())['files']),393)
        original=D.parents[2]/'tools/audit_gemma_training_native.py'
        def signature(path):return ast.dump(next(x for x in ast.parse(path.read_bytes()).body if isinstance(x,ast.FunctionDef) and x.name=='check_sample'),include_attributes=False)
        self.assertEqual(signature(D/'audit_protocol.py'),signature(original))
        with self.assertRaises(ValueError):worker.clock(100,491,200)
        with self.assertRaises(ValueError):worker.clock(float('nan'),490,200)

    def test_clean_only_delivered_imports_and_no_local_source_reader(self):
        with TemporaryDirectory() as tmp:
            for p in D.iterdir():
                if p.is_file():(Path(tmp)/p.name).write_bytes(p.read_bytes())
            r=subprocess.run([sys.executable,'-I','-S','-B','-c',
                "import sys;sys.path.insert(0,sys.argv[1]);import worker,transform,audit_protocol;assert 'torch' not in sys.modules;print('CLEAN_CPU_SOURCE_IMPORT_OK')",tmp],capture_output=True,timeout=5)
            self.assertEqual(r.returncode,0,r.stderr)
        for p in D.glob('*.py'):ast.parse(p.read_bytes(),feature_version=(3,10))
        for p in (D/'worker.py',D/'transform.py'):
            raw=p.read_text(encoding='utf8');self.assertNotIn('local-only/',raw)
            self.assertNotIn('import subprocess',raw)

if __name__=='__main__':unittest.main()
