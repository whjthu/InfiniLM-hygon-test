"""Complete-file identity boundaries with tiny byte fixtures, never model weights."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from hygon_indexed_model_identity import inspect_listed_files,stat_identity


class ListedIdentity(unittest.TestCase):
    def fixture(self,root):
        (root/'transformer').mkdir();(root/'tokenizer').mkdir()
        h={'w':dict(dtype='F32',shape=[2],data_offsets=[0,8])}
        raw=json.dumps(h).encode();weight=len(raw).to_bytes(8,'little')+raw+b'\x01\x02\x03\x04'*2
        (root/'transformer/part.safetensors').write_bytes(weight)
        (root/'tokenizer/config.json').write_bytes(b'{"fixture":true}')
        (root/'empty.txt').write_bytes(b'')
        files={n:stat_identity(root/n) for n in ('transformer/part.safetensors','tokenizer/config.json','empty.txt')}
        return files

    def test_complete_multicomponent_hash_and_header(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);events=[]
            result=inspect_listed_files(root,files,check_deadline=lambda:None,progress=events.append)
            self.assertEqual((result['planned_files'],result['completed_files']),(3,3))
            self.assertEqual(result['full_file_bytes_read'],sum(p['bytes'] for p in files.values()))
            for n,v in result['files'].items():self.assertEqual(v['SHA256'],hashlib.sha256((root/n).read_bytes()).hexdigest())
            self.assertEqual(result['files']['transformer/part.safetensors']['tensors']['w']['shape'],[2])
            self.assertEqual(events[0]['stage'],'listed_plan')
            self.assertFalse(result['tensor_values_decoded'])

    def test_missing_late_file_rejects_before_any_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);(root/'transformer/part.safetensors').unlink();events=[]
            with self.assertRaisesRegex(ValueError,'stat'):inspect_listed_files(root,files,check_deadline=lambda:None,progress=events.append)
            self.assertEqual(events,[])

    def test_escape_and_noncanonical_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);anchor=next(iter(files.values()))
            for name in ('../outside','/absolute','a//b','a/./b','C:/outside','a\\b'):
                with self.subTest(name=name),self.assertRaises(ValueError):
                    inspect_listed_files(root,{name:anchor},check_deadline=lambda:None,progress=lambda _:None)

    def test_invalid_anchor_and_read_budget_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);anchor=dict(next(iter(files.values())))
            for size in (True,-1,65*1024**3):
                pin=dict(anchor,bytes=size)
                with self.subTest(size=size),self.assertRaises(ValueError):
                    inspect_listed_files(root,{'x':pin},check_deadline=lambda:None,progress=lambda _:None)

    def test_mutation_during_read_preserves_partial_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);events=[];calls=0
            def deadline():
                nonlocal calls
                calls+=1
                if events and events[-1].get('current_file')=='tokenizer/config.json':
                    (root/'transformer/part.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):inspect_listed_files(root,files,check_deadline=deadline,progress=events.append)
            self.assertEqual(events[0]['planned_files'],['empty.txt','tokenizer/config.json','transformer/part.safetensors'])
            self.assertEqual(events[-1]['completed_files'],['empty.txt','tokenizer/config.json'])

    def test_deadline_failure_keeps_initial_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);events=[]
            def deadline():
                if events:raise TimeoutError('fixture deadline')
            with self.assertRaises(TimeoutError):inspect_listed_files(root,files,check_deadline=deadline,progress=events.append)
            self.assertEqual(events[0]['stage'],'listed_plan')
            self.assertEqual(events[0]['completed_files'],[])

    def test_payload_layout_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);p=root/'transformer/part.safetensors'
            p.write_bytes(p.read_bytes()+b'extra');files['transformer/part.safetensors']=stat_identity(p)
            with self.assertRaisesRegex(ValueError,'completely described'):
                inspect_listed_files(root,files,check_deadline=lambda:None,progress=lambda _:None)

    def test_file_count_bound_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);files=self.fixture(root);anchor=next(iter(files.values()))
            with self.assertRaisesRegex(ValueError,'bounded'):
                inspect_listed_files(root,{str(i):anchor for i in range(129)},check_deadline=lambda:None,progress=lambda _:None)


if __name__=='__main__':unittest.main()
