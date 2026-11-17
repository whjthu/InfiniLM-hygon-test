"""Pure stdlib negative-contract and AST compatibility checks; no Torch imports."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

D = Path(__file__).resolve().parent
P = json.loads((D / 'PARENT_PINS.json').read_bytes())
spec = importlib.util.spec_from_file_location('strict_input_metadata', D/'deltas/vlm/runtime/native_input_binding.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class BadInputs(unittest.TestCase):
    def test_original_bindings_are_not_enabled(self):
        for case, pin in P['case_contract_sources'].items():
            b = json.loads(Path(pin['binding_path']).read_bytes())
            self.assertNotIn('input_fingerprint_reference', b)
            with self.assertRaisesRegex(ValueError, 'initial digest'):
                M.check_contract(b)

    def test_changed_model_provider_GC_LoRA_contract_rejected(self):
        for case, pin in P['case_contract_sources'].items():
            original = json.loads(Path(pin['binding_path']).read_bytes())
            for field in M.CONTRACT_KEYS:
                changed = copy.deepcopy(original)
                changed[field] = None if changed.get(field) is not None else {}
                with self.subTest(case=case, field=field):
                    with self.assertRaises(ValueError):
                        M.check_contract(changed)

    def test_T01_not_routed_into_new_validator(self):
        with self.assertRaisesRegex(ValueError, 'T03/T04'):
            M.check_contract({'spec': {'case': 'T01.sft'}})

    def test_reference_pin_malformed_rejected(self):
        for value in (None, {}, {'path': 'relative'}, {'path': '/x', 'bytes': True, 'SHA256': None}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                M.read_reference(value)

    def test_reference_duplicate_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'reference.json'
            raw = b'{"case":"T03.sft","case":"T04.sft"}'
            path.write_bytes(raw)
            pin = dict(path=str(path), bytes=len(raw), SHA256=hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                M.decode_reference(raw)
            path.write_bytes(raw+b' ')
            with self.assertRaises(ValueError):
                M.read_reference(pin)

    def test_incomplete_schema_rejected_without_input_hash_fixtures(self):
        # Isolate the document schema gate using the real frozen metadata contract.
        # No accepted input digest, initial values, model or provider is fabricated.
        for case, frozen in M.CONTRACTS.items():
            with patch.object(M, 'check_contract', return_value=frozen):
                for doc in (None, {}, {'schema': 'native_first_two_inputs_v1'}):
                    with self.assertRaisesRegex(ValueError, 'schema'):
                        M.validate_document({'spec': {'case': case}}, doc, None)

    def test_wrong_algorithm_provenance_rejected(self):
        for case, frozen in M.CONTRACTS.items():
            doc = dict(schema='native_first_two_inputs_v1', case=case,
                       reference_request_id=frozen['reference_request_id'],
                       original_binding_SHA256=frozen['original_binding_SHA256'],
                       original_value_audit_SHA256=frozen['original_value_audit_SHA256'],
                       fingerprint_source_SHA256=None, projection=['args','kwargs'], world_size=8,
                       accumulation=4, source_receipts=frozen['source_receipts'], rows=[])
            with patch.object(M, 'check_contract', return_value=frozen):
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    M.validate_document({'spec': {'case': case}}, doc, None)

    def test_missing_extra_and_bad_rank_micro_step_rows(self):
        case='T03.sft'
        rows=[dict(rank=i//8,micro_ordinal=i%8+1,global_step=(i%8)//4,
                   path=None,bytes=None,file_SHA256=None,input_fingerprint_SHA256=None)
              for i in range(64)]
        # Deliberately missing all hashes; these are invalid schema fixtures,
        # never purported native inputs or accepted reference data.
        for bad in (rows[:-1], rows+[None], [None]*64):
            with self.assertRaises(ValueError):M.validate_rows(case,bad)
        for field,value in [('rank',True),('rank',0),('micro_ordinal',2),('global_step',1)]:
            bad=copy.deepcopy(rows);bad[8][field]=value
            with self.subTest(field=field,value=value),self.assertRaisesRegex(ValueError,'order'):
                M.validate_rows(case,bad)
        with self.assertRaisesRegex(ValueError,'identity'):
            M.validate_rows(case,rows)

    def test_exact_sixteen_original_initial_metadata_pins(self):
        saved=json.loads((D/'ORIGINAL_INITIAL_PINS.json').read_bytes())
        self.assertEqual(saved['cases'],M.ORIGINAL_INITIAL_PINS)
        self.assertEqual(saved['independent_result_SHA256'],M.ORIGINAL_INITIAL_SOURCE_SHA256)
        for case, rows in M.ORIGINAL_INITIAL_PINS.items():
            original=json.loads(Path(P['case_contract_sources'][case]['binding_path']).read_bytes())
            self.assertEqual(len({r['initial_file_SHA256'] for r in rows}),8)
            for rank,row in enumerate(rows):
                binding=copy.deepcopy(original)
                binding['initial_trainables_SHA256']=row['named_initial_digest']
                binding['initial_state']=dict(path=str(D/('metadata-only-rank-%02d.pt'%rank)),
                                             bytes=row['initial_file_bytes'],SHA256=row['initial_file_SHA256'])
                self.assertEqual(M.check_contract(binding,rank=rank),M.CONTRACTS[case])
                # Positive metadata comparison only; no file/model/tensor is opened.
                wrong=copy.deepcopy(binding)
                wrong['initial_trainables_SHA256']=row['initial_file_SHA256']
                with self.assertRaisesRegex(ValueError,'initial digest'):
                    M.check_contract(wrong,rank=rank)
                wrong=copy.deepcopy(binding)
                wrong['initial_state']['SHA256']=row['named_initial_digest']
                with self.assertRaisesRegex(ValueError,'qualified original case'):
                    M.check_contract(wrong,rank=rank)
                wrong=copy.deepcopy(binding)
                wrong['initial_state']['bytes']+=1
                with self.assertRaises(ValueError):M.check_contract(wrong,rank=rank)
                wrong=copy.deepcopy(binding)
                wrong['initial_state']['SHA256']=rows[(rank+1)%8]['initial_file_SHA256']
                with patch.object(M,'read_reference') as reader:
                    with self.assertRaisesRegex(ValueError,'exact original rank'):
                        M.strict_native_input_pins(wrong,rank,None)
                    reader.assert_not_called()
                with self.assertRaises(ValueError):M.check_contract(binding,rank=True)

    def test_parent_T01_and_remaining_plugin_AST_preserved(self):
        parent = Path(P['parent_plugin']['path']).read_bytes()
        self.assertEqual(hashlib.sha256(parent).hexdigest(), P['parent_plugin']['SHA256'])
        before = ast.parse(parent)
        after = ast.parse((D/'deltas/vlm/plugins/native_first_updates.py').read_bytes())
        old_begin = next(n for n in ast.walk(before) if isinstance(n, ast.FunctionDef) and n.name=='on_train_begin')
        new_begin = next(n for n in ast.walk(after) if isinstance(n, ast.FunctionDef) and n.name=='on_train_begin')
        body=[]
        for n in new_begin.body:
            if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('strict_inputs','strict_providers') for t in n.targets):
                continue
            if isinstance(n, ast.If) and 'input_fingerprint_reference' in ast.unparse(n.test):
                continue
            if isinstance(n, ast.If) and ast.unparse(n.test)=='strict_inputs':
                body.extend(n.orelse)
                continue
            body.append(n)
        new_begin.body=body
        old_guard=next(n for n in ast.walk(old_begin) if isinstance(n,ast.If) and 'expected frozen Qwen input fingerprints' in ast.unparse(n.body))
        # Select the inner rejecting guard, not its parent dispatch block.
        old_guard=next(n for n in ast.walk(old_begin) if isinstance(n,ast.If) and isinstance(n.body[0],ast.Raise) and 'expected frozen Qwen' in ast.unparse(n.body[0]))
        new_guard=next(n for n in ast.walk(new_begin) if isinstance(n,ast.If) and isinstance(n.body[0],ast.Raise) and 'expected frozen Qwen' in ast.unparse(n.body[0]))
        new_guard.test=copy.deepcopy(old_guard.test)
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BadInputs))
    assert 'torch' not in sys.modules and 'swift' not in sys.modules
    print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,Torch_imports=0,Swift_imports=0,
                         normalized_input_hashes_generated=0,actual_models=0,GPU_GO=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
