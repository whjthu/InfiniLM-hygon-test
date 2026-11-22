"""New protocol/ABI tests only. Synthetic predictions/labels, never actual dev or model."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import effect_contract as c
from content_vs_format import measure
from score_after import validate_seal

D=Path(__file__).resolve().parent

def predictions(text='0.00'):
    return [dict(id=f'fixture-{i}',ordinal=i,text=text,status='COMPLETED',natural_EOS=True,cap_hit=False) for i in range(36)]

class Scope(unittest.TestCase):
    def test_three_full32_ids_operative_false_and_no_target_defaults(self):
        ids=(c.REQUEST,c.READ_REQUEST,c.SCORE_REQUEST)
        self.assertEqual(len(set(ids)),3)
        for value in ids:self.assertEqual(len(value),32);self.assertFalse(set(value)-set('0123456789abcdef'))
        for kind in ('GPU','READ','SCORE'):
            with self.assertRaises(ValueError):c.verify(kind)
        import infer_dev36
        self.assertNotIn('torch',sys.modules)
        self.assertEqual(c.GENERIC_GATE['bytes'],8507)
        self.assertEqual(c.GENERIC_CP['SHA256'],'0e484f8efe89b58db5c6251954b9bc5e25e5098a3b4429834954d7cc7cf83ae3')

    def test_exact_checkpoint_schema_not_filename_shape_guess(self):
        names={'adapter_model.safetensors','adapter_config.json','trainer_state.json','args.json','optimizer.pt','scheduler.pt',
               *(f'rng_state_{i}.pth' for i in range(8))}
        record=dict(request_id='fixture',step=25,required_members=14,
            files={n:dict(bytes=1,SHA256='a'*64,file_identity=[1,2,1,3,4]) for n in names})
        self.assertEqual(len(c.checkpoint(record,'/fixture','fixture')['files']),14)
        for key,value in (('step',20),('required_members',13),('request_id','other')):
            bad=dict(record);bad[key]=value
            with self.assertRaises(ValueError):c.checkpoint(bad,'/fixture','fixture')
        bad=json.loads(json.dumps(record));bad['files'].pop('rng_state_7.pth')
        with self.assertRaises(ValueError):c.checkpoint(bad,'/fixture','fixture')

    def test_raw_prepad_EOS_and_full_cap_are_distinct(self):
        self.assertTrue(c.eos([1,2,9],[9])['natural_EOS'])
        self.assertFalse(c.eos([1,9,2],[9])['natural_EOS'])
        value=c.eos([1]*255+[9],[9]);self.assertTrue(value['cap_hit']);self.assertFalse(value['natural_EOS'])
        for bad in ([],[True],[1.0]):
            with self.assertRaises(ValueError):c.eos(bad,[9])

    def test_partial_seal_rejects_before_gold_and_empty_output_preserves_failure(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            empty=folder/'empty';empty.write_bytes(b'')
            self.assertEqual(c.read(empty,allow_empty=True),b'')
            with self.assertRaises(ValueError):c.read(empty)
            # Absent prediction directory and no labels are touched on early terminal failure.
            with self.assertRaisesRegex(ValueError,'BEFORE any labels'):
                validate_seal(dict(request=c.REQUEST,passed=False,independent_passed=False,
                    all108sealed_before_gold=False,dev_gold_read=False,blind_read=False),folder/'does-not-exist')
            good=dict(request=c.REQUEST,passed=True,independent_passed=True,all108sealed_before_gold=True,
                dev_gold_read=False,blind_read=False,sealed_predictions={})
            with self.assertRaisesRegex(ValueError,'exact sealed four'):validate_seal(good,folder)

    def test_actual_four_file_seal_pin_order_EOS_types(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);pins={}
            final=dict(request=c.REQUEST,passed=True,all108sealed_before_gold=True,all_prediction_files_closed=True,
                dev_gold_read=False,blind_read=False)
            for arm in c.ARMS:
                raw=b''.join((json.dumps(r)+'\n').encode() for r in predictions())
                name=arm+'-predictions-private.jsonl';(folder/name).write_bytes(raw);pins[name]=c.pin(raw)
            raw=json.dumps(final).encode();(folder/'generation-finalized.json').write_bytes(raw);pins['generation-finalized.json']=c.pin(raw)
            independent=dict(request=c.REQUEST,passed=True,independent_passed=True,all108sealed_before_gold=True,
                dev_gold_read=False,blind_read=False,sealed_predictions=pins)
            self.assertEqual(sum(map(len,validate_seal(independent,folder).values())),108)
            (folder/'generic25-predictions-private.jsonl').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_seal(independent,folder)

    def test_format_content_cap_and_missing_keep36(self):
        labels=[dict(id=f'fixture-{i}',kind='depth_change',answer_type='decimal2',answer='0.00',
            event_ids=['E01','E02'],data_status='QUALIFIED',chart_id='fixture',group='fixture') for i in range(36)]
        rows=dict(base=predictions('0.00'),domain25=predictions('FINAL: 1.00'),generic25=predictions('FINAL: 0.00'))
        value=measure(labels,rows)
        self.assertEqual(value['arms']['base']['content_correct'],36)
        self.assertEqual(value['arms']['base']['strict_FINAL'],0)
        self.assertEqual(value['arms']['domain25']['strict_FINAL'],36)
        self.assertEqual(value['arms']['domain25']['content_correct'],0)
        self.assertEqual(value['domain_vs_generic']['both_parsed_loss'],36)
        rows['generic25'][0].update(cap_hit=True,natural_EOS=False)
        self.assertEqual(measure(labels,rows)['arms']['generic25']['content_correct'],35)
        empty=measure(labels,{a:[] for a in c.ARMS})
        self.assertTrue(all(x['missing']==x['denominator']==36 for x in empty['arms'].values()))

    def test_original_gold_free_dev_and_scorer_are_exact_copies_clean_import(self):
        parent=D.parent/'training-qwen-usgs-multievent72-domain25-v1'
        for name in ('contracts.py','frozen_score.py','response_protocol.py'):
            self.assertEqual((D/name).read_bytes(),(parent/name).read_bytes())
        self.assertEqual((D/'dev_inputs.py').read_bytes(),(parent/'prepare_dev.py').read_bytes())
        with TemporaryDirectory() as tmp:
            for p in D.iterdir():
                if p.is_file():(Path(tmp)/p.name).write_bytes(p.read_bytes())
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',
                "import sys;sys.path.insert(0,sys.argv[1]);import bind_ready,infer_dev36,read_terminal,score_after;assert 'torch' not in sys.modules;print('CLEAN_EFFECT_IMPORT_OK')",tmp],
                capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
        for p in D.glob('*.py'):ast.parse(p.read_bytes(),feature_version=(3,10))
        with self.assertRaises(ValueError):c.clock(float('nan'),100,50)
        with self.assertRaises(ValueError):c.clock(100,1000,99)
        self.assertEqual(c.clock(100,1000,200),800)

    def test_windows_ctime_provenance_and_linux_full_identity(self):
        path_stat=[1,2,3,4,5];fd_stat=[1,2,3,4,6]
        self.assertTrue(c.open_identity_matches(path_stat,fd_stat,windows=True))
        self.assertFalse(c.open_identity_matches(path_stat,fd_stat,windows=False))
        for index in range(4):
            changed=list(fd_stat);changed[index]+=1
            self.assertFalse(c.open_identity_matches(path_stat,changed,windows=True))
        self.assertTrue(c.open_identity_matches(path_stat,path_stat,windows=False))

    def test_binding_never_substitutes_old_initial_or_future_checkpoint(self):
        source=dict(bytes=1,SHA256='a'*64)
        value=dict(state='ACTUAL_THREE_ARM_GEMMA25_BINDING',request=c.REQUEST,source_manifest=source,
            domain_terminal=c.DOMAIN_GATE,generic_terminal=c.GENERIC_GATE,initial_named_SHA256=c.NAMED,
            initial=dict(path=str(c.DOMAIN/'continuous/train/continuous-witness/rank-00/target-fresh-initial.pt')),
            checkpoints=dict(domain25=dict(root=str(c.DOMAIN/'continuous/train/checkpoint-25')),
                             generic25=dict(root=str(c.GENERIC/'continuous/train/checkpoint-25'))),
            framework_root=str(c.DOMAIN/'framework'),framework=dict(SHA256=c.FRAMEWORK_SHA),runtime=dict(SHA256=c.RUNTIME_SHA),
            model_root=str(c.R/'results/training-gemma-original-model-v1/gemma-3-4b-it'),planned_each=36,planned_total=108,
            generation_cap=256,dev_gold_read=False,blind_read=False)
        c.binding_scope(value,source)
        for key,wrong in (('initial_named_SHA256','b'*64),('generic_terminal',None),('planned_total',72)):
            bad=dict(value);bad[key]=wrong
            with self.assertRaises(ValueError):c.binding_scope(bad,source)
        bad=json.loads(json.dumps(value));bad['initial']['path']='/original-A100/initial.pt'
        with self.assertRaises(ValueError):c.binding_scope(bad,source)

if __name__=='__main__':unittest.main()
