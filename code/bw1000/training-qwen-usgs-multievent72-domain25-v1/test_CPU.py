"""Actual source ABI plus small rejection fixtures. No Torch, remote or native execution."""
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import shutil
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import qwen_domain_recipe as recipe
import native72
from metric_worker import eos,CAP
import prepare_dev
from update_contract import check_update,check_complete

D=Path(__file__).resolve().parent
G=D.parents[2]


def fake_cpu():
    return dict(state='NATIVE_QWEN_MULTIEVENT72_PASS',rows=72,dataset_SHA256=recipe.DATASET_SHA256,
        labels_and_EOS_valid=72,full_sequence_retained=72,actual_GPU_allocated=0,explicit_GPU_queries=0,
        actual_updates=0,step='1103606.999',collator_rows=72,max_length=4096,max_pixels=262144)


class Contract(unittest.TestCase):
    def test_actual_original_common_cli_binding(self):
        translator=G/'code/bw1000/training-qwen-target-native-CPU-v1/deltas/tools/hygon_vlm_recipe.py'
        original=(G/'code/bw1000/training-qwen-native-inputs-CPU-v1/original-launch.json').read_bytes()
        self.assertEqual(hashlib.sha256(translator.read_bytes()).hexdigest(),recipe.TRANSLATOR_SHA256)
        self.assertEqual(hashlib.sha256(original).hexdigest(),recipe.ORIGINAL_SHA256)
        with TemporaryDirectory() as tmp:
            framework=Path(tmp)/'framework';(framework/'tools').mkdir(parents=True)
            (framework/'tools/hygon_vlm_recipe.py').write_bytes(translator.read_bytes())
            root=PurePosixPath('/public/home/test/newscope')
            plan=recipe.make_plan(framework,original,json.dumps(fake_cpu()).encode(),root,
                root/'src/qwen_treasury_witness.py',29941,'continuous',root/'view/training800.jsonl','a'*64)
            command=plan['command'];value=lambda k:command[command.index(k)+1]
            self.assertEqual(value('--model'),str(recipe.MODEL))
            self.assertEqual(value('--max_steps'),'25')
            self.assertEqual(value('--max_length'),'4096')
            self.assertEqual(value('--gradient_accumulation_steps'),'4')
            self.assertEqual(value('--gradient_checkpointing'),'true')
            self.assertEqual(value('--model_type'),'qwen3_vl')
            self.assertIn('qwen_treasury_continuous_witness',command)
            self.assertNotIn('nvidia_baseline',command)
            self.assertNotIn('--resume_from_checkpoint',command)

    def test_new_native_gate_cannot_inherit_gemma_or88(self):
        self.assertEqual(recipe.check_preflight(json.dumps(fake_cpu()).encode()),'1103606.999')
        for changes in ({'rows':88},{'state':'GEMMA_NATIVE72_PASS'},{'labels_and_EOS_valid':71},
                        {'actual_GPU_allocated':1},{'actual_updates':1},{'full_sequence_retained':71}):
            with self.assertRaises(ValueError):recipe.check_preflight(json.dumps(dict(fake_cpu(),**changes)).encode())

    def test_real_template_suffix_not_arbitrary_after_eos(self):
        native72.check_suffix([3,9,10],[-100,9,10],[9,10],9)
        native72.check_suffix([3,9,10],[-100,9,-100],[9,10],9)
        for ids,labels in (([3,9,11],[-100,9,11]),([3,9,10],[-100,-100,10])):
            with self.assertRaises(ValueError):native72.check_suffix(ids,labels,[9,10],9)

    def test_eos_and_missing_cap_keep_real_denominator(self):
        self.assertTrue(eos([5,9],[9])['natural_EOS'])
        self.assertFalse(eos([9,5],[9])['natural_EOS'])
        self.assertFalse(eos([9,9],[9])['natural_EOS'])
        self.assertTrue(eos([5]*(CAP-1)+[9],[9])['cap_hit'])
        for ids in ([],[True,9],[1]*(CAP+1)):
            with self.assertRaises(ValueError):eos(ids,[9])
        from frozen_score import score
        label=dict(id='synthetic',data_status='QUALIFIED',answer_type='decimal2',answer='0.00',kind='depth_change',chart_id='x')
        value=score([label],[])
        self.assertEqual((value['denominator'],value['correct'],value['failures']),(1,0,1))

    def test_valid_lr0_step_counter_and_total_no_change_rejection(self):
        check_update(1,[1.0]*504,504,False,[0.0])
        with self.assertRaises(ValueError):check_update(2,[1.0]*504,504,True,[1e-4])
        with self.assertRaises(ValueError):check_complete([False]*25)
        self.assertEqual(check_complete([False]+[True]*24)['updates'],25)

    def test_cpu_guard_before_model_import_and_target_clean_import(self):
        with TemporaryDirectory() as tmp:
            model=Path(tmp)/'model';model.mkdir();(model/'unwanted.safetensors').write_bytes(b'x')
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1'),\
                 patch.object(native72.md,'version',side_effect=lambda n:native72.VERSIONS[n]):
                with self.assertRaisesRegex(ValueError,'metadata only'):
                    native72.check(Path(tmp)/'missing.jsonl',model,Path(tmp),1e20)
            staged=Path(tmp)/'delivered';staged.mkdir()
            for name in ('controller.py','mid_epoch_recovery.py','owned_processes.py','qwen_domain_recipe.py',
                         'slurm_accounting.py','snapshot.py','data_view.py','prepare_dev.py','native72.py',
                         'metric_worker.py','contracts.py','read_terminal.py'):
                (staged/name).write_bytes((D/name).read_bytes())
            result=subprocess.run([sys.executable,'-I','-S','-B','-c',
                "import sys;sys.path.insert(0,sys.argv[1]);import controller,native72,metric_worker,read_terminal;print('CLEAN_IMPORT_OK')",str(staged)],
                capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'CLEAN_IMPORT_OK')

    def test_transport_required_and_all_source_3_10_parse(self):
        import launch
        with self.assertRaises(TypeError):launch.main('a'*40)
        for p in D.glob('*.py'):ast.parse(p.read_bytes(),feature_version=(3,10))
        source=(D/'controller.py').read_text(encoding='utf8')
        self.assertLess(source.index("stages['native72']=step"),source.index('stages["gpu"] = step'))
        self.assertIn('return run_metric(deadline)',source)
        self.assertNotIn('comparison=compare(RUN)',source)

    def test_final_manifest_ingress_and_tightened_actual72_gate(self):
        import verify_package,receive
        verify_package.verify(D,adopted=False)
        with self.assertRaises(ValueError):verify_package.verify(D,adopted=True)
        mf=(D/'SOURCE_FILES.json').read_bytes()
        source={'SOURCE_FILES.json':base64.b64encode(mf).decode('ascii')}
        for name in json.loads(mf)['files']:source[name]=base64.b64encode((D/name).read_bytes()).decode('ascii')
        epoch=time.time()
        payload=dict(epoch=epoch,deadline=epoch+2400,request=verify_package.REQUEST,files=source,
            publication='a'*40,source_manifest_SHA256=hashlib.sha256(mf).hexdigest(),
            claim_SHA256=hashlib.sha256((D/'CLAIM.json').read_bytes()).hexdigest())
        with TemporaryDirectory() as tmp:
            with patch.object(receive,'ROOT',Path(tmp)/'must-not-create'):
                with self.assertRaises(AssertionError):receive.main(payload)
                self.assertFalse((Path(tmp)/'must-not-create').exists())
            root=Path(tmp)/'repo';target=root/D.relative_to(G)
            shutil.copytree(D,target,ignore=shutil.ignore_patterns('__pycache__'))
            claim=json.loads((target/'CLAIM.json').read_bytes())
            for key in ('root_GO','CPU_GO','GPU_GO','training_GO','READ_GO','metric_inference_GO','SCORE_GO'):claim[key]=True
            for ref in claim['public_prerequisites']:
                out=root/ref['path'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes((G/ref['path']).read_bytes())
            raw=(json.dumps(claim,sort_keys=True,indent=2)+'\n').encode();(target/'CLAIM.json').write_bytes(raw)
            manifest=json.loads(mf);manifest['files']['CLAIM.json']=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
            (target/'SOURCE_FILES.json').write_text(json.dumps(manifest),encoding='utf8')
            self.assertTrue(verify_package.verify(target,adopted=True)['passed'])
        for changes in ({'collator_rows':71},{'max_length':512},{'max_pixels':1048576}):
            with self.assertRaises(ValueError):recipe.check_preflight(json.dumps(dict(fake_cpu(),**changes)).encode())
        # Only this original ABI boundary is repeated after its CPU gate was tightened.
        self.test_actual_original_common_cli_binding()
        self.test_real_template_suffix_not_arbitrary_after_eos()


if __name__=='__main__':unittest.main()
