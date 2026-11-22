"""New split/launcher/metric boundaries only; no native/model/target execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import shutil
import base64
import zlib

import split_contract as c
import run_tum_continuous_once as controller

HERE=Path(__file__).resolve().parent


def roster():
    return [dict(pin=c.sha(str(i).encode()),pair=dict(sequence=sequence,
        files_SHA256={name:c.sha(f'{i}/{name}'.encode()) for name in
        ('a.png','b.png','c.png','d.png','rgb.txt','depth.txt','groundtruth.txt')}))
        for i,sequence in enumerate([c.TRAIN]*7+list(c.HOLDOUT))]


class Contract(unittest.TestCase):
    def test_whole_recording_split_and_collision_rejected(self):
        train,held=c.partition(roster())
        self.assertEqual((len(train),len(held)),(7,2))
        bad=roster()
        bad[-1]['pair']['files_SHA256']['a.png']=bad[0]['pair']['files_SHA256']['a.png']
        with self.assertRaisesRegex(ValueError,'duplicate'):
            c.partition(bad)
        bad=roster()
        bad[-1]['pair']['sequence']=c.TRAIN
        with self.assertRaises(ValueError):
            c.partition(bad)

    def test_positive_25_steps_and_complete_cursor(self):
        rows=[dict(global_optimizer_step=i,batch_cursor=(i-1)%7,loss=.2,
                   grad_norm=.01,camera_head_grad_norm_before_clip=.01) for i in range(1,26)]
        c.verify_steps(rows)
        self.assertEqual([sum(r['batch_cursor']==i for r in rows) for i in range(7)], [4,4,4,4,3,3,3])
        for wrong in (rows[:-1], [dict(r,camera_head_grad_norm_before_clip=0) for r in rows],
                      [dict(r,batch_cursor=0) for r in rows]):
            with self.assertRaises(ValueError):
                c.verify_steps(wrong)

    def test_original_clock_and_exact_accounting(self):
        c.check_clock(100,1900,200)
        for args in ((100,1900,99),(100,1900,1890),(float('nan'),1900,200),(100,1901,200)):
            with self.assertRaises(ValueError):
                c.check_clock(*args)
        row=f'{c.PARENT}.999|{c.LABEL}-gpu|COMPLETED|0:0|8|{c.NODE}|cpu=4,gres/dcu=1,node=1\n'
        with patch.object(controller.subprocess,'check_output',return_value=row):
            self.assertEqual(controller.accounting(c.LABEL+'-gpu')[0], c.PARENT+'.999')
        for text in (row+'malformed\n', row.replace(c.PARENT+'.999','123.1'), row+row):
            with patch.object(controller.subprocess,'check_output',return_value=text):
                with self.assertRaises(ValueError):
                    controller.accounting(c.LABEL+'-gpu')

    def test_canonical_metric_retains_failed_pair(self):
        spec=importlib.util.spec_from_file_location('frozen_pose_metric',HERE/'score_vggt_tum_pair.py')
        metric=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metric)
        rows=[dict(sequence=s,frame_ids=[0,1],baseline=dict(pair_error_deg=0 if i==0 else None),
                   updated=dict(pair_error_deg=None)) for i,s in enumerate(c.HOLDOUT)]
        value=metric.summarize_development(rows)
        self.assertEqual(value['pairs'],2)
        self.assertEqual(value['baseline']['development_AUC_at_5'],.5)
        self.assertEqual(value['updated']['failed_pairs'],2)
        self.assertEqual(value['updated']['development_AUC_at_30'],0)
        self.assertEqual(c.sha((HERE/'score_vggt_tum_pair.py').read_bytes()),
                         '6454c9a5f10459e5f2eb8b268918d1303d3f04c2038a12cca22c20ac87b58e22')

    def test_runtime_hook_and_prediction_seal_source(self):
        source=(HERE/'run_tum_continuous_once.py').read_text(encoding='utf8')
        self.assertIn('src/domain_overlay.py:/sealed/framework/generation/src/gencuda/domain_overlay.py:ro',source)
        self.assertIn('for arm in ("continuous",):',source)
        self.assertNotIn('from compare_resume import compare',source)
        evaluation=(HERE/'evaluate.py').read_text(encoding='utf8')
        self.assertLess(evaluation.index("sealed = save(output/'predictions-sealed-private.json'"),
                        evaluation.index("truth, _ = prepare_pair("))
        for path in HERE.glob('*.py'):
            ast.parse(path.read_bytes(),feature_version=(3,10))

    def test_operative_claim_ingress_and_windows_built_posix_wire(self):
        import launch
        with self.assertRaisesRegex(ValueError,'claim'):
            launch.frozen(launch.REPO,'a'*40,epoch=100,deadline=1900)
        with TemporaryDirectory() as tmp:
            repo=Path(tmp)
            target=repo/launch.BASE
            shutil.copytree(HERE,target,ignore=shutil.ignore_patterns('__pycache__'))
            claim=json.loads((target/'CLAIM.json').read_bytes())
            claim.update(root_GO=True,GPU_GO=True,training_GO=True)
            raw=c.encoded(claim)
            (target/'CLAIM.json').write_bytes(raw)
            manifest=json.loads((target/'SOURCE_FILES.json').read_bytes())
            manifest['files']['CLAIM.json']=dict(bytes=len(raw),SHA256=c.sha(raw))
            (target/'SOURCE_FILES.json').write_bytes(c.encoded(manifest))
            _,_,_,wire=launch.frozen(repo,'a'*40,epoch=100,deadline=1900)
            compile(wire,'held-wire','exec')
            params=ast.literal_eval(ast.parse(wire).body[0].value)
            self.assertTrue(params['directory'].startswith('/public/home/'))
            self.assertNotIn('\\',params['directory'])
            body=zlib.decompress(base64.b64decode(params['body']))
            payload=ast.literal_eval(ast.parse(body).body[0].value)
            self.assertEqual((payload['epoch'],payload['deadline']),(100,1900))
            self.assertEqual(payload['request'],c.REQUEST)
            self.assertIn('src/domain_overlay.py',payload['files'])
            self.assertIn(b'os.execv(',wire)
            self.assertNotIn(b'start_new_session=True',wire)
            self.assertLess(len(wire),131072)
            # This file actually executes inside the container; imports must resolve with only delivered members.
            sources=target/'clean-target-src'
            sources.mkdir()
            for name,item in payload['files'].items():
                if name.startswith('src/'):
                    (sources/Path(name).name).write_bytes(base64.b64decode(item['b64']))
            import subprocess,sys
            outcome=subprocess.run([sys.executable,'-I','-S','-B','-c',
                "import sys;sys.path.insert(0,sys.argv[1]);import split_contract,run_tum_continuous_once,read_terminal;print('CLEAN_STDLIB_IMPORT_OK')",
                str(sources)],capture_output=True,text=True,timeout=10)
            self.assertEqual(outcome.returncode,0,outcome.stderr)
            self.assertEqual(outcome.stdout.strip(),'CLEAN_STDLIB_IMPORT_OK')


if __name__=='__main__':
    unittest.main()
