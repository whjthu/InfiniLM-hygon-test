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
import view_schedule as views
import read_terminal
import dev24_decision
from bidirectional import reverse_pair
from checkpoint_io import checkpoint_sha256

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
        rows=[dict(global_optimizer_step=i,batch_cursor=(i-1)%views.VIEW_BATCHES,loss=.2,
                   grad_norm=.01,camera_head_grad_norm_before_clip=.01) for i in range(1,26)]
        c.verify_steps(rows)
        self.assertEqual([sum(views.ORDER[r['batch_cursor']][0]==i for r in rows) for i in range(7)], [4,4,4,4,3,3,3])
        self.assertEqual(sum(views.ORDER[r["batch_cursor"]][1]=="forward" for r in rows), 13)
        self.assertEqual([views.ORDER[r["batch_cursor"]][0] for r in rows], [i % 7 for i in range(25)])
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

    def test_orientation_order_and_frame_identity(self):
        rows = views.bindings(views.PAIR_PINS)
        for row in rows:
            start = row["physical_index"] * 4
            row["input_frame_ids"] = [start, start+1][::(-1 if row["direction"]=="reverse" else 1)]
        views.validate_observed_views(rows)
        wrong = copy.deepcopy(rows)
        wrong[-1]["input_frame_ids"][0] += 1
        for invalid in (wrong, rows[:-1], rows[::-1]):
            with self.assertRaises(ValueError):
                views.validate_observed_views(invalid)
        with self.assertRaises(ValueError):
            views.bindings(views.PAIR_PINS[::-1])

    def test_reused_inverse_changes_camera_gauge_and_all_frame_fields(self):
        identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        e1 = [[0,-1,0,2],[1,0,0,3],[0,0,1,4]]
        raw = dict(seq_name=c.TRAIN, ids=[7,31], extrinsics=[identity,e1],
                   images=["rgbA","rgbB"], depths=["depthA","depthB"],
                   intrinsics=["intrA","intrB"], point_masks=["maskA","maskB"])
        result = reverse_pair(raw)
        self.assertEqual(result["ids"], [31,7])
        self.assertEqual(result["extrinsics"], [identity,[[0,1,0,-3],[-1,0,0,2],[0,0,1,-4]]])
        for key in ("images", "depths", "intrinsics", "point_masks"):
            self.assertEqual(result[key], raw[key][::-1])
        self.assertEqual(raw["extrinsics"][1], e1)

    def test_dev24_stop_gate_keeps_original_failed_windows(self):
        scalar = dict(development_AUC_at_5=.05, development_AUC_at_15=.2,
                      development_AUC_at_30=.39, failed_pairs=2)
        payload = dict(assessed_windows=24, input_pairs=22, actual_prediction_events=44,
                       raw_pose_predictions=dict(base=22,near7_bidir25=22),
                       AUC=dict(all_windows=dict(base=scalar,near7_bidir25=dict(scalar))))
        decision = dev24_decision.evaluate(payload)
        self.assertEqual(decision["decision"], "FOLLOWUP_MAY_BE_CONSIDERED")
        self.assertFalse(decision["promotion"])
        for changes in (dict(development_AUC_at_30=.38),
                        dict(development_AUC_at_15=.188), dict(failed_pairs=3)):
            payload["AUC"]["all_windows"]["near7_bidir25"] = dict(scalar, **changes)
            self.assertEqual(dev24_decision.evaluate(payload)["decision"], "STOP_NEAR_ONLY_AUGMENTATION")
        payload["AUC"]["all_windows"]["near7_bidir25"] = dict(scalar, failed_pairs=1)
        self.assertEqual(dev24_decision.evaluate(payload)["decision"], "MISSING_OR_INCOMPLETE_DEV24")
        self.assertEqual(dev24_decision.evaluate(dict(payload,assessed_windows=22))["decision"],
                         "MISSING_OR_INCOMPLETE_DEV24")

    def test_partial_terminal_retains_full_denominator(self):
        with TemporaryDirectory() as td:
            target = (Path(td)/c.LABEL).resolve()
            root = target/"arms/continuous"
            train = root/"train"
            train.mkdir(parents=True)
            (root/"evaluation").mkdir()
            data = [dict(global_optimizer_step=i) for i in range(1,4)]
            (train/"steps.ndjson").write_bytes(b"".join(c.encoded(x) for x in data))
            (root/"evaluation/predictions-sealed-private.json").write_bytes(c.encoded(
                dict(gold_read=False,rows=[dict(status="invalid")])))
            (target/"intent.json").write_bytes(c.encoded(dict(request=c.REQUEST,parent=c.PARENT,node=c.NODE,
                epoch=100,deadline=1900,publication="a"*40,claim_SHA256="b"*64,source_manifest_SHA256="c"*64)))
            (target/"outcome.json").write_bytes(c.encoded(dict(request=c.REQUEST,publication="a"*40,stages={})))
            result = read_terminal.consume(target, {}, {})
            self.assertEqual((result["planned_updates"],result["actual_updates"],result["missing_updates"]), (25,3,22))
            self.assertEqual((result["planned_predictions"],result["actual_prediction_slots"],
                              result["missing_prediction_slots"]), (4,1,3))
            self.assertFalse(result["qualified"])

    def test_streaming_checkpoint_identity(self):
        with TemporaryDirectory() as td:
            target = (Path(td)/"checkpoint.pt").resolve()
            raw = b"first" + bytes(4 << 20) + b"tail"
            target.write_bytes(raw)
            self.assertEqual(checkpoint_sha256(target), c.sha(raw))
            target.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "file type or size"):
                checkpoint_sha256(target)

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
        with TemporaryDirectory() as tmp:
            repo=Path(tmp)
            target=repo/launch.BASE
            shutil.copytree(HERE,target,ignore=shutil.ignore_patterns('__pycache__'))
            claim=json.loads((target/'CLAIM.json').read_bytes())
            manifest=json.loads((target/'SOURCE_FILES.json').read_bytes())
            def bind_claim(authorized):
                claim.update(root_GO=authorized,GPU_GO=authorized,training_GO=authorized)
                raw=c.encoded(claim)
                (target/'CLAIM.json').write_bytes(raw)
                manifest['files']['CLAIM.json']=dict(bytes=len(raw),SHA256=c.sha(raw))
                (target/'SOURCE_FILES.json').write_bytes(c.encoded(manifest))
            bind_claim(False)
            with self.assertRaisesRegex(ValueError,'claim'):
                launch.frozen(repo,'a'*40,epoch=100,deadline=1900)
            bind_claim(True)
            _,_,_,wire=launch.frozen(repo,'a'*40,epoch=100,deadline=1900)
            compile(wire,'held-wire','exec')
            params=ast.literal_eval(ast.parse(wire).body[0].value)
            self.assertTrue(params['directory'].startswith('/public/home/'))
            self.assertNotIn('\\',params['directory'])
            body=zlib.decompress(base64.b64decode(params['body']))
            payload=ast.literal_eval(ast.parse(body).body[0].value)
            self.assertEqual((payload['epoch'],payload['deadline']),(100,1900))
            self.assertEqual(payload['request'],c.REQUEST)
            self.assertEqual(params['request'],c.REQUEST)
            self.assertEqual(controller.REQUEST,c.REQUEST)
            self.assertRegex(c.REQUEST,r'^[0-9a-f]{32}$')
            for new_name in ('view_schedule.py','checkpoint_io.py','bidirectional.py'):
                self.assertIn('src/'+new_name,payload['files'])
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
