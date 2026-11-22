"""New resume/gate/comparison fixtures only. No model, Torch, Slurm or native data."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from contract import D,REQUEST,DISPATCH_REQUEST,validate_prior_gate
from compare_recovery import compare
from controller import container_command


def write(root,relative,data):
    p=root/relative;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data),encoding='utf8')


def synthetic_pair(root):
    for rank in range(8):
        state=lambda step:dict(adapter={'SHA256':'adapter'+str(step),'finite':True},
                               optimizer={'SHA256':'optimizer'+str(step),'finite':True},
                               scheduler={'SHA256':'scheduler'+str(step),'finite':True})
        for arm,start,count,cps in [('continuous',0,100,[10,20,25]),('resumed',20,20,[25])]:
            kind='continuous-witness' if arm=='continuous' else 'resume-witness'
            prefix=Path(arm)/'train'/kind/('rank-%02d'%rank)
            inputs=[dict(ordinal=i+1,global_step=start+i//4,
                         full_input={'SHA256':'input'+str(rank)+'-'+str(start*4+i)}) for i in range(count)]
            outputs=[dict(ordinal=i+1,loss={'SHA256':'loss'+str(rank)+'-'+str(start*4+i)},
                          loss_scalar=float(start*4+i)/1000) for i in range(count)]
            result=dict(rank=rank,world_size=8,completed_global_step=25,inputs=inputs,outputs=outputs,
                        saved_checkpoints=cps,steps=[dict(global_step=i,state=state(i)) for i in range(start+1,26)])
            if arm=='resumed':
                result.update(native_resume_window_complete=True,
                    RNG_loads=[dict(saved_equal=True,global_step=20,actual={'SHA256':'RNG20-'+str(rank)})])
                write(root,prefix/'native-data-skip.json',dict(rank=rank,before=100,skipped=80,after=20))
                write(root,prefix/'restored-before-RNG.json',dict(restored_parameter_optimizer_scheduler_equal=True,actual=state(20)))
            write(root,prefix/'result.json',result)
            write(root,prefix/'owner.json',dict(rank=rank,local_rank=rank,
                PID=100+rank+(100 if arm=='resumed' else 0),startticks=1000+rank+(100 if arm=='resumed' else 0)))
            for step in cps:
                write(root,prefix/('checkpoint-%d-rng.json'%step),dict(rank=rank,RNG={'SHA256':'RNG'+str(step)+'-'+str(rank)}))


class Resume(unittest.TestCase):
    def test_prior_complete25_required(self):
        names={'adapter_model.safetensors','adapter_config.json','trainer_state.json','args.json',
               'optimizer.pt','scheduler.pt',*(f'rng_state_{i}.pth' for i in range(8))}
        gate=dict(request_id=REQUEST,passed=True,independent_passed=True,accounting_C0=True,
            owned_cleanup_passed=True,actual_updates=25,arm='continuous',ranks_completed=8,
            microbatches_per_rank=100,checkpoint20_files={n:dict(bytes=1,SHA256='a'*64) for n in names})
        self.assertIs(validate_prior_gate(gate),gate)
        for change in ({'actual_updates':24},{'independent_passed':False},{'ranks_completed':7},
                       {'checkpoint20_files':{}},{'arm':'resumed'}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_prior_gate(dict(gate,**change))

    def test_exact_comparer_success_mismatch_and_partial_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);synthetic_pair(root)
            result=compare(root)
            self.assertTrue(result['exact_paired_recovery'])
            self.assertEqual(result['compared_tail_microbatch_pairs'],160)
            self.assertEqual(result['compared_rank_update_pairs'],40)
            path=root/'resumed/train/resume-witness/rank-03/result.json'
            altered=json.loads(path.read_bytes());altered['outputs'][0]['loss_scalar']+=.125
            path.write_text(json.dumps(altered),encoding='utf8')
            result=compare(root)
            self.assertFalse(result['exact_paired_recovery'])
            self.assertEqual(result['compared_tail_microbatch_pairs'],160)
            self.assertEqual(result['compared_rank_update_pairs'],40)
            self.assertAlmostEqual(result['maximum_absolute_loss_gap'],.125)
            path.unlink() # only this temporary synthetic fixture, never an original
            partial=compare(root)
            self.assertFalse(partial['exact_paired_recovery'])
            self.assertEqual(partial['planned_tail_microbatch_pairs'],160)
            self.assertEqual(partial['planned_rank_update_pairs'],40)
            self.assertEqual(partial['compared_tail_microbatch_pairs'],140)

    def test_resume_entry_and_immutable_continuous_mounts(self):
        source=(D/'supervisor.py').read_text()
        self.assertIn("str(CORE/'worker.py'),'resumed'",source)
        self.assertNotIn("str(CORE/'worker.py'),'continuous'",source)
        self.assertIn('continuous_independent_terminal=CONTINUOUS_GATE_PIN',source)
        self.assertNotIn('initial_trainables',source)
        command=container_command();text=' '.join(command)
        self.assertIn('continuous:ro',text)
        self.assertIn('continuous-independent-terminal.json:ro',text)
        self.assertEqual(command[-1],'gemma-resume20to25')
        self.assertNotEqual(REQUEST,DISPATCH_REQUEST)
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp)
            for p in D.glob('*.py'):(temp/p.name).write_bytes(p.read_bytes())
            code='import sys,runpy;sys.path.insert(0,'+repr(str(temp))+');'
            code+=';'.join('runpy.run_path('+repr(str(temp/n))+',run_name="source_only")' for n in
                          ['controller.py','supervisor.py','read_terminal.py'])
            code+=';assert "torch" not in sys.modules;print("RESUME_SOURCE_IMPORT_PASS")'
            x=subprocess.run([sys.executable,'-I','-S','-B','-c',code],capture_output=True,timeout=10)
            self.assertEqual(x.returncode,0,x.stderr.decode())
            self.assertEqual(x.stdout.strip(),b'RESUME_SOURCE_IMPORT_PASS')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Resume))
    print(json.dumps(dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        fixture_only=True,Torch_imported='torch' in sys.modules,Slurm_calls=0,GPU_calls=0,
        actual_updates=0,stage_restarted=False,continuous25_restarted=False,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.py')}),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
