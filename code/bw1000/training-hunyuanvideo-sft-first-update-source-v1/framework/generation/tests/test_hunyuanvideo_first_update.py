import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from gencuda.hunyuanvideo_first_update import BOUNDARIES, instrument_source


class HunyuanFirstUpdateTest(unittest.TestCase):
    def test_actual_backend_matches_and_rejects_moved_clip_boundary(self):
        path = SRC / 'gencuda/backends/hunyuanvideo_training_walltime.py'
        source = path.read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'run_hunyuanvideo_training')
        original = ast.get_source_segment(source, node)
        instrument_source(original)
        with self.assertRaisesRegex(ValueError, 'boundaries changed'):
            instrument_source(original.replace('clip_grad_norm_(trainable, max_grad_norm)',
                                               'clip_grad_norm_(trainable, 1.0)'))

    def test_only_first_native_step_is_observed(self):
        trace = []
        record = lambda phase: lambda *a, **kw: trace.append(phase)
        loss = SimpleNamespace(backward=record('backward'))
        torch = SimpleNamespace(
            cat=record('cat'),
            nn=SimpleNamespace(functional=SimpleNamespace(mse_loss=lambda *_:loss),
                               utils=SimpleNamespace(clip_grad_norm_=record('clip'))))
        optimizer = SimpleNamespace(zero_grad=record('zero'),step=record('step'))
        observer = SimpleNamespace(**{phase:record(phase) for phase,_ in BOUNDARIES.values()})
        declarations = ('transformer = latents = cond = text_emb = text_mask = byt5_text_states = '
                        'byt5_text_mask = latents_noised = target = prediction = noise = timestep = '
                        'trainable = max_grad_norm = None')
        source = ('def run_hunyuanvideo_training():\n    '+declarations+'\n'
                  '    def train_step(global_step):\n'
                  + ''.join('        '+line+'\n' for line in BOUNDARIES)
                  + '    train_step(1)\n    train_step(2)\n')
        scope = dict(torch=torch,optimizer=optimizer,_hunyuan_capture=observer)
        exec(instrument_source(source),scope)
        scope['run_hunyuanvideo_training']()
        self.assertEqual(trace,['zero','before','cat','inputs','forward','backward','preclip',
                                'clip','postclip','step','updated','zero','cat','backward','clip','step'])


if __name__ == '__main__':
    unittest.main()
