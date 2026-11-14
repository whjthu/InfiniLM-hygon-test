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
        instrument_source(original, replay=True)
        with self.assertRaisesRegex(ValueError, 'boundaries changed'):
            instrument_source(original.replace(' / num_timesteps', ' / (num_timesteps + 1)'), replay=True)
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


    def test_replay_substitutes_before_constructing_only_first_step_inputs(self):
        calls = []

        class NativeTimestep:
            def view(self, *shape):
                return 8

        class FixedTimestep:
            def view(self, *shape):
                return 12

        source = '''def run_hunyuanvideo_training():
    def train_step(global_step):
        optimizer.zero_grad(set_to_none=True)
        noise = draw('noise', 3)
        timestep = draw('time', NativeTimestep())
        scale = timestep.view(-1, 1, 1, 1, 1) / num_timesteps
        latents_noised = noise * scale
        target = noise
        hidden_states = torch.cat((latents_noised, cond), dim=1)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
    train_step(1)
    train_step(2)
'''
        def draw(name, value):
            calls.append(name)
            return value

        observer = SimpleNamespace(**{phase: (lambda *args: None) for phase, _ in BOUNDARIES.values()})
        observer.replay = lambda noise, timestep: (7, FixedTimestep())
        observer.inputs = lambda noise, timestep, hidden, target: calls.append(('inputs', noise, hidden))
        no_op = lambda *args, **kwargs: None
        namespace = dict.fromkeys(('transformer', 'latents', 'cond', 'text_emb', 'text_mask',
                                   'byt5_text_states', 'byt5_text_mask', 'prediction',
                                   'trainable', 'max_grad_norm'))
        namespace.update(_hunyuan_capture=observer, draw=draw, num_timesteps=4,
                         NativeTimestep=NativeTimestep,
                         optimizer=SimpleNamespace(zero_grad=no_op, step=no_op),
                         torch=SimpleNamespace(
                             cat=lambda values, **_: calls.append(('cat', values[0])) or values[0],
                             nn=SimpleNamespace(
                                 functional=SimpleNamespace(mse_loss=lambda *_: SimpleNamespace(backward=no_op)),
                                 utils=SimpleNamespace(clip_grad_norm_=no_op))))
        exec(instrument_source(source, replay=True), namespace)
        namespace['run_hunyuanvideo_training']()
        self.assertEqual(calls, ['noise', 'time', ('cat', 21), ('inputs', 7, 21),
                                 'noise', 'time', ('cat', 6)])


if __name__ == '__main__':
    unittest.main()
